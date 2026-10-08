"""Record N seconds of 48 kHz stereo from one device to recordings/<name>.wav (24-bit PCM).

This is the ONLY place EchoTrace stores raw audio, and only when you run it explicitly.

    python scripts/record_clip.py --seconds 30 --name demo_sequence_01
    python scripts/record_clip.py --device "Microphone Array 1" --seconds 10 --name lone_door_slam
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echotrace import config  # noqa: E402
from echotrace.audio.sources import LiveMicSource, rms_db  # noqa: E402

COUNTDOWN_S = 3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=None, help=f"index or name substring (default '{config.DEFAULT_DEVICE}')")
    ap.add_argument("--seconds", type=float, required=True)
    ap.add_argument("--name", default=None, help="file name without .wav (default clip_<timestamp>)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = ap.parse_args()

    name = args.name or f"clip_{datetime.now():%Y%m%d_%H%M%S}"
    out = config.RECORDINGS_DIR / f"{name}.wav"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists (use --force to overwrite)")
    config.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    src = LiveMicSource(args.device)
    print(f"Recording {args.seconds:.0f} s from {src.describe()} -> {out}")
    for k in range(COUNTDOWN_S, 0, -1):
        print(f"  starting in {k}...")
        time.sleep(1)
    print("  >>> RECORDING")

    need = int(args.seconds * src.sr)
    blocks, got, last_print = [], 0, 0
    src.start()
    try:
        while got < need:
            b = src.read(timeout=2.0)
            if b is None:
                raise SystemExit("no audio for 2 s (device busy?)")
            blocks.append(b.data)
            got += len(b.data)
            if got - last_print >= src.sr:
                last_print = got
                l, r = rms_db(b.data)
                print(f"  {got / src.sr:5.1f} s  L {l:6.1f} dB  R {r:6.1f} dB")
    except KeyboardInterrupt:
        print("  stopped early")
    finally:
        src.stop()

    x = np.concatenate(blocks)[:need]
    sf.write(out, x, src.sr, subtype="PCM_24")
    l, r = x[:, 0].astype(float), x[:, 1].astype(float)
    corr = np.corrcoef(l, r)[0, 1] if l.std() > 0 and r.std() > 0 else float("nan")
    print(f"Saved {out} ({len(x) / src.sr:.1f} s, {src.sr} Hz, 2 ch) | RMS L {rms_db(x)[0]:.1f} R {rms_db(x)[1]:.1f} dBFS"
          f" | corr(L,R) {corr:.3f}" + ("  WARNING: channels ~identical, localization will be OFF" if corr > 0.999 else ""))
    if src.dropped or src.overflows:
        print(f"WARNING: dropped blocks {src.dropped}, driver overflows {src.overflows}")


if __name__ == "__main__":
    main()
