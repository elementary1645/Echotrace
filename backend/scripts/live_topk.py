"""Live PANNs check: top-5 AudioSet labels, category scores, inference time and a per-channel
RMS meter, every hop. Ctrl+C prints a summary (per category, top labels, per test segment).

    python scripts/live_topk.py                       # default device (config.DEFAULT_DEVICE)
    python scripts/live_topk.py --guided --log        # walk through the Phase 1 test script, save CSV
    python scripts/live_topk.py --device "Microphone Array 1" --window 2.0 --cpu

Segment keys (tag what you are doing; shown in the summary + CSV):
    0 background  1 footsteps  2 impact  3 distress  4 alarm  5 glass  6 knock  7 chatter
    space = untagged    n = skip to next step (--guided)
"""
import argparse
import csv
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echotrace import config  # noqa: E402
from echotrace.audio.sources import LiveMicSource, rms_db  # noqa: E402
from echotrace.tagging.categories import CategoryMapper  # noqa: E402
from echotrace.tagging.panns_tagger import PannsTagger, RollingWindow  # noqa: E402

try:
    import msvcrt   # Windows: non-blocking key presses
except ImportError:   # pragma: no cover
    msvcrt = None

KEYS = {"0": "BACKGROUND", "1": "FOOTSTEPS", "2": "IMPACT", "3": "DISTRESS", "4": "ALARM",
        "5": "GLASS", "6": "KNOCK", "7": "CHATTER", " ": None}

# The Phase 1 test script: (segment, seconds, instruction). Distances are from the laptop's
# top bezel (where the mics are). Keep the laptop where it will be during the demo.
GUIDED_STEPS = [
    ("BACKGROUND", 20, "Stay quiet. Let the room's normal background noise play. Don't touch the laptop."),
    ("FOOTSTEPS", 20, "Walk with shoes on, firm normal steps, 1-2 m from the laptop: left -> right -> back."),
    ("IMPACT", 20, "Fist-thud a table ~1 m away (or drop a closed book flat) every 2-3 s. Also a door slam if one is near."),
    ("DISTRESS", 20, "Play a scream/shout clip from your phone at ~70% volume, phone 1 m away, speaker facing the laptop."),
    ("ALARM", 20, "Play a phone alarm or smoke-alarm clip, ~70% volume, 1 m away."),
    ("GLASS", 20, "Play a glass-breaking clip from your phone, ~70% volume, 1 m away (repeat it)."),
    ("KNOCK", 20, "Knuckle-knock a door or table ~1 m away: 3 knocks, pause 2 s, repeat."),
    ("CHATTER", 20, "Talk normally (or play people chatting) ~1 m away. This is a NEGATIVE: nothing should alert."),
]
GET_READY_S = 5
METER_LO_DB, METER_HI_DB, METER_W = -80.0, -20.0, 10


def meter(db: float) -> str:
    n = int(np.clip((db - METER_LO_DB) / (METER_HI_DB - METER_LO_DB), 0, 1) * METER_W)
    return "#" * n + "." * (METER_W - n)


class Guide:
    """Advances through GUIDED_STEPS on the session clock; None segment while getting ready."""

    def __init__(self, steps):
        self.plan, t = [], 0.0
        for seg, secs, text in steps:
            self.plan.append((t, t + GET_READY_S, None, f"GET READY -> {seg}: {text}"))
            self.plan.append((t + GET_READY_S, t + GET_READY_S + secs, seg, text))
            t += GET_READY_S + secs
        self.offset = 0.0
        self.last_announced = -1

    def at(self, t: float):
        t += self.offset
        for k, (a, b, seg, text) in enumerate(self.plan):
            if a <= t < b:
                if k != self.last_announced:
                    self.last_announced = k
                    print("\n" + "=" * 100 + f"\n>>> {text}" + (f"  ({b - a:.0f} s)" if seg else "") + "\n" + "=" * 100)
                return seg, b - t
        return "DONE", 0.0

    def skip(self, t: float):
        t += self.offset
        for a, b, _, _ in self.plan:
            if a <= t < b:
                self.offset += b - t
                return


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=None, help=f"index or name substring (default '{config.DEFAULT_DEVICE}')")
    ap.add_argument("--window", type=float, default=config.WINDOW_S, help="analysis window, s")
    ap.add_argument("--hop", type=float, default=config.HOP_S, help="hop between inferences, s")
    ap.add_argument("--cpu", action="store_true", help="force CPU inference")
    ap.add_argument("--gain", type=float, default=config.TAGGER_GAIN_DB, help="digital gain (dB) on the tagger input")
    ap.add_argument("--threshold", type=float, default=config.ON_THRESHOLD, help="'fired' threshold for the summary")
    ap.add_argument("--guided", action="store_true", help="walk through the Phase 1 test script")
    ap.add_argument("--seconds", type=float, default=None, help="stop automatically after N s")
    ap.add_argument("--log", action="store_true", help="save every hop (incl. all 527 probs) to recordings/topk_*.csv")
    args = ap.parse_args()

    mapper = CategoryMapper()
    tagger = PannsTagger(device="cpu" if args.cpu else "auto", gain_db=args.gain)
    src = LiveMicSource(args.device)
    win = RollingWindow(window_s=args.window, hop_s=args.hop)
    cats = list(config.CATEGORIES)

    print("=" * 100)
    print(f"EchoTrace live_topk | source LIVE | device {src.describe()}")
    print(f"model CNN14 on {tagger.device.upper()} | window {args.window:.2f} s | hop {args.hop:.2f} s | "
          f"model SR {config.MODEL_SR} | gain {args.gain:+.0f} dB | 'fired' threshold {args.threshold:.2f}")
    print("Keys: 0 bg 1 footsteps 2 impact 3 distress 4 alarm 5 glass 6 knock 7 chatter, space untagged"
          + (", n next step" if args.guided else "") + ". Ctrl+C = stop + summary.")
    print("=" * 100)

    writer, log_f, log_path = None, None, None
    if args.log:
        config.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = config.RECORDINGS_DIR / f"topk_{datetime.now():%Y%m%d_%H%M%S}.csv"
        log_f = open(log_path, "w", newline="", encoding="utf-8")
        writer = csv.writer(log_f)
        writer.writerow(["t", "wall", "segment", "rms_l_db", "rms_r_db", "inf_ms"]
                        + [f"cat_{c}" for c in cats]
                        + [f"top{k}_{x}" for k in range(1, config.TOP_K + 1) for x in ("label", "p")]
                        + mapper.label_names)

    guide = Guide(GUIDED_STEPS) if args.guided else None
    segment = None
    hist_probs, hist_seg, inf_ms_all = [], [], []
    skipped_hops = 0
    src.start()
    t0 = time.monotonic()
    try:
        while True:
            block = src.read(timeout=2.0)
            if block is None:
                print("!! no audio for 2 s (device busy? another app holding WDM-KS?)")
                continue
            t = block.t_capture - t0
            if args.seconds and t >= args.seconds:
                break
            while msvcrt and msvcrt.kbhit():
                k = msvcrt.getwch().lower()
                if k in KEYS and not guide:
                    segment = KEYS[k]
                elif k == "n" and guide:
                    guide.skip(t)
            if guide:
                segment, left = guide.at(t)
                if segment == "DONE":
                    break
            hops = win.push(block.data)
            if not hops:
                continue
            skipped_hops += hops - 1

            stereo = win.stereo()
            probs, inf_ms = tagger.infer_window(stereo)
            rms = rms_db(win.last(args.hop))
            scores = mapper.scores(probs)
            top = mapper.top_k(probs)
            hist_probs.append(probs)
            hist_seg.append(segment)
            inf_ms_all.append(inf_ms)

            tag = f"[{segment or '-'}{f' {left:2.0f}s' if guide and segment else ''}]"
            behind = src.q.qsize()
            print(f"{tag:<16} t={t:7.1f}  L {rms[0]:6.1f} {meter(rms[0])}  R {rms[1]:6.1f} {meter(rms[1])}  "
                  f"inf {inf_ms:5.1f} ms" + (f"  queue {behind}" if behind > 2 else ""))
            print("   top: " + " | ".join(f"{lab} {p:.2f}" for lab, p in top))
            print("   cat: " + "  ".join(f"{c}{'*' if scores[c][0] >= args.threshold else ' '}{scores[c][0]:.2f}"
                                         for c in cats))
            if writer:
                writer.writerow([f"{t:.3f}", datetime.now().isoformat(timespec="milliseconds"), segment or "",
                                 f"{rms[0]:.1f}", f"{rms[1]:.1f}", f"{inf_ms:.1f}"]
                                + [f"{scores[c][0]:.4f}" for c in cats]
                                + [v for lab, p in top for v in (lab, f"{p:.4f}")]
                                + [f"{p:.4f}" for p in probs])
                log_f.flush()
    except KeyboardInterrupt:
        pass
    finally:
        src.stop()
        if log_f:
            log_f.close()

    summarize(np.array(hist_probs), hist_seg, inf_ms_all, mapper, cats, args.threshold,
              dropped=src.dropped, overflows=src.overflows, skipped=skipped_hops)
    if log_path:
        print(f"\nCSV log: {log_path}")


def summarize(P, segs, inf_ms, mapper, cats, thr, dropped, overflows, skipped) -> None:
    print("\n" + "=" * 100 + "\nSUMMARY")
    if len(P) == 0:
        print("no hops processed")
        return
    names = mapper.label_names
    ms = np.array(inf_ms)
    print(f"hops {len(P)} | inference mean {ms.mean():.1f} ms, p95 {np.percentile(ms, 95):.1f} ms | "
          f"dropped blocks {dropped}, driver overflows {overflows}, skipped hops {skipped}")

    cat_scores = {c: P[:, mapper.indices[c]].max(axis=1) for c in cats}
    print(f"\nCategory     max    mean   hops>={thr:.2f}   label at max")
    for c in cats:
        s = cat_scores[c]
        i = int(np.argmax(s))
        best = names[mapper.indices[c][int(np.argmax(P[i, mapper.indices[c]]))]]
        n = int((s >= thr).sum())
        print(f"{c:<11} {s.max():5.2f}  {s.mean():5.2f}   {n:4d} ({100 * n / len(s):4.1f}%)   {best}")

    in_top = Counter()
    for row in P:
        in_top.update(np.argsort(row)[::-1][:config.TOP_K].tolist())
    print(f"\nTop 10 labels overall (hops in top-{config.TOP_K}, max p, mean p):")
    for j, n in in_top.most_common(10):
        print(f"  {names[j]:<40} {n:4d}  max {P[:, j].max():.2f}  mean {P[:, j].mean():.2f}")

    print("\nCategory member labels (max p over the session):")
    for c in cats:
        print(f"  {c:<10} " + ", ".join(f"{names[j]} {P[:, j].max():.2f}" for j in mapper.indices[c]))

    seg_names = [s for s in dict.fromkeys(segs) if s]
    if seg_names:
        print("\nPer test segment (top 5 labels by MAX p, then category max / hops>=thr):")
        segs = np.array([s or "" for s in segs])
        for s in seg_names:
            Q = P[segs == s]
            top = np.argsort(Q.max(axis=0))[::-1][:5]
            print(f"  {s:<11} ({len(Q)} hops): " + " | ".join(f"{names[j]} {Q[:, j].max():.2f}" for j in top))
            print(" " * 15 + "  ".join(
                f"{c} {Q[:, mapper.indices[c]].max():.2f}/{int((Q[:, mapper.indices[c]].max(axis=1) >= thr).sum())}"
                for c in cats))
    print("=" * 100)


if __name__ == "__main__":
    main()
