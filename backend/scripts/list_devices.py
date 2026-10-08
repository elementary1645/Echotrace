"""List audio input devices and mark the ones usable for localization.

STEREO  = device reports >= 2 input channels
48k-2ch = sounddevice accepts 48 kHz x 2 channels on it (what EchoTrace captures)

Windows lists each physical mic once per host API (MME, DirectSound, WASAPI, WDM-KS).
For localization prefer the raw WDM-KS array endpoint ("Microphone Array N"): WASAPI/MME
entries of a laptop array often pass through Windows voice processing (echo cancellation,
noise suppression) which merges the channels into identical mono. EchoTrace detects
identical channels at runtime and turns localization OFF.

    python scripts/list_devices.py          # input devices only
    python scripts/list_devices.py --all    # include output-only devices
"""
import argparse
import sys
from pathlib import Path

import sounddevice as sd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echotrace import config  # noqa: E402


LOOPBACK_HINTS = ("stereo mix", "speaker", "what u hear", "loopback")


def supports(dev_index: int, channels: int, sr: int) -> bool:
    try:
        sd.check_input_settings(device=dev_index, channels=channels, samplerate=sr)
        return True
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="also list output-only devices")
    args = ap.parse_args()

    hostapis = sd.query_hostapis()
    default_in = sd.default.device[0]
    sr, ch = config.CAPTURE_SR, config.CAPTURE_CHANNELS

    print(f"{'idx':>3}  {'host API':<22} {'in':>2} {'out':>3} {'def SR':>7}  flags              name")
    print("-" * 100)
    candidates = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] < 1 and not args.all:
            continue
        flags = []
        if i == default_in:
            flags.append("DEFAULT")
        if d["max_input_channels"] >= 2:
            flags.append("STEREO")
            if supports(i, ch, sr):
                flags.append(f"{sr//1000}k-{ch}ch")
                candidates.append((i, d["name"], hostapis[d["hostapi"]]["name"]))
        api = hostapis[d["hostapi"]]["name"]
        if any(k in d["name"].lower() for k in LOOPBACK_HINTS):
            flags.append("LOOPBACK?")   # captures what the PC plays, not a microphone
            candidates = [c for c in candidates if c[0] != i]
        print(f"{i:>3}  {api:<22} {d['max_input_channels']:>2} {d['max_output_channels']:>3} "
              f"{int(d['default_samplerate']):>7}  {' '.join(flags):<18} {d['name']}")

    print()
    if candidates:
        print(f"Devices that accept {sr} Hz x {ch} ch (candidates for LIVE + localization):")
        for i, name, api in candidates:
            print(f"  --device {i:<3} {name}  [{api}]")
    else:
        print(f"No input device accepts {sr} Hz x {ch} ch. Classification will still work on a mono device;")
        print("localization will be OFF (zones = UNKNOWN).")
    print(f"\nDefault input device index: {default_in}")


if __name__ == "__main__":
    main()
