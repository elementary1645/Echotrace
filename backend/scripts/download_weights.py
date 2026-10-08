"""Fetch the PANNs CNN14 weights + AudioSet label CSV into ~/panns_data without wget.

Run this BEFORE anything imports panns_inference (its config module shells out to wget
at import time when the CSV is missing). Idempotent: files that already pass the size
check are skipped. Dropped connections resume from <file>.part (also across re-runs).

    python scripts/download_weights.py          # download what's missing
    python scripts/download_weights.py --check  # only verify, exit 1 if anything is missing
"""
import argparse
import csv
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from echotrace import config  # noqa: E402

RETRIES = 10
CHUNK = 1 << 20          # 1 MiB
PRINT_EVERY = 16 << 20   # progress line every 16 MiB


def is_ok(path: Path, min_bytes: int) -> bool:
    return path.is_file() and path.stat().st_size >= min_bytes


def download(url: str, dest: Path, min_bytes: int) -> None:
    """Download to <dest>.part, resuming with HTTP Range after drops (Zenodo supports it)."""
    part = dest.with_name(dest.name + ".part")
    print(f"  GET {url}\n  -> {dest}")
    total = None
    for attempt in range(1, RETRIES + 1):
        have = part.stat().st_size if part.exists() else 0
        if total and have >= total:
            break
        req = urllib.request.Request(url, headers={"User-Agent": "echotrace-setup"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                resumed = bool(have) and resp.status == 206
                if not resumed:
                    have = 0
                length = resp.headers.get("Content-Length")
                total = int(length) + have if length else total
                if resumed:
                    print(f"  resuming at {have/1e6:.1f} MB")
                t0, done, last_print = time.time(), have, 0
                with open(part, "ab" if resumed else "wb") as f:
                    while chunk := resp.read(CHUNK):
                        f.write(chunk)
                        done += len(chunk)
                        if total and (done - last_print >= PRINT_EVERY or done == total):
                            last_print = done
                            rate = (done - have) / max(time.time() - t0, 1e-6) / 1e6
                            print(f"  {done/1e6:8.1f} / {total/1e6:.1f} MB  ({rate:.2f} MB/s)", flush=True)
        except Exception as exc:  # network hiccup: keep the .part, the next attempt resumes
            print(f"  attempt {attempt}/{RETRIES} failed: {exc}")
            time.sleep(min(2 * attempt, 10))
            continue
        size = part.stat().st_size
        if (total and size < total) or size < min_bytes:
            print(f"  attempt {attempt}/{RETRIES}: connection closed early at {size/1e6:.1f} MB; resuming")
            continue
        break
    size = part.stat().st_size if part.exists() else 0
    if size < min_bytes or (total and size != total):
        raise SystemExit(f"FAILED to download {url} after {RETRIES} attempts "
                         f"({size:,} bytes kept in {part}; re-run to resume)")
    part.replace(dest)


def check_csv(path: Path) -> int:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    n = len(rows) - 1  # header: index,mid,display_name
    if n != config.NUM_CLASSES:
        raise SystemExit(f"{path} has {n} classes, expected {config.NUM_CLASSES}; delete it and re-run")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only verify files, do not download")
    args = ap.parse_args()

    config.PANNS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = [
        ("labels CSV", config.LABELS_CSV_URL, config.LABELS_CSV, config.LABELS_CSV_MIN_BYTES),
        ("CNN14 weights", config.WEIGHTS_URL, config.WEIGHTS_FILE, config.WEIGHTS_MIN_BYTES),
    ]
    missing = False
    for name, url, dest, min_bytes in files:
        if is_ok(dest, min_bytes):
            print(f"[ok]   {name}: {dest} ({dest.stat().st_size:,} bytes)")
            continue
        if dest.exists():
            print(f"[bad]  {name}: {dest} is only {dest.stat().st_size:,} bytes -> replacing")
            if not args.check:
                dest.unlink()
        if args.check:
            print(f"[miss] {name}: {dest}")
            missing = True
            continue
        print(f"[get]  {name}")
        download(url, dest, min_bytes)
        print(f"[ok]   {name}: {dest} ({dest.stat().st_size:,} bytes)")

    if config.LABELS_CSV.exists():
        print(f"[ok]   labels CSV has {check_csv(config.LABELS_CSV)} classes")
    if missing:
        sys.exit(1)
    print("All PANNs files present. panns_inference will not need wget.")


if __name__ == "__main__":
    main()
