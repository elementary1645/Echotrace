"""EchoTrace configuration: every tunable lives here.

Stdlib-only on purpose, so scripts (e.g. download_weights.py) can import it before
torch / panns_inference are installed or imported.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = BACKEND_DIR.parent
RECORDINGS_DIR = REPO_DIR / "recordings"
LABELS_DIR = RECORDINGS_DIR / "labels"
DOCS_DIR = REPO_DIR / "docs"
CALIBRATION_FILE = BACKEND_DIR / "calibration.json"   # local, git-ignored

# ---------------------------------------------------------------- PANNs model files
# panns_inference hard-codes ~/panns_data and shells out to wget if files are missing.
PANNS_DATA_DIR = Path.home() / "panns_data"
WEIGHTS_FILE = PANNS_DATA_DIR / "Cnn14_mAP=0.431.pth"
LABELS_CSV = PANNS_DATA_DIR / "class_labels_indices.csv"
WEIGHTS_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
LABELS_CSV_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
WEIGHTS_MIN_BYTES = 300_000_000   # panns_inference itself re-downloads if the file is < 3e8 bytes
LABELS_CSV_MIN_BYTES = 10_000     # ~14 KB, 527 classes + header
NUM_CLASSES = 527

# ---------------------------------------------------------------- audio capture
# Laptop Intel SST array: the raw WDM-KS endpoints are the only true-stereo path
# (WASAPI/MME apply voice processing that merges channels). They accept 48 kHz only
# (96 kHz probed 2026-10-08: rejected), so localization also runs at 48 kHz.
DEFAULT_DEVICE = "Microphone Array 2"            # name substring (indices shift with Bluetooth)
FALLBACK_DEVICES = ["Microphone Array 1"]
HOSTAPI_PREFERENCE = ["Windows WDM-KS", "Windows WASAPI", "MME", "Windows DirectSound"]
CAPTURE_SR = 48_000          # mic sample rate (Hz); GCC-PHAT runs at this rate
CAPTURE_CHANNELS = 2
BLOCK_SIZE = 2_400           # frames per sounddevice callback (50 ms at 48 kHz)
AUDIO_QUEUE_BLOCKS = 200     # ~10 s of capture buffered before blocks are dropped
RMS_FLOOR_DB = -120.0        # dBFS reported for digital silence

# ---------------------------------------------------------------- tagging (PANNs)
MODEL_SR = 32_000            # CNN14 input rate (48k -> 32k = resample_poly up 2, down 3)
WINDOW_S = 1.0               # rolling analysis window (try 2.0 if probabilities are too low)
HOP_S = 0.5                  # inference every HOP_S seconds
TOP_K = 5
# Raw WDM-KS array has no AGC (quiet room ~ -56 dBFS); CNN14 was trained on louder YouTube
# audio. Digital gain applied to the tagger input only (GCC-PHAT is level-invariant).
TAGGER_GAIN_DB = 0.0

# ---------------------------------------------------------------- categories
# Category score = max over member labels (AudioSet display names, resolved by name at
# startup; a missing name is a hard error). Starting points, to be tuned from live_topk.
CATEGORIES = {
    "FOOTSTEPS": ["Walk, footsteps", "Run", "Shuffle"],
    "IMPACT":    ["Thump, thud", "Slam", "Bang", "Smash, crash", "Knock"],
    "DISTRESS":  ["Screaming", "Yell", "Crying, sobbing", "Shout", "Groan", "Whimper", "Wail, moan"],
    "ALARM":     ["Smoke detector, smoke alarm", "Fire alarm", "Alarm", "Siren", "Buzzer", "Alarm clock"],
    "GLASS":     ["Glass", "Shatter", "Breaking", "Chink, clink"],
    "DOOR":      ["Door", "Sliding door", "Cupboard open or close"],
}

# ---------------------------------------------------------------- detection (Phase 2 debounce)
ON_THRESHOLD = 0.30          # category score that starts an event
OFF_THRESHOLD = 0.15         # score below which an event ends

# ---------------------------------------------------------------- localization (Phase 2)
MIC_SPACING_M = 0.065        # measured between the two mic holes on the bezel
SPEED_OF_SOUND = 343.0       # m/s at ~20 C
# max |tau| = d / c = 0.065 / 343 = 190 us = 9.1 samples at 48 kHz -> sub-sample peak needed:
GCC_INTERP = 8               # upsample the cross-correlation by zero-padded IFFT, then
                             # parabolic interpolation around the peak (~1/16 sample)
CENTRE_HALF_ANGLE_DEG = 20.0 # |angle| < this -> CENTRE (= +/-3.1 samples at d = 6.5 cm)
