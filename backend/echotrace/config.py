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
CAPTURE_SR = 48_000          # mic sample rate (Hz); GCC-PHAT runs at this rate
CAPTURE_CHANNELS = 2
BLOCK_SIZE = 2_400           # frames per sounddevice callback (50 ms at 48 kHz)

# ---------------------------------------------------------------- tagging (PANNs)
MODEL_SR = 32_000            # CNN14 input rate
WINDOW_S = 1.0               # rolling analysis window (try 2.0 if probabilities are too low)
HOP_S = 0.5                  # inference every HOP_S seconds
TOP_K = 5
