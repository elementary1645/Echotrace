"""PANNs CNN14 tagger: 48 kHz stereo blocks -> rolling window -> 32 kHz mono -> clipwise probs.

The Tagger interface is just `infer(mono_model_sr) -> probs (527,)`, so another model can be
swapped in behind it.
"""
import time
from math import gcd

import numpy as np
from scipy.signal import resample_poly

from echotrace import config


def downmix(x: np.ndarray) -> np.ndarray:
    """(n, ch) or (n,) float -> (n,) mono float32."""
    return (x.mean(axis=1) if x.ndim == 2 else x).astype(np.float32)


def resample(x: np.ndarray, sr_in: int = config.CAPTURE_SR, sr_out: int = config.MODEL_SR) -> np.ndarray:
    """Polyphase resample of a 1-D signal (48k -> 32k is up 2 / down 3)."""
    if sr_in == sr_out:
        return x.astype(np.float32)
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


class RollingWindow:
    """Keeps the last `window_s` of capture-rate stereo audio and signals every `hop_s`.

    The whole window is resampled at inference time (not block by block), so there are no
    filter edge artifacts at block boundaries. The stereo copy is kept for localization.
    """

    def __init__(self, sr: int = config.CAPTURE_SR, channels: int = config.CAPTURE_CHANNELS,
                 window_s: float = config.WINDOW_S, hop_s: float = config.HOP_S):
        self.sr, self.channels = sr, channels
        self.win = int(round(window_s * sr))
        self.hop = int(round(hop_s * sr))
        self.buf = np.zeros((self.win, channels), np.float32)
        self.since_hop = 0
        self.total = 0   # frames pushed so far

    def push(self, block: np.ndarray) -> int:
        """Append a (n, ch) block. Returns how many hop boundaries were crossed."""
        n = len(block)
        if n >= self.win:
            self.buf[:] = block[-self.win:]
        else:
            self.buf = np.roll(self.buf, -n, axis=0)
            self.buf[-n:] = block
        self.total += n
        self.since_hop += n
        hops, self.since_hop = divmod(self.since_hop, self.hop)
        return hops

    def stereo(self) -> np.ndarray:
        return self.buf.copy()

    def last(self, seconds: float) -> np.ndarray:
        n = min(self.win, int(round(seconds * self.sr)))
        return self.buf[-n:]


class PannsTagger:
    """CNN14 via panns_inference. device: 'auto' (CUDA if available), 'cuda' or 'cpu'."""

    def __init__(self, device: str = "auto", warmup_s: float = config.WINDOW_S,
                 gain_db: float = config.TAGGER_GAIN_DB):
        self.gain = 10 ** (gain_db / 20)
        missing = [p for p, n in ((config.WEIGHTS_FILE, config.WEIGHTS_MIN_BYTES),
                                  (config.LABELS_CSV, config.LABELS_CSV_MIN_BYTES))
                   if not p.is_file() or p.stat().st_size < n]
        if missing:
            # panns_inference would shell out to wget (absent on Windows) on import
            raise FileNotFoundError(f"Missing PANNs files {[str(p) for p in missing]}; "
                                    "run: python scripts/download_weights.py")
        import torch
        from panns_inference import AudioTagging   # import only after the files exist

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = AudioTagging(checkpoint_path=str(config.WEIGHTS_FILE), device=device)
        self.infer(np.zeros(int(warmup_s * config.MODEL_SR), np.float32))   # warm-up (CUDA init, kernels)

    def infer(self, mono: np.ndarray) -> np.ndarray:
        """mono float32 at MODEL_SR -> clipwise probabilities (527,)."""
        clipwise, _ = self.model.inference(mono[None, :].astype(np.float32))
        return clipwise[0]

    def infer_window(self, stereo_capture: np.ndarray) -> tuple[np.ndarray, float]:
        """(n, ch) at CAPTURE_SR -> (probs, inference_ms incl. downmix + resample)."""
        t0 = time.perf_counter()
        mono = resample(downmix(stereo_capture))
        if self.gain != 1.0:
            mono = np.clip(mono * self.gain, -1.0, 1.0)
        probs = self.infer(mono)
        return probs, (time.perf_counter() - t0) * 1000.0
