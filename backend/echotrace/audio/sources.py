"""Audio sources. Every source yields AudioBlock(data (n, 2) float32 @ CAPTURE_SR, t_capture).

Phase 1: LiveMicSource (+ device selection by name). WavReplaySource / SimSource: Phase 2.
"""
import queue
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from echotrace import config


@dataclass
class AudioBlock:
    data: np.ndarray      # (n, 2) float32 at CAPTURE_SR (mono devices are duplicated to 2 ch)
    t_capture: float      # time.monotonic() when the block arrived from the driver


# ---------------------------------------------------------------- device selection
def pick_device(spec: str | int, devices: list[dict], hostapis: list[dict],
                supports: Callable[[int], bool]) -> int | None:
    """Return the device index for `spec` (index or case-insensitive name substring), or None.

    Among name matches prefer: accepts CAPTURE_SR x CAPTURE_CHANNELS, then host API order in
    config.HOSTAPI_PREFERENCE, then more input channels.
    """
    if isinstance(spec, int) or str(spec).strip().isdigit():
        i = int(spec)
        return i if 0 <= i < len(devices) and devices[i]["max_input_channels"] > 0 else None
    needle = str(spec).lower()
    matches = [i for i, d in enumerate(devices)
               if d["max_input_channels"] > 0 and needle in d["name"].lower()]
    if not matches:
        return None

    def rank(i: int):
        api = hostapis[devices[i]["hostapi"]]["name"]
        api_rank = config.HOSTAPI_PREFERENCE.index(api) if api in config.HOSTAPI_PREFERENCE else 99
        return (not supports(i), api_rank, -devices[i]["max_input_channels"])

    return min(matches, key=rank)


def resolve_device(spec: str | int | None = None) -> tuple[int, dict, str]:
    """spec None -> config.DEFAULT_DEVICE then FALLBACK_DEVICES. Returns (index, info, host API)."""
    import sounddevice as sd
    devices, hostapis = list(sd.query_devices()), list(sd.query_hostapis())

    def supports(i: int) -> bool:
        try:
            sd.check_input_settings(device=i, channels=min(config.CAPTURE_CHANNELS, devices[i]["max_input_channels"]),
                                    samplerate=config.CAPTURE_SR)
            return True
        except Exception:
            return False

    specs = [spec] if spec is not None else [config.DEFAULT_DEVICE, *config.FALLBACK_DEVICES]
    for s in specs:
        i = pick_device(s, devices, hostapis, supports)
        if i is not None:
            return i, devices[i], hostapis[devices[i]["hostapi"]]["name"]
    raise SystemExit(f"No input device matches {specs}. Run: python scripts/list_devices.py")


# ---------------------------------------------------------------- sources
class LiveMicSource:
    """sounddevice capture from ONE device (one clock), 48 kHz, 2 ch.

    The callback only copies the block into a queue; all processing happens elsewhere.
    """
    mode = "LIVE"

    def __init__(self, device: str | int | None = None, sr: int = config.CAPTURE_SR,
                 block_size: int = config.BLOCK_SIZE):
        self.index, info, self.hostapi = resolve_device(device)
        self.device_name = info["name"]
        self.sr = sr
        self.block_size = block_size
        self.channels = min(config.CAPTURE_CHANNELS, info["max_input_channels"])
        self.q: queue.Queue[AudioBlock] = queue.Queue(maxsize=config.AUDIO_QUEUE_BLOCKS)
        self.dropped = 0           # blocks lost because the consumer fell behind
        self.overflows = 0         # driver-reported input overflows
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        if status.input_overflow:
            self.overflows += 1
        data = indata.copy()
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        try:
            self.q.put_nowait(AudioBlock(data, time.monotonic()))
        except queue.Full:
            self.dropped += 1

    def start(self) -> "LiveMicSource":
        import sounddevice as sd
        self.stream = sd.InputStream(device=self.index, samplerate=self.sr, channels=self.channels,
                                     blocksize=self.block_size, dtype="float32", callback=self._callback)
        self.stream.start()
        return self

    def read(self, timeout: float = 1.0) -> AudioBlock | None:
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def describe(self) -> str:
        return f"#{self.index} '{self.device_name}' [{self.hostapi}] {self.sr} Hz x {self.channels} ch"


def rms_db(x: np.ndarray) -> list[float]:
    """Per-channel RMS in dBFS for an (n, ch) block."""
    r = np.sqrt(np.mean(np.square(x, dtype=np.float64), axis=0))
    return [max(config.RMS_FLOOR_DB, 20 * np.log10(v)) if v > 0 else config.RMS_FLOOR_DB for v in r]
