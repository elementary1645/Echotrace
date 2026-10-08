from echotrace.audio.sources import pick_device, rms_db

import numpy as np

HOSTAPIS = [{"name": "MME"}, {"name": "Windows WASAPI"}, {"name": "Windows WDM-KS"}]
DEVICES = [
    {"name": "Microphone Array (2- Intel Sma", "hostapi": 0, "max_input_channels": 4},
    {"name": "Microphone Array (2- Intel Smart Sound)", "hostapi": 1, "max_input_channels": 2},
    {"name": "Microphone Array 1 ()", "hostapi": 2, "max_input_channels": 2},
    {"name": "Microphone Array 2 ()", "hostapi": 2, "max_input_channels": 2},
    {"name": "PC Speaker", "hostapi": 2, "max_input_channels": 0},
    {"name": "Headset", "hostapi": 1, "max_input_channels": 1},
]


def ok(i):
    return True


def test_pick_by_exact_substring_and_index():
    assert pick_device("Microphone Array 2", DEVICES, HOSTAPIS, ok) == 3
    assert pick_device("microphone array 1", DEVICES, HOSTAPIS, ok) == 2
    assert pick_device("3", DEVICES, HOSTAPIS, ok) == 3
    assert pick_device(4, DEVICES, HOSTAPIS, ok) is None          # output-only device
    assert pick_device("nonexistent", DEVICES, HOSTAPIS, ok) is None


def test_pick_prefers_supported_then_hostapi_order():
    # "Microphone Array" matches 0-3; all supported -> WDM-KS first (config.HOSTAPI_PREFERENCE)
    assert pick_device("Microphone Array", DEVICES, HOSTAPIS, ok) in (2, 3)
    # if the WDM-KS ones can't do 48k x 2ch, fall back to the next host API
    assert pick_device("Microphone Array", DEVICES, HOSTAPIS, lambda i: i in (0, 1)) == 1


def test_rms_db():
    x = np.stack([np.full(100, 0.1), np.zeros(100)], axis=1)
    l, r = rms_db(x)
    assert abs(l - (-20.0)) < 1e-6
    assert r == -120.0
