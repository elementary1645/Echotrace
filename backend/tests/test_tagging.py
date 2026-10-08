import numpy as np
import pytest
import soundfile as sf

from echotrace import config
from echotrace.tagging.categories import CategoryMapper, load_label_names, resolve
from echotrace.tagging.panns_tagger import RollingWindow, downmix, resample

have_csv = config.LABELS_CSV.is_file()
have_weights = config.WEIGHTS_FILE.is_file() and config.WEIGHTS_FILE.stat().st_size >= config.WEIGHTS_MIN_BYTES


# ---------------------------------------------------------------- categories
@pytest.mark.skipif(not have_csv, reason="run scripts/download_weights.py")
def test_all_category_names_resolve():
    names = load_label_names()
    assert len(names) == config.NUM_CLASSES
    idx = resolve(config.CATEGORIES, names)
    for cat, members in config.CATEGORIES.items():
        assert [names[i] for i in idx[cat]] == members


def test_unknown_name_fails_loudly_with_suggestion():
    names = ["Walk, footsteps", "Run", "Glass"]
    with pytest.raises(ValueError, match="Walk footsteps.*did you mean"):
        resolve({"FOOTSTEPS": ["Walk footsteps"]}, names)


def test_category_score_is_max_over_members():
    names = [f"L{i}" for i in range(6)]
    m = CategoryMapper({"A": ["L1", "L3"], "B": ["L5"]}, label_names=names)
    p = np.array([0.9, 0.2, 0.0, 0.6, 0.0, 0.1])
    s = m.scores(p)
    assert s["A"] == (pytest.approx(0.6), "L3")
    assert s["B"] == (pytest.approx(0.1), "L5")
    assert m.top_k(p, 2) == [("L0", pytest.approx(0.9)), ("L3", pytest.approx(0.6))]


# ---------------------------------------------------------------- resampling / windowing
def test_resample_48k_to_32k_shape_and_frequency():
    sr_in, f0 = config.CAPTURE_SR, 1000.0
    t = np.arange(sr_in) / sr_in
    y = resample(np.sin(2 * np.pi * f0 * t).astype(np.float32))
    assert y.dtype == np.float32
    assert len(y) == config.MODEL_SR                      # 1 s in -> 1 s out at 32 kHz
    spec = np.abs(np.fft.rfft(y))
    peak_hz = np.argmax(spec) * config.MODEL_SR / len(y)
    assert abs(peak_hz - f0) <= 1.0                       # pitch preserved


def test_downmix_stereo_to_mono():
    x = np.stack([np.ones(10), -np.ones(10) * 0.5], axis=1)
    assert np.allclose(downmix(x), 0.25)


def test_rolling_window_hops_and_content():
    w = RollingWindow(sr=1000, channels=2, window_s=1.0, hop_s=0.5)
    block = np.ones((100, 2), np.float32)
    hops = sum(w.push(block * (i + 1)) for i in range(12))   # 1.2 s pushed
    assert hops == 2
    assert w.stereo().shape == (1000, 2)
    assert w.stereo()[-1, 0] == 12 and w.stereo()[0, 0] == 3   # oldest kept block is #3


# ---------------------------------------------------------------- tagger smoke test
@pytest.mark.skipif(not have_weights, reason="run scripts/download_weights.py")
def test_tagger_smoke_on_generated_wav(tmp_path):
    from echotrace.tagging.panns_tagger import PannsTagger

    sr = config.CAPTURE_SR
    t = np.arange(int(1.0 * sr)) / sr
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    wav = tmp_path / "tone.wav"
    sf.write(wav, np.stack([tone, tone], axis=1), sr)       # 48 kHz stereo, like the mic
    x, file_sr = sf.read(wav, dtype="float32")
    assert file_sr == sr and x.shape == (sr, 2)

    tagger = PannsTagger(device="cpu")
    probs, ms = tagger.infer_window(x)
    assert probs.shape == (config.NUM_CLASSES,)
    assert np.all(np.isfinite(probs)) and probs.min() >= 0 and probs.max() <= 1
    assert ms > 0
    names = load_label_names()
    top = [names[i] for i in np.argsort(probs)[::-1][:10]]
    assert any(k in " ".join(top) for k in ("Sine wave", "Tone", "Hum", "Beep", "Music")), top
