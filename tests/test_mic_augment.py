import numpy as np

from inherent.data.mic_augment import apply_device_mic_coloration, mic_augment_enabled


def test_mic_augment_changes_audio_but_stays_finite():
    rng = np.random.default_rng(0)
    wave = (0.2 * np.sin(np.linspace(0, 12 * np.pi, 1600))).astype(np.float32)
    colored = apply_device_mic_coloration(wave, 16000, seed=42)
    assert colored.shape == wave.shape
    assert np.isfinite(colored).all()
    assert not np.allclose(colored, wave)


def test_mic_augment_disabled_by_env(monkeypatch):
    monkeypatch.setenv("INHERENT_SYNTHETIC_MIC_AUGMENT", "0")
    assert mic_augment_enabled() is False
