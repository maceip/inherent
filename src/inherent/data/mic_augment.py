"""Make clean TTS speech sound closer to on-device mic capture.

Applied after ffmpeg normalization (16 kHz mono). Uses band limiting, light
compression, and optional background noise at a randomized SNR — the same
ideas documented for directedness negatives (MUSAN/DEMAND) but without
requiring external noise corpora at synthesis time.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..features.frontend import SAMPLE_RATE

_DEFAULT_SNR_DB_RANGE = (10.0, 22.0)


def mic_augment_enabled() -> bool:
    raw = os.environ.get("INHERENT_SYNTHETIC_MIC_AUGMENT", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def parse_snr_db_range(raw: tuple[float, float] | list[float] | None) -> tuple[float, float]:
    if raw is None:
        return _DEFAULT_SNR_DB_RANGE
    if len(raw) != 2:
        raise ValueError(f"snr_db_range must have exactly two values, got {raw!r}")
    low, high = float(raw[0]), float(raw[1])
    if low > high:
        raise ValueError(f"snr_db_range low must be <= high, got {raw!r}")
    return low, high


def apply_device_mic_coloration(
    audio: np.ndarray,
    sample_rate: int,
    *,
    snr_db: float | None = None,
    snr_db_range: tuple[float, float] = _DEFAULT_SNR_DB_RANGE,
    seed: int | None = None,
) -> np.ndarray:
    """Return mono float32 audio with wearable/phone-like coloration."""
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"mic augment expects sample_rate={SAMPLE_RATE}, got {sample_rate}")
    wave = np.asarray(audio, dtype=np.float32).reshape(-1)
    if wave.size == 0:
        return wave

    rng = np.random.default_rng(seed)
    if snr_db is None:
        snr_db = float(rng.uniform(snr_db_range[0], snr_db_range[1]))

    wave = _bandlimit(wave, sample_rate, low_hz=180.0, high_hz=3600.0)
    wave = _soft_compress(wave, mix=float(rng.uniform(0.55, 0.85)))
    wave = _mix_pink_noise(wave, snr_db=snr_db, rng=rng)
    wave = _apply_short_room(wave, sample_rate, rng=rng)
    peak = float(np.max(np.abs(wave))) or 1.0
    if peak > 0.98:
        wave = wave * (0.95 / peak)
    return wave.astype(np.float32, copy=False)


def augment_wav_file(
    path: str | Path,
    *,
    snr_db: float | None = None,
    snr_db_range: tuple[float, float] = _DEFAULT_SNR_DB_RANGE,
    seed: int | None = None,
) -> Path:
    """Read a WAV, apply mic coloration in place, return the path."""
    import soundfile as sf

    target = Path(path).expanduser()
    audio, sample_rate = sf.read(str(target), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    colored = apply_device_mic_coloration(
        audio,
        int(sample_rate),
        snr_db=snr_db,
        snr_db_range=snr_db_range,
        seed=seed,
    )
    sf.write(str(target), colored, SAMPLE_RATE, subtype="PCM_16")
    return target


def _bandlimit(wave: np.ndarray, sample_rate: int, *, low_hz: float, high_hz: float) -> np.ndarray:
    from scipy.signal import butter, filtfilt

    nyquist = sample_rate / 2.0
    low = max(low_hz / nyquist, 1e-4)
    high = min(high_hz / nyquist, 0.999)
    if low >= high:
        return wave
    b, a = butter(4, [low, high], btype="band")
    return filtfilt(b, a, wave).astype(np.float32, copy=False)


def _soft_compress(wave: np.ndarray, *, mix: float) -> np.ndarray:
    clipped = np.tanh(wave * 2.2)
    return (mix * clipped + (1.0 - mix) * wave).astype(np.float32, copy=False)


def _mix_pink_noise(wave: np.ndarray, *, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    noise = _pink_noise(wave.size, rng)
    signal_power = float(np.mean(wave**2)) or 1e-8
    noise_power = float(np.mean(noise**2)) or 1e-8
    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power / noise_power)
    return (wave + noise * scale).astype(np.float32, copy=False)


def _pink_noise(length: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(length).astype(np.float32)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(length, d=1.0)
    weights = np.ones_like(freqs, dtype=np.float32)
    weights[1:] = 1.0 / np.sqrt(freqs[1:]).astype(np.float32)
    colored = np.fft.irfft(spectrum * weights, n=length).astype(np.float32)
    peak = float(np.max(np.abs(colored))) or 1.0
    return colored / peak


def _apply_short_room(wave: np.ndarray, sample_rate: int, *, rng: np.random.Generator) -> np.ndarray:
    """Very short diffuse tail — cheap stand-in for room IR."""
    tail_ms = float(rng.uniform(25.0, 80.0))
    tail_samples = max(1, int(sample_rate * tail_ms / 1000.0))
    decay = np.exp(-np.linspace(0.0, 4.0, tail_samples, dtype=np.float32))
    impulse = decay * float(rng.uniform(0.08, 0.18))
    impulse[0] = 1.0
    from scipy.signal import fftconvolve

    return fftconvolve(wave, impulse, mode="same").astype(np.float32, copy=False)
