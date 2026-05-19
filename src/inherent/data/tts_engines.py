"""Pluggable TTS backends for synthetic intent data generation."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..features.frontend import SAMPLE_RATE

OPENF5_TTS_ENGINE = "openf5-tts"
SUPERTONIC_TTS_ENGINE = "supertonic-3"
SUPPORTED_TTS_ENGINES = (OPENF5_TTS_ENGINE, SUPERTONIC_TTS_ENGINE)

# Map inherent voice ids (used in configs/manifests) to Supertonic preset names.
SUPERTONIC_VOICE_BY_ID = {
    "openvoice": "M1",
    "cosyvoice2": "F1",
}


class TtsRuntime(Protocol):
    engine: str

    def synthesize_to_wav(self, *, prompt: str, voice_id: str, output_path: Path) -> None: ...


def resolve_tts_engine(intents_cfg: dict) -> str:
    engine = str(intents_cfg.get("synthetic_tts_engine", OPENF5_TTS_ENGINE)).strip()
    if engine not in SUPPORTED_TTS_ENGINES:
        raise ValueError(
            f"unsupported synthetic_tts_engine {engine!r}; "
            f"expected one of {SUPPORTED_TTS_ENGINES}"
        )
    return engine


def create_tts_runtime(engine: str):
    if engine == OPENF5_TTS_ENGINE:
        from . import synthesis

        return synthesis._OpenF5Runtime(synthesis._openf5_model_files())
    if engine == SUPERTONIC_TTS_ENGINE:
        return _SupertonicRuntime()
    raise ValueError(f"unsupported TTS engine {engine!r}")


class _SupertonicRuntime:
    engine = SUPERTONIC_TTS_ENGINE

    def __init__(self) -> None:
        try:
            from supertonic import TTS
        except ImportError as exc:
            raise RuntimeError(
                "supertonic package is required for synthetic_tts_engine=supertonic-3; "
                "install with: pip install supertonic"
            ) from exc
        self._tts = TTS(auto_download=True)
        self._styles: dict[str, object] = {}

    def _voice_style(self, voice_id: str):
        if voice_id not in self._styles:
            preset = SUPERTONIC_VOICE_BY_ID.get(voice_id, voice_id)
            self._styles[voice_id] = self._tts.get_voice_style(voice_name=preset)
        return self._styles[voice_id]

    def synthesize_to_wav(self, *, prompt: str, voice_id: str, output_path: Path) -> None:
        import numpy as np
        import soundfile as sf

        wav, _duration = self._tts.synthesize(
            prompt,
            voice_style=self._voice_style(voice_id),
            lang="en",
        )
        audio = np.asarray(wav, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            raise RuntimeError(f"supertonic returned empty audio for prompt: {prompt!r}")
        sample_rate = SAMPLE_RATE
        temp_path = output_path.with_suffix(".supertonic-tmp.wav")
        self._tts.save_audio(wav, str(temp_path))
        if temp_path.is_file():
            audio, sample_rate = sf.read(str(temp_path), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            temp_path.unlink(missing_ok=True)
        if int(sample_rate) != SAMPLE_RATE:
            from scipy.signal import resample_poly

            audio = resample_poly(audio, SAMPLE_RATE, int(sample_rate)).astype(np.float32)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, SAMPLE_RATE, subtype="PCM_16")
