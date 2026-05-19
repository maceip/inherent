import pytest

from inherent.data.tts_engines import resolve_tts_engine


def test_resolve_tts_engine_accepts_known_backends():
    assert resolve_tts_engine({"synthetic_tts_engine": "openf5-tts"}) == "openf5-tts"
    assert resolve_tts_engine({"synthetic_tts_engine": "supertonic-3"}) == "supertonic-3"


def test_resolve_tts_engine_rejects_unknown():
    with pytest.raises(ValueError, match="unsupported"):
        resolve_tts_engine({"synthetic_tts_engine": "dramabox"})
