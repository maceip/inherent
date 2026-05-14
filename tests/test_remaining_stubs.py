from pathlib import Path

import pytest

from inherent.data import directedness, intents


@pytest.mark.xfail(raises=NotImplementedError, strict=True)
def test_openwakeword_negative_loader_is_still_stubbed():
    directedness.load_openwakeword_negatives(Path("data/openwakeword_features"))


@pytest.mark.xfail(raises=NotImplementedError, strict=True)
def test_falai_loader_is_still_stubbed():
    intents.load_falai(Path("data/falai"))


@pytest.mark.xfail(raises=NotImplementedError, strict=True)
def test_minds14_loader_is_still_stubbed():
    intents.load_minds14(Path("data/minds14"))


@pytest.mark.xfail(raises=NotImplementedError, strict=True)
def test_slue_hvb_loader_is_still_stubbed():
    intents.load_slue_hvb(Path("data/slue_hvb"))


@pytest.mark.xfail(raises=NotImplementedError, strict=True)
def test_call_center_loader_is_still_stubbed():
    intents.load_axondata_call_center(Path("data/axondata"))
