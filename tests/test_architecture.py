import pytest
import torch

from inherent import NUM_HEADS, NUM_INTENT_HEADS
from inherent.config import ModelConfig
from inherent.models import JointAudioIntentInferenceModel, JointAudioIntentModel


def tiny_config(**overrides):
    values = {
        "hidden_size": 32,
        "num_layers": 2,
        "num_attention_heads": 4,
        "conv_kernel_size": 7,
        "mel_bins": 128,
        "max_frames": 3000,
    }
    values.update(overrides)
    return ModelConfig(**values)


def test_conformer_forward_returns_one_pass_contract_logits():
    model = JointAudioIntentModel(tiny_config())
    mel = torch.randn(2, 80, 128)
    lengths = torch.tensor([80, 47])

    logits = model(mel, lengths=lengths)

    assert logits.shape == (2, NUM_HEADS)
    assert torch.isfinite(logits).all()
    interesting, intents = model.split_heads(logits)
    assert interesting.shape == (2, 1)
    assert intents.shape == (2, NUM_INTENT_HEADS)


def test_inference_wrapper_returns_sigmoid_scores():
    model = JointAudioIntentInferenceModel(JointAudioIntentModel(tiny_config()))
    scores = model(torch.randn(2, 80, 128))

    assert scores.shape == (2, NUM_HEADS)
    assert torch.all(scores >= 0)
    assert torch.all(scores <= 1)


def test_padding_batchmate_does_not_change_logits_in_train_mode():
    torch.manual_seed(0)
    model = JointAudioIntentModel(tiny_config())
    model.train()
    short = torch.randn(1, 80, 128)
    long = torch.randn(1, 160, 128)
    padded_short = torch.nn.functional.pad(short, (0, 0, 0, 80))

    single_logits = model(short, lengths=torch.tensor([80]))
    batched_logits = model(
        torch.cat([padded_short, long], dim=0),
        lengths=torch.tensor([80, 160]),
    )[:1]

    assert torch.allclose(single_logits, batched_logits, atol=1e-5, rtol=1e-5)


def test_mel_bin_mismatch_is_rejected():
    model = JointAudioIntentModel(tiny_config())

    with pytest.raises(ValueError, match="last dimension"):
        model(torch.randn(1, 80, 64))


def test_head_count_mismatch_is_rejected():
    with pytest.raises(ValueError, match="num_heads"):
        JointAudioIntentModel(tiny_config(num_heads=12))


def test_unsupported_backbone_is_rejected_by_config():
    with pytest.raises(ValueError, match="unsupported backbone"):
        tiny_config(backbone="wav2vec2_base")


def test_time_dimension_over_contract_limit_is_rejected():
    model = JointAudioIntentModel(tiny_config())

    with pytest.raises(ValueError, match="time dimension"):
        model(torch.randn(1, 3001, 128))
