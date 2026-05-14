"""Joint audio-to-intent Conformer.

The runtime contract is deliberately simple: feed one mel tensor shaped
``[B, T, 128]`` and get one logits tensor shaped ``[B, 13]``. Index 0 is the
``isInteresting`` directedness gate. Indices 1..12 are the category heads in
``inherent.HEAD_ORDER``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..config import ModelConfig, RUNTIME_MAX_FRAMES, SUPPORTED_BACKBONE
from .. import NUM_HEADS, NUM_INTENT_HEADS


class SmallConformer(nn.Module):
    """CPU-sized Conformer encoder for mel input.

    Input is ``[B, T, mel_bins]``. Output is ``[B, T', hidden]`` with lengths
    reduced by the temporal subsampler. The subsampler keeps long 60-second
    windows tractable while preserving the 1-second low-latency use case.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.mel_bins = cfg.mel_bins
        self.max_frames = cfg.max_frames
        self.subsample = TemporalConvSubsampler(cfg.mel_bins, cfg.hidden_size)
        self.positional_encoding = SinusoidalPositionalEncoding(cfg.hidden_size)
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    cfg.hidden_size,
                    cfg.num_attention_heads,
                    cfg.conv_kernel_size,
                )
                for _ in range(cfg.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(cfg.hidden_size)

    def forward(
        self,
        mel: torch.Tensor,
        lengths: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _validate_mel(mel, self.mel_bins, self.max_frames)
        _validate_lengths(lengths, batch_size=mel.size(0), max_time=mel.size(1))
        _validate_padding_mask(padding_mask, batch_size=mel.size(0), max_time=mel.size(1))
        x, lengths = self.subsample(mel, lengths)
        padding_mask = _downsample_padding_mask(padding_mask, x.size(1), lengths)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)
        return self.norm(x), padding_mask


class TemporalConvSubsampler(nn.Module):
    """Two stride-2 temporal convolutions: ``T`` becomes roughly ``ceil(T / 4)``."""

    def __init__(self, mel_bins: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(mel_bins, hidden, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )

    def forward(
        self,
        mel: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.net(mel.transpose(1, 2)).transpose(1, 2)
        if lengths is not None:
            lengths = _conv1d_out_length(lengths)
            lengths = _conv1d_out_length(lengths)
            lengths = lengths.clamp(min=1, max=x.size(1))
        return x, lengths


class SinusoidalPositionalEncoding(nn.Module):
    """Deterministic positional signal for variable-length mel sequences."""

    def __init__(self, hidden: int, max_len: int = 4096):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden)
        )
        encoding = torch.zeros(max_len, hidden, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.encoding.size(1):
            raise ValueError(
                f"sequence length {x.size(1)} exceeds positional encoding capacity "
                f"{self.encoding.size(1)}"
            )
        return x + self.encoding[:, : x.size(1)].to(dtype=x.dtype)


class ConformerBlock(nn.Module):
    """Macaron-style Conformer block: FFN/2, MHSA, Conv, FFN/2, LayerNorm."""

    def __init__(self, hidden: int, n_heads: int, kernel_size: int):
        super().__init__()
        self.ffn1 = FeedForward(hidden)
        self.attn = MultiHeadSelfAttention(hidden, n_heads)
        self.attn_norm = nn.LayerNorm(hidden)
        self.conv = ConvolutionModule(hidden, kernel_size)
        self.ffn2 = FeedForward(hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        attn_out = self.attn(self.attn_norm(x), padding_mask=padding_mask)
        x = x + attn_out
        x = x + self.conv(x, padding_mask=padding_mask)
        x = x + 0.5 * self.ffn2(x)
        return self.norm(x)


class FeedForward(nn.Module):
    def __init__(self, hidden: int, expansion: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * expansion),
            nn.SiLU(),
            nn.Linear(hidden * expansion, hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden: int, n_heads: int):
        super().__init__()
        if hidden % n_heads != 0:
            raise ValueError(
                f"hidden_size={hidden} must be divisible by num_attention_heads={n_heads}"
            )
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.hidden = hidden
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.out = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, time, hidden = x.shape
        qkv = self.qkv(x)
        q = qkv[..., : self.hidden].view(batch, time, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = qkv[..., self.hidden : 2 * self.hidden].view(batch, time, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = qkv[..., 2 * self.hidden :].view(batch, time, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch, time, hidden)
        return self.out(context)


class ConvolutionModule(nn.Module):
    def __init__(self, hidden: int, kernel_size: int):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("conv_kernel_size must be odd to preserve sequence length")
        self.norm = nn.LayerNorm(hidden)
        self.pointwise1 = nn.Conv1d(hidden, hidden * 2, kernel_size=1)
        self.depthwise = nn.Conv1d(
            hidden,
            hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden,
        )
        self.channel_norm = nn.LayerNorm(hidden)
        self.pointwise2 = nn.Conv1d(hidden, hidden, kernel_size=1)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        residual_mask = padding_mask
        conv_mask = padding_mask.unsqueeze(1) if padding_mask is not None else None
        x = self.norm(x).transpose(1, 2)
        x = self.pointwise1(x)
        x = nn.functional.glu(x, dim=1)
        if conv_mask is not None:
            x = x.masked_fill(conv_mask, 0.0)
        x = self.depthwise(x)
        if conv_mask is not None:
            x = x.masked_fill(conv_mask, 0.0)
        x = self.channel_norm(x.transpose(1, 2)).transpose(1, 2)
        if conv_mask is not None:
            x = x.masked_fill(conv_mask, 0.0)
        x = nn.functional.silu(x)
        x = self.pointwise2(x)
        x = x.transpose(1, 2)
        if residual_mask is not None:
            x = x.masked_fill(residual_mask.unsqueeze(-1), 0.0)
        return x


class JointAudioIntentModel(nn.Module):
    """One-pass directedness gate plus 12 intent category heads.

    ``forward`` returns logits, not probabilities, so training can use
    ``binary_cross_entropy_with_logits``. Use ``predict_proba`` for sigmoid
    scores in the exact order expected by genesis.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.num_heads != NUM_HEADS:
            raise ValueError(
                f"cfg.num_heads={cfg.num_heads} does not match HEAD_ORDER size {NUM_HEADS}"
            )

        if cfg.backbone != SUPPORTED_BACKBONE:
            raise ValueError(f"unsupported backbone {cfg.backbone!r}")
        self.backbone = SmallConformer(cfg)

        self.attn_pool = AttentivePool(cfg.hidden_size)
        self.interesting_head = nn.Linear(cfg.hidden_size, 1)
        self.intent_head = nn.Linear(cfg.hidden_size, NUM_INTENT_HEADS)

    def forward(
        self,
        mel: torch.Tensor,
        lengths: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features, encoded_padding_mask = self.backbone(
            mel,
            lengths=lengths,
            padding_mask=padding_mask,
        )
        pooled = self.attn_pool(features, padding_mask=encoded_padding_mask)
        interesting_logit = self.interesting_head(pooled)
        intent_logits = self.intent_head(pooled)
        return torch.cat([interesting_logit, intent_logits], dim=-1)

    @torch.no_grad()
    def predict_proba(
        self,
        mel: torch.Tensor,
        lengths: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.sigmoid(self.forward(mel, lengths=lengths, padding_mask=padding_mask))

    @staticmethod
    def split_heads(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if logits.size(-1) != NUM_HEADS:
            raise ValueError(f"expected {NUM_HEADS} logits, got {logits.size(-1)}")
        return logits[..., :1], logits[..., 1:]


class JointAudioIntentInferenceModel(nn.Module):
    """Export wrapper that returns contract-ready sigmoid scores."""

    def __init__(self, model: JointAudioIntentModel):
        super().__init__()
        self.model = model
        self.output_head = nn.Linear(model.interesting_head.in_features, NUM_HEADS)
        with torch.no_grad():
            self.output_head.weight.copy_(
                torch.cat([model.interesting_head.weight, model.intent_head.weight], dim=0)
            )
            self.output_head.bias.copy_(
                torch.cat([model.interesting_head.bias, model.intent_head.bias], dim=0)
            )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        features, encoded_padding_mask = self.model.backbone(mel)
        if encoded_padding_mask is not None:
            features = features.masked_fill(encoded_padding_mask.unsqueeze(-1), 0.0)
        features_nct = features.transpose(1, 2)
        attention_weight = self.model.attn_pool.attn.weight.squeeze(0).view(1, -1, 1)
        attention_bias = self.model.attn_pool.attn.bias.view(1, 1, 1)
        attention_logits = (features_nct * attention_weight).sum(dim=1, keepdim=True) + attention_bias
        attention = torch.softmax(attention_logits, dim=-1)
        pooled = (features_nct * attention).sum(dim=-1)
        return torch.sigmoid(self.output_head(pooled))


class AttentivePool(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.attn = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.attn(x)
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask.unsqueeze(-1), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        if padding_mask is not None:
            weights = weights.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
        return (weights * x).sum(dim=1)


def _validate_mel(mel: torch.Tensor, mel_bins: int, max_frames: int = RUNTIME_MAX_FRAMES) -> None:
    if mel.ndim != 3:
        raise ValueError(f"mel_spectrogram must have shape [B, T, mel_bins], got {tuple(mel.shape)}")
    if mel.size(0) < 1:
        raise ValueError("mel_spectrogram batch dimension must be positive")
    if mel.size(1) < 1 or mel.size(1) > max_frames:
        raise ValueError(f"mel_spectrogram time dimension must be in [1, {max_frames}], got {mel.size(1)}")
    if mel.size(-1) != mel_bins:
        raise ValueError(
            f"mel_spectrogram last dimension must be {mel_bins}, got {mel.size(-1)}"
        )
    if not torch.is_floating_point(mel):
        raise TypeError(f"mel_spectrogram must be floating point, got {mel.dtype}")


def _validate_lengths(lengths: torch.Tensor | None, *, batch_size: int, max_time: int) -> None:
    if lengths is None:
        return
    if lengths.ndim != 1 or lengths.numel() != batch_size:
        raise ValueError(f"lengths must have shape [{batch_size}], got {tuple(lengths.shape)}")
    if torch.is_floating_point(lengths):
        raise TypeError(f"lengths must be integer, got {lengths.dtype}")
    if torch.any(lengths < 1) or torch.any(lengths > max_time):
        raise ValueError(f"lengths must be in [1, {max_time}]")


def _validate_padding_mask(
    padding_mask: torch.Tensor | None,
    *,
    batch_size: int,
    max_time: int,
) -> None:
    if padding_mask is None:
        return
    if padding_mask.shape != (batch_size, max_time):
        raise ValueError(
            f"padding_mask must have shape [{batch_size}, {max_time}], got {tuple(padding_mask.shape)}"
        )
    if padding_mask.dtype != torch.bool:
        raise TypeError(f"padding_mask must be bool, got {padding_mask.dtype}")


def _conv1d_out_length(lengths: torch.Tensor) -> torch.Tensor:
    # kernel=3, stride=2, padding=1, dilation=1.
    return torch.div(lengths + 1, 2, rounding_mode="floor")


def _downsample_padding_mask(
    padding_mask: torch.Tensor | None,
    time: int,
    lengths: torch.Tensor | None,
) -> torch.Tensor | None:
    if lengths is not None:
        positions = torch.arange(time, device=lengths.device).unsqueeze(0)
        return positions >= lengths.unsqueeze(1)
    if padding_mask is None:
        return None
    mask = padding_mask[:, None, :].to(dtype=torch.float32)
    mask = nn.functional.max_pool1d(mask, kernel_size=3, stride=2, padding=1)
    mask = nn.functional.max_pool1d(mask, kernel_size=3, stride=2, padding=1)
    return mask.squeeze(1).to(dtype=torch.bool)
