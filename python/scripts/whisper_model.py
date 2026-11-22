"""Whisper (encoder-decoder speech recognition) for tinygrad.

The attribute names match Hugging Face's safetensors keys (for example `model.encoder.layers.0.fc1`),
so the weights load with nn.state.load_state_dict and need no renaming.

Decoding is greedy and recomputes the decoder over the whole prefix at each step. That is simple
and correct, and it is fast enough for these model sizes. The encoder output is computed once.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tinygrad import Device, Tensor, nn
from tinygrad.nn.state import safe_load

AUDIO_CTX = 1500  # encoder positions for 30 s of audio


@dataclass(frozen=True)
class WhisperConfig:
    d_model: int
    encoder_layers: int
    decoder_layers: int
    encoder_heads: int
    decoder_heads: int
    vocab_size: int
    num_mel_bins: int
    max_target_positions: int

    @staticmethod
    def from_json(path: Path) -> "WhisperConfig":
        c = json.loads(path.read_text())
        return WhisperConfig(
            d_model=c["d_model"],
            encoder_layers=c["encoder_layers"],
            decoder_layers=c["decoder_layers"],
            encoder_heads=c["encoder_attention_heads"],
            decoder_heads=c["decoder_attention_heads"],
            vocab_size=c["vocab_size"],
            num_mel_bins=c["num_mel_bins"],
            max_target_positions=c.get("max_target_positions", 448),
        )


class Attention:
    def __init__(self, d: int, heads: int):
        self.heads, self.head_dim = heads, d // heads
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d, bias=False)  # Whisper's key projection has no bias
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def __call__(self, x: Tensor, kv: Tensor | None = None, mask: Tensor | None = None) -> Tensor:
        B, T, D = x.shape
        src = x if kv is None else kv
        S = src.shape[1]
        q = self.q_proj(x).reshape(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(src).reshape(B, S, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(src).reshape(B, S, self.heads, self.head_dim).transpose(1, 2)
        o = q.scaled_dot_product_attention(k, v, attn_mask=mask)
        return self.out_proj(o.transpose(1, 2).reshape(B, T, D))


class FeedForward:
    def __init__(self, d: int):
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)

    def __call__(self, x: Tensor) -> Tensor:
        return self.fc2(self.fc1(x).gelu(approximate="none"))


class EncoderLayer:
    def __init__(self, d: int, heads: int):
        self.self_attn = Attention(d, heads)
        self.self_attn_layer_norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)
        self.final_layer_norm = nn.LayerNorm(d)

    def __call__(self, x: Tensor) -> Tensor:
        x = x + self.self_attn(self.self_attn_layer_norm(x))
        return x + self.fc2(self.fc1(self.final_layer_norm(x)).gelu(approximate="none"))


class DecoderLayer:
    def __init__(self, d: int, heads: int):
        self.self_attn = Attention(d, heads)
        self.self_attn_layer_norm = nn.LayerNorm(d)
        self.encoder_attn = Attention(d, heads)
        self.encoder_attn_layer_norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)
        self.final_layer_norm = nn.LayerNorm(d)

    def __call__(self, x: Tensor, enc: Tensor, mask: Tensor) -> Tensor:
        x = x + self.self_attn(self.self_attn_layer_norm(x), mask=mask)
        x = x + self.encoder_attn(self.encoder_attn_layer_norm(x), kv=enc)
        return x + self.fc2(self.fc1(self.final_layer_norm(x)).gelu(approximate="none"))


class Encoder:
    def __init__(self, cfg: WhisperConfig):
        d = cfg.d_model
        self.conv1 = nn.Conv1d(cfg.num_mel_bins, d, 3, padding=1)
        self.conv2 = nn.Conv1d(d, d, 3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(AUDIO_CTX, d)
        self.layers = [EncoderLayer(d, cfg.encoder_heads) for _ in range(cfg.encoder_layers)]
        self.layer_norm = nn.LayerNorm(d)

    def __call__(self, mel: Tensor) -> Tensor:
        x = self.conv1(mel).gelu(approximate="none")
        x = self.conv2(x).gelu(approximate="none")  # (B, d, 1500)
        x = x.transpose(1, 2) + self.embed_positions.weight[: x.shape[2]]
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)


class Decoder:
    def __init__(self, cfg: WhisperConfig):
        d = cfg.d_model
        self.embed_tokens = nn.Embedding(cfg.vocab_size, d)
        self.embed_positions = nn.Embedding(cfg.max_target_positions, d)
        self.layers = [DecoderLayer(d, cfg.decoder_heads) for _ in range(cfg.decoder_layers)]
        self.layer_norm = nn.LayerNorm(d)

    def __call__(self, tokens: Tensor, enc: Tensor) -> Tensor:
        T = tokens.shape[1]
        x = self.embed_tokens(tokens) + self.embed_positions.weight[:T]
        mask = Tensor.full((1, 1, T, T), float("-inf")).triu(1)
        for layer in self.layers:
            x = layer(x, enc, mask)
        return self.layer_norm(x)


class WhisperCore:
    def __init__(self, cfg: WhisperConfig):
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)


class WhisperForConditionalGeneration:
    def __init__(self, cfg: WhisperConfig):
        self.config = cfg
        self.model = WhisperCore(cfg)

    def encode(self, mel: Tensor) -> Tensor:
        return self.model.encoder(mel)

    def logits(self, tokens: Tensor, enc: Tensor) -> Tensor:
        h = self.model.decoder(tokens, enc)
        # Output projection is tied to the token embedding.
        return h @ self.model.decoder.embed_tokens.weight.T


def load_whisper(model_dir: Path) -> tuple[WhisperForConditionalGeneration, WhisperConfig]:
    cfg = WhisperConfig.from_json(model_dir / "config.json")
    weights = {k: v.to(Device.DEFAULT).realize() for k, v in safe_load(str(model_dir / "model.safetensors")).items()}
    model = WhisperForConditionalGeneration(cfg)
    nn.state.load_state_dict(model, weights, verbose=False, consume=True, realize=False, strict=True)
    return model, cfg
