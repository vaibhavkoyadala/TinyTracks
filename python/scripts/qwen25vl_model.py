"""Qwen2.5-VL for tinygrad: vision tower (windowed attention) and language model (multimodal RoPE).

Written against Hugging Face's modeling_qwen2_5_vl.py. The weights keep Hugging Face's names, and
the activations are fp32 while the weights are fp16, so the residual stream does not overflow.

The language model takes rotary cos/sin from the caller, because multimodal RoPE gives every token
three position indices (temporal, height, width). The KV cache is a plain list of tensors per layer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tinygrad import Device, Tensor, dtypes
from tinygrad.nn.state import safe_load

from vl_image import QWEN25_SPEC, ImageInput, window_index, vision_positions

ARCHITECTURES = {"Qwen2_5_VLForConditionalGeneration"}
IMAGE_SPEC = QWEN25_SPEC
FP16 = dtypes.float16
FP32 = dtypes.float32
MROPE_SECTION = (16, 24, 24)


@dataclass(frozen=True)
class VisionConfig:
    depth: int
    hidden: int
    heads: int
    intermediate: int
    out_hidden: int
    fullatt_blocks: tuple[int, ...]
    patch: int
    temporal_patch: int
    merge: int
    deepstack: tuple[int, ...] = ()      # Qwen2.5-VL has no DeepStack levels


@dataclass(frozen=True)
class TextConfig:
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    eps: float
    rope_theta: float
    vocab: int
    mrope_section: tuple[int, int, int] = MROPE_SECTION


def split_weights(weights: dict) -> tuple[dict, dict]:
    """Visual weights lose the 'visual.' prefix. The text weights keep their names."""
    return {k[len("visual."):]: v for k, v in weights.items() if k.startswith("visual.")}, weights


def mrope_cos_sin(positions: np.ndarray, inv_freq: np.ndarray, section: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """positions (3, N) -> cos/sin (N, head_dim) with contiguous sections (temporal, then height, then width)."""
    return rope_tables(positions.T, inv_freq, list(section))


def load_configs(model_dir: Path) -> tuple[VisionConfig, TextConfig, dict]:
    cfg = json.loads((model_dir / "config.json").read_text())
    v = cfg["vision_config"]
    vision = VisionConfig(
        depth=v["depth"], hidden=v["hidden_size"], heads=v["num_heads"], intermediate=v["intermediate_size"],
        out_hidden=v["out_hidden_size"], fullatt_blocks=tuple(v["fullatt_block_indexes"]),
        patch=v["patch_size"], temporal_patch=v["temporal_patch_size"], merge=v["spatial_merge_size"],
    )
    heads = cfg["num_attention_heads"]
    text = TextConfig(
        layers=cfg["num_hidden_layers"], hidden=cfg["hidden_size"], heads=heads,
        kv_heads=cfg["num_key_value_heads"], head_dim=cfg["hidden_size"] // heads,
        intermediate=cfg["intermediate_size"], eps=cfg["rms_norm_eps"],
        rope_theta=float(cfg["rope_theta"]), vocab=cfg["vocab_size"],
    )
    return vision, text, cfg


def load_weights(model_dir: Path) -> dict[str, Tensor]:
    """Read every shard, move each tensor to the compute device, and cast it to fp16."""
    index = model_dir / "model.safetensors.index.json"
    files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    weights: dict[str, Tensor] = {}
    for name in files:
        for key, tensor in safe_load(str(model_dir / name)).items():
            weights[key] = tensor.to(Device.DEFAULT).cast(FP16).realize()
    return weights


def rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    xf = x.cast(FP32)
    var = (xf * xf).mean(axis=-1, keepdim=True)
    return (xf * (var + eps).rsqrt()) * weight.cast(FP32)


def linear(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    y = (x.cast(FP16) @ weight.T).cast(FP32)
    return y + bias.cast(FP32) if bias is not None else y


def rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return (-x[..., half:]).cat(x[..., :half], dim=-1)


def rope_tables(positions: np.ndarray, inv_freq: np.ndarray, sections: list[int] | None = None):
    """Build cos/sin for half-split RoPE.

    positions: (N, k) for k components (2 for vision h/w, 3 for text temporal/height/width).
    With sections, frequency slot j uses component (section index of j), as in multimodal RoPE.
    """
    freqs = positions[..., None].astype(np.float32) * inv_freq.astype(np.float32)  # (N, k, F)
    if sections is None:
        angle = np.concatenate([freqs[:, 0], freqs[:, 1]], axis=-1) if freqs.shape[1] == 2 else freqs[:, 0]
    else:
        pieces = []
        start = 0
        for comp, size in enumerate(sections):
            pieces.append(freqs[:, comp, start:start + size])
            start += size
        angle = np.concatenate(pieces, axis=-1)
    angle = np.concatenate([angle, angle], axis=-1)   # (N, 2F): the half-split layout
    return np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)


class VisionTower:
    def __init__(self, cfg: VisionConfig, w: dict[str, Tensor]):
        self.cfg = cfg
        self.w = w

    def _attn(self, x: Tensor, i: int, cos: Tensor, sin: Tensor, mask: Tensor | None) -> Tensor:
        c = self.cfg
        n = x.shape[0]
        hd = c.hidden // c.heads
        p = f"blocks.{i}.attn."
        qkv = linear(x, self.w[p + "qkv.weight"], self.w[p + "qkv.bias"]).reshape(n, 3, c.heads, hd)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]          # (N, heads, hd)
        cos3, sin3 = cos.reshape(n, 1, hd), sin.reshape(n, 1, hd)
        q = q * cos3 + rotate_half(q) * sin3
        k = k * cos3 + rotate_half(k) * sin3
        q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))  # (1, heads, N, hd)
        o = q.scaled_dot_product_attention(k, v, attn_mask=mask)
        o = o.squeeze(0).transpose(0, 1).reshape(n, c.hidden)
        return linear(o, self.w[p + "proj.weight"], self.w[p + "proj.bias"])

    def _mlp(self, x: Tensor, prefix: str) -> Tensor:
        gate = linear(x, self.w[prefix + "gate_proj.weight"], self.w[prefix + "gate_proj.bias"]).silu()
        up = linear(x, self.w[prefix + "up_proj.weight"], self.w[prefix + "up_proj.bias"])
        return linear(gate * up, self.w[prefix + "down_proj.weight"], self.w[prefix + "down_proj.bias"])

    def __call__(self, image: ImageInput) -> list[Tensor]:
        """Patches of one image -> [(llm_tokens, out_hidden)] in the order the language model expects.

        Returned as a list so it matches the Qwen3-VL tower, which also returns its DeepStack levels.
        """
        c = self.cfg
        n = image.pixels.shape[0]
        order, cu_window = window_index(image.grid)           # order over merged tokens
        unit = c.merge * c.merge
        pix = image.pixels.reshape(n // unit, unit, -1)[order].reshape(n, -1)
        pos = vision_positions(image.grid).reshape(n // unit, unit, 2)[order].reshape(n, 2)

        inv = 1.0 / (10000.0 ** (np.arange(0, (c.hidden // c.heads) // 2, 2, dtype=np.float32) / ((c.hidden // c.heads) // 2)))
        cos_np, sin_np = rope_tables(pos, inv)
        cos, sin = Tensor(cos_np).realize(), Tensor(sin_np).realize()

        patch_w = self.w["patch_embed.proj.weight"].reshape(c.hidden, -1)
        x = linear(Tensor(pix), patch_w)

        # Block-diagonal mask for windowed blocks: a token attends only inside its own window.
        win_mask = None
        if len(cu_window) > 2:
            ids = np.zeros(n, dtype=np.int64)
            for j in range(len(cu_window) - 1):
                ids[cu_window[j]:cu_window[j + 1]] = j
            allowed = ids[:, None] == ids[None, :]
            win_mask = Tensor(np.where(allowed, 0.0, -np.inf).astype(np.float32)).reshape(1, 1, n, n).realize()

        for i in range(c.depth):
            mask = None if i in c.fullatt_blocks else win_mask
            x = x + self._attn(rms_norm(x, self.w[f"blocks.{i}.norm1.weight"], 1e-6), i, cos, sin, mask)
            x = x + self._mlp(rms_norm(x, self.w[f"blocks.{i}.norm2.weight"], 1e-6), f"blocks.{i}.mlp.")

        m = rms_norm(x, self.w["merger.ln_q.weight"], 1e-6).reshape(n // unit, c.hidden * unit)
        m = linear(m, self.w["merger.mlp.0.weight"], self.w["merger.mlp.0.bias"]).gelu(approximate="none")
        m = linear(m, self.w["merger.mlp.2.weight"], self.w["merger.mlp.2.bias"])
        return [m[Tensor(np.argsort(order))].realize()]


class TextModel:
    def __init__(self, cfg: TextConfig, w: dict[str, Tensor]):
        self.cfg = cfg
        self.w = w
        self.inv_freq = 1.0 / (cfg.rope_theta ** (np.arange(0, cfg.head_dim, 2, dtype=np.float32) / cfg.head_dim))

    def embed(self, ids: list[int]) -> Tensor:
        return self.w["model.embed_tokens.weight"][Tensor(ids)].cast(FP32).reshape(1, len(ids), -1)

    def _layer(self, i: int, x: Tensor, cos: Tensor, sin: Tensor, past, mask) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        c = self.cfg
        p = f"model.layers.{i}."
        t = x.shape[1]
        h = rms_norm(x, self.w[p + "input_layernorm.weight"], c.eps)
        q = linear(h, self.w[p + "self_attn.q_proj.weight"], self.w[p + "self_attn.q_proj.bias"])
        k = linear(h, self.w[p + "self_attn.k_proj.weight"], self.w[p + "self_attn.k_proj.bias"])
        v = linear(h, self.w[p + "self_attn.v_proj.weight"], self.w[p + "self_attn.v_proj.bias"])
        q = q.reshape(1, t, c.heads, c.head_dim).transpose(1, 2)
        k = k.reshape(1, t, c.kv_heads, c.head_dim).transpose(1, 2)
        v = v.reshape(1, t, c.kv_heads, c.head_dim).transpose(1, 2)
        cos4, sin4 = cos.reshape(1, 1, t, c.head_dim), sin.reshape(1, 1, t, c.head_dim)
        q = q * cos4 + rotate_half(q) * sin4
        k = k * cos4 + rotate_half(k) * sin4
        if past is not None:
            k = past[0].cat(k, dim=2)
            v = past[1].cat(v, dim=2)
        o = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)
        o = linear(o.transpose(1, 2).reshape(1, t, c.hidden), self.w[p + "self_attn.o_proj.weight"])
        x = x + o
        h2 = rms_norm(x, self.w[p + "post_attention_layernorm.weight"], c.eps)
        gate = linear(h2, self.w[p + "mlp.gate_proj.weight"]).silu()
        up = linear(h2, self.w[p + "mlp.up_proj.weight"])
        x = x + linear(gate * up, self.w[p + "mlp.down_proj.weight"])
        return x, (k, v)

    def _layer_decode(self, i: int, x: Tensor, cos4: Tensor, sin4: Tensor, kc: Tensor, vc: Tensor,
                      onehot: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """One generated token against a fixed-size cache. Shapes never change, so kernels are reused."""
        c = self.cfg
        p = f"model.layers.{i}."
        h = rms_norm(x, self.w[p + "input_layernorm.weight"], c.eps)
        q = linear(h, self.w[p + "self_attn.q_proj.weight"], self.w[p + "self_attn.q_proj.bias"])
        k = linear(h, self.w[p + "self_attn.k_proj.weight"], self.w[p + "self_attn.k_proj.bias"])
        v = linear(h, self.w[p + "self_attn.v_proj.weight"], self.w[p + "self_attn.v_proj.bias"])
        q = q.reshape(1, 1, c.heads, c.head_dim).transpose(1, 2)
        k = k.reshape(1, 1, c.kv_heads, c.head_dim).transpose(1, 2)
        v = v.reshape(1, 1, c.kv_heads, c.head_dim).transpose(1, 2)
        q = q * cos4 + rotate_half(q) * sin4
        k = k * cos4 + rotate_half(k) * sin4
        # Write this position into the cache.
        kc = kc * (1.0 - onehot) + k * onehot
        vc = vc * (1.0 - onehot) + v * onehot
        o = q.scaled_dot_product_attention(kc, vc, attn_mask=mask, enable_gqa=True)
        o = linear(o.transpose(1, 2).reshape(1, 1, c.hidden), self.w[p + "self_attn.o_proj.weight"])
        x = x + o
        h2 = rms_norm(x, self.w[p + "post_attention_layernorm.weight"], c.eps)
        gate = linear(h2, self.w[p + "mlp.gate_proj.weight"]).silu()
        up = linear(h2, self.w[p + "mlp.up_proj.weight"])
        x = x + linear(gate * up, self.w[p + "mlp.down_proj.weight"])
        return x, kc, vc

    def decode_body(self, x: Tensor, cos4: Tensor, sin4: Tensor, onehot: Tensor, mask: Tensor, *flat: Tensor):
        """One generated token against fixed-size caches, as a pure function of its tensor inputs.

        flat holds (K, V) per layer, each (1, kv_heads, max_len, head_dim). Returns (logits, *new flat caches).
        The runtime wraps this in TinyJit, so the graph is recorded once and replayed for every token.
        """
        new_flat: list[Tensor] = []
        h = x
        for i in range(self.cfg.layers):
            h, kc, vc = self._layer_decode(i, h, cos4, sin4, flat[2 * i], flat[2 * i + 1], onehot, mask)
            new_flat += [kc, vc]
        hn = rms_norm(h, self.w["model.norm.weight"], self.cfg.eps)[:, 0]
        logits = (hn.cast(FP16) @ self.w["model.embed_tokens.weight"].T).cast(FP32)
        return (logits.realize(), *new_flat)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, cache: list | None,
                deepstack: list[Tensor] | None = None) -> tuple[Tensor, list]:
        """Run the whole stack on x (1, T, hidden). Returns last-position logits and the updated cache.

        Qwen2.5-VL has no DeepStack levels, so `deepstack` must be None.
        """
        assert deepstack is None, "Qwen2.5-VL does not use DeepStack features"
        t = x.shape[1]
        past_len = cache[0][0].shape[2] if cache else 0
        mask = None
        if t > 1:  # prefill: causal over the new tokens (no prefix cache is reused)
            mask = Tensor.full((1, 1, t, t), float("-inf")).triu(1).realize()
        new_cache = []
        for i in range(self.cfg.layers):
            x, kv = self._layer(i, x, cos, sin, cache[i] if cache else None, mask)
            new_cache.append(kv)
        h = rms_norm(x[:, -1:], self.w["model.norm.weight"], self.cfg.eps)[:, 0]
        # The output head is tied to the token embedding.
        logits = (h.cast(FP16) @ self.w["model.embed_tokens.weight"].T).cast(FP32)
        return logits.realize(), new_cache
