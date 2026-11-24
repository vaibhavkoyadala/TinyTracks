"""Qwen3-VL for tinygrad: vision tower (DeepStack merges) and language model (QK-norm, interleaved MRoPE).

Written against Hugging Face's modeling_qwen3_vl.py. The weights keep Hugging Face's names (split
into a visual and a text dict by `split_weights`), and the activations are fp32 while the weights
are fp16, as in qwen25vl_model.py.

Differences from Qwen2.5-VL that this module handles:
  - The vision tower uses learned position embeddings (bilinearly interpolated to the image grid),
    LayerNorm, GELU MLPs with biases, and no windowed attention.
  - Three intermediate vision layers also emit features ("DeepStack") that are added to the first
    three language-model layers at the image token positions.
  - The text model normalises q and k per head (QK-norm) and assigns the rotary frequencies to the
    temporal, height and width positions in an interleaved pattern.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tinygrad import Tensor, dtypes

from qwen25vl_model import linear, load_weights, rms_norm, rotate_half, rope_tables
from vl_image import QWEN3_SPEC, ImageInput, vision_positions

ARCHITECTURES = {"Qwen3VLForConditionalGeneration"}
IMAGE_SPEC = QWEN3_SPEC
FP16 = dtypes.float16
FP32 = dtypes.float32
EPS_VISION = 1e-6


@dataclass(frozen=True)
class VisionConfig:
    depth: int
    hidden: int
    heads: int
    intermediate: int
    out_hidden: int
    patch: int
    temporal_patch: int
    merge: int
    deepstack: tuple[int, ...]
    num_grid: int                 # side of the learned position-embedding grid (sqrt of its row count)


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
    mrope_section: tuple[int, int, int]


def load_configs(model_dir: Path) -> tuple[VisionConfig, TextConfig, dict]:
    cfg = json.loads((model_dir / "config.json").read_text())
    v = cfg["vision_config"]
    vision = VisionConfig(
        depth=v["depth"], hidden=v["hidden_size"], heads=v["num_heads"], intermediate=v["intermediate_size"],
        out_hidden=v["out_hidden_size"], patch=v["patch_size"], temporal_patch=v["temporal_patch_size"],
        merge=v["spatial_merge_size"], deepstack=tuple(v["deepstack_visual_indexes"]),
        num_grid=math.isqrt(v["num_position_embeddings"]),
    )
    t = cfg["text_config"]
    text = TextConfig(
        layers=t["num_hidden_layers"], hidden=t["hidden_size"], heads=t["num_attention_heads"],
        kv_heads=t["num_key_value_heads"], head_dim=t["head_dim"], intermediate=t["intermediate_size"],
        eps=t["rms_norm_eps"], rope_theta=float(t["rope_theta"]), vocab=t["vocab_size"],
        mrope_section=tuple(t["rope_scaling"]["mrope_section"]),
    )
    return vision, text, cfg


def split_weights(weights: dict) -> tuple[dict, dict]:
    """Visual weights lose the 'model.visual.' prefix; text weights lose 'model.language_model.'."""
    visual = {k[len("model.visual."):]: v for k, v in weights.items() if k.startswith("model.visual.")}
    text = {k[len("model.language_model."):]: v for k, v in weights.items() if k.startswith("model.language_model.")}
    if "lm_head.weight" in weights:
        text["lm_head.weight"] = weights["lm_head.weight"]
    return visual, text


def layer_norm(x: Tensor, weight: Tensor, bias: Tensor, eps: float) -> Tensor:
    xf = x.cast(FP32)
    mean = xf.mean(axis=-1, keepdim=True)
    var = ((xf - mean) * (xf - mean)).mean(axis=-1, keepdim=True)
    return (xf - mean) * (var + eps).rsqrt() * weight.cast(FP32) + bias.cast(FP32)


def mrope_cos_sin(positions: np.ndarray, inv_freq: np.ndarray, section: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """positions (3, N) -> cos/sin (N, head_dim) with interleaved multimodal RoPE.

    Frequency slot j takes its position from the height component when j % 3 == 1 and j < 3*section[1],
    from the width component when j % 3 == 2 and j < 3*section[2], and from the temporal component otherwise.
    """
    n_freq = inv_freq.shape[0]
    comp = np.zeros(n_freq, dtype=np.int64)
    for dim, offset in ((1, 1), (2, 2)):
        comp[offset:3 * section[dim]:3] = dim
    freqs = positions.T[:, :, None].astype(np.float32) * inv_freq.astype(np.float32)   # (N, 3, F)
    angle = freqs[:, comp, np.arange(n_freq)]                                           # (N, F)
    angle = np.concatenate([angle, angle], axis=-1)                                     # half-split layout
    return np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)


class VisionTower:
    """Returns [merged features, deepstack level 0, 1, 2], each (llm_tokens, out_hidden)."""

    def __init__(self, cfg: VisionConfig, w: dict[str, Tensor]):
        self.cfg = cfg
        self.w = w
        half = (cfg.hidden // cfg.heads) // 2
        self.inv = 1.0 / (10000.0 ** (np.arange(0, half, 2, dtype=np.float32) / half))

    def _pos_embed(self, grid: tuple[int, int, int]) -> Tensor:
        """Learned position embeddings, bilinearly interpolated to (h, w), in merge-block order."""
        c = self.cfg
        _, h, w = grid
        g = c.num_grid
        hi = np.linspace(0, g - 1, h, dtype=np.float32)
        wi = np.linspace(0, g - 1, w, dtype=np.float32)
        hf, wf = hi.astype(np.int64), wi.astype(np.int64)
        hc, wc = np.minimum(hf + 1, g - 1), np.minimum(wf + 1, g - 1)
        dh, dw = hi - hf, wi - wf
        bh, bhc = hf * g, hc * g
        idx = [
            (bh[:, None] + wf[None]).ravel(), (bh[:, None] + wc[None]).ravel(),
            (bhc[:, None] + wf[None]).ravel(), (bhc[:, None] + wc[None]).ravel(),
        ]
        wts = [
            ((1 - dh)[:, None] * (1 - dw)[None]).ravel(), ((1 - dh)[:, None] * dw[None]).ravel(),
            (dh[:, None] * (1 - dw)[None]).ravel(), (dh[:, None] * dw[None]).ravel(),
        ]
        table = self.w["pos_embed.weight"].cast(FP32)
        pos = None
        for i, wt in zip(idx, wts):
            term = table[Tensor(i.astype(np.int32))] * Tensor(wt.astype(np.float32)).reshape(-1, 1)
            pos = term if pos is None else pos + term
        m = c.merge
        pos = pos.reshape(h // m, m, w // m, m, c.hidden).permute(0, 2, 1, 3, 4)
        return pos.reshape(h * w, c.hidden)

    def _attn(self, x: Tensor, i: int, cos: Tensor, sin: Tensor) -> Tensor:
        c = self.cfg
        n = x.shape[0]
        hd = c.hidden // c.heads
        p = f"blocks.{i}.attn."
        qkv = linear(x, self.w[p + "qkv.weight"], self.w[p + "qkv.bias"]).reshape(n, 3, c.heads, hd)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        cos3, sin3 = cos.reshape(n, 1, hd), sin.reshape(n, 1, hd)
        q = q * cos3 + rotate_half(q) * sin3
        k = k * cos3 + rotate_half(k) * sin3
        q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))   # (1, heads, N, hd)
        o = q.scaled_dot_product_attention(k, v)                         # full attention: one image, no windows
        o = o.squeeze(0).transpose(0, 1).reshape(n, c.hidden)
        return linear(o, self.w[p + "proj.weight"], self.w[p + "proj.bias"])

    def _mlp(self, x: Tensor, prefix: str) -> Tensor:
        h = linear(x, self.w[prefix + "linear_fc1.weight"], self.w[prefix + "linear_fc1.bias"]).gelu()   # tanh GELU
        return linear(h, self.w[prefix + "linear_fc2.weight"], self.w[prefix + "linear_fc2.bias"])

    def _merge(self, x: Tensor, prefix: str, postshuffle_norm: bool) -> Tensor:
        """Patch merger: LayerNorm, group each 2x2 block into one token, then a two-layer MLP."""
        c = self.cfg
        unit = c.merge * c.merge
        if postshuffle_norm:
            y = layer_norm(x.reshape(-1, c.hidden * unit), self.w[prefix + "norm.weight"], self.w[prefix + "norm.bias"], EPS_VISION)
        else:
            y = layer_norm(x, self.w[prefix + "norm.weight"], self.w[prefix + "norm.bias"], EPS_VISION).reshape(-1, c.hidden * unit)
        y = linear(y, self.w[prefix + "linear_fc1.weight"], self.w[prefix + "linear_fc1.bias"]).gelu(approximate="none")
        return linear(y, self.w[prefix + "linear_fc2.weight"], self.w[prefix + "linear_fc2.bias"])

    def __call__(self, image: ImageInput) -> list[Tensor]:
        c = self.cfg
        pos = vision_positions(image.grid, c.merge)                 # (N, 2) in merge-block order
        cos_np, sin_np = rope_tables(pos, self.inv)
        cos, sin = Tensor(cos_np).realize(), Tensor(sin_np).realize()

        patch_w = self.w["patch_embed.proj.weight"].reshape(c.hidden, -1)
        x = linear(Tensor(image.pixels), patch_w, self.w["patch_embed.proj.bias"]) + self._pos_embed(image.grid)

        deep_index = {layer: k for k, layer in enumerate(c.deepstack)}
        levels: list[Tensor] = [None] * len(c.deepstack)
        for i in range(c.depth):
            p = f"blocks.{i}."
            x = x + self._attn(layer_norm(x, self.w[p + "norm1.weight"], self.w[p + "norm1.bias"], EPS_VISION), i, cos, sin)
            x = x + self._mlp(layer_norm(x, self.w[p + "norm2.weight"], self.w[p + "norm2.bias"], EPS_VISION), p + "mlp.")
            if i in deep_index:
                levels[deep_index[i]] = self._merge(x, f"deepstack_merger_list.{deep_index[i]}.", True).realize()
        main = self._merge(x, "merger.", False).realize()
        return [main] + levels


class TextModel:
    def __init__(self, cfg: TextConfig, w: dict[str, Tensor]):
        self.cfg = cfg
        self.w = w
        self.inv_freq = 1.0 / (cfg.rope_theta ** (np.arange(0, cfg.head_dim, 2, dtype=np.float32) / cfg.head_dim))

    def embed(self, ids: list[int]) -> Tensor:
        return self.w["embed_tokens.weight"][Tensor(ids)].cast(FP32).reshape(1, len(ids), -1)

    def _project(self, i: int, h: Tensor, t: int) -> tuple[Tensor, Tensor, Tensor]:
        """q, k, v as (1, heads, t, head_dim). q and k are normalised per head before RoPE (QK-norm)."""
        c = self.cfg
        p = f"layers.{i}.self_attn."
        q = linear(h, self.w[p + "q_proj.weight"]).reshape(1, t, c.heads, c.head_dim)
        k = linear(h, self.w[p + "k_proj.weight"]).reshape(1, t, c.kv_heads, c.head_dim)
        v = linear(h, self.w[p + "v_proj.weight"]).reshape(1, t, c.kv_heads, c.head_dim)
        q = rms_norm(q, self.w[p + "q_norm.weight"], c.eps)
        k = rms_norm(k, self.w[p + "k_norm.weight"], c.eps)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def _feed_forward(self, i: int, x: Tensor) -> Tensor:
        p = f"layers.{i}.mlp."
        h2 = rms_norm(x, self.w[f"layers.{i}.post_attention_layernorm.weight"], self.cfg.eps)
        gate = linear(h2, self.w[p + "gate_proj.weight"]).silu()
        up = linear(h2, self.w[p + "up_proj.weight"])
        return x + linear(gate * up, self.w[p + "down_proj.weight"])

    def _layer(self, i: int, x: Tensor, cos: Tensor, sin: Tensor, past, mask) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        c = self.cfg
        t = x.shape[1]
        h = rms_norm(x, self.w[f"layers.{i}.input_layernorm.weight"], c.eps)
        q, k, v = self._project(i, h, t)
        cos4, sin4 = cos.reshape(1, 1, t, c.head_dim), sin.reshape(1, 1, t, c.head_dim)
        q = q * cos4 + rotate_half(q) * sin4
        k = k * cos4 + rotate_half(k) * sin4
        if past is not None:
            k = past[0].cat(k, dim=2)
            v = past[1].cat(v, dim=2)
        o = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)
        o = linear(o.transpose(1, 2).reshape(1, t, c.heads * c.head_dim), self.w[f"layers.{i}.self_attn.o_proj.weight"])
        x = self._feed_forward(i, x + o)
        return x, (k, v)

    def _layer_decode(self, i: int, x: Tensor, cos4: Tensor, sin4: Tensor, kc: Tensor, vc: Tensor,
                      onehot: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """One generated token against a fixed-size cache. Shapes never change, so kernels are reused."""
        c = self.cfg
        h = rms_norm(x, self.w[f"layers.{i}.input_layernorm.weight"], c.eps)
        q, k, v = self._project(i, h, 1)
        q = q * cos4 + rotate_half(q) * sin4
        k = k * cos4 + rotate_half(k) * sin4
        kc = kc * (1.0 - onehot) + k * onehot
        vc = vc * (1.0 - onehot) + v * onehot
        o = q.scaled_dot_product_attention(kc, vc, attn_mask=mask, enable_gqa=True)
        o = linear(o.transpose(1, 2).reshape(1, 1, c.heads * c.head_dim), self.w[f"layers.{i}.self_attn.o_proj.weight"])
        return self._feed_forward(i, x + o), kc, vc

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
        hn = rms_norm(h, self.w["norm.weight"], self.cfg.eps)[:, 0]
        logits = (hn.cast(FP16) @ self.w["lm_head.weight"].T).cast(FP32)
        return (logits.realize(), *new_flat)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, cache: list | None,
                deepstack: list[Tensor] | None = None) -> tuple[Tensor, list]:
        """Run the stack on x (1, T, hidden). deepstack[k] (1, T, hidden) is added after layer k.

        Returns last-position logits and the cache of this call.
        """
        t = x.shape[1]
        mask = None
        if t > 1:  # prefill: causal over the new tokens (no prefix cache is reused)
            mask = Tensor.full((1, 1, t, t), float("-inf")).triu(1).realize()
        new_cache = []
        for i in range(self.cfg.layers):
            x, kv = self._layer(i, x, cos, sin, cache[i] if cache else None, mask)
            if deepstack is not None and i < len(deepstack):
                x = x + deepstack[i]
            new_cache.append(kv)
        h = rms_norm(x[:, -1:], self.w["norm.weight"], self.cfg.eps)[:, 0]
        logits = (h.cast(FP16) @ self.w["lm_head.weight"].T).cast(FP32)
        return logits.realize(), new_cache
