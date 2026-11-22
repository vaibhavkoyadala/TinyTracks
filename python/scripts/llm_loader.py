"""Load Hugging Face Llama / Qwen2 checkpoints into tinygrad's transformer.

tinygrad's LLM code is written for GGUF tensor names. Hugging Face safetensors use
different names, but Llama and Qwen2 share one layout: separate q/k/v/o projections,
a gated SwiGLU MLP, RMSNorm, and "half-split" RoPE. So the names map one-to-one, and
no RoPE weight permutation is needed (GGUF's llama converter permutes them, HF does not).
"""
from __future__ import annotations

import json
from pathlib import Path

from tinygrad import Device, Tensor, dtypes, nn
from llm_model import DenseConfig, Transformer

SUPPORTED_ARCHITECTURES = {"LlamaForCausalLM", "Qwen2ForCausalLM"}

# HF tensor name -> tinygrad (GGUF-style) name, relative to a layer's "model.layers.N." prefix
_LAYER_MAP = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.q_proj.bias": "attn_q.bias",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.k_proj.bias": "attn_k.bias",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.v_proj.bias": "attn_v.bias",
    "self_attn.o_proj.weight": "attn_output.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}


class UnsupportedArchitecture(ValueError):
    pass


def read_config(model_dir: Path) -> dict:
    return json.loads((model_dir / "config.json").read_text())


def is_supported(model_dir: Path) -> bool:
    try:
        return read_config(model_dir).get("architectures", [""])[0] in SUPPORTED_ARCHITECTURES
    except (OSError, ValueError, IndexError):
        return False


def _safetensor_files(model_dir: Path) -> list[Path]:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        return [model_dir / n for n in names]
    return [model_dir / "model.safetensors"]


def _load_raw_weights(model_dir: Path) -> dict[str, Tensor]:
    weights: dict[str, Tensor] = {}
    for path in _safetensor_files(model_dir):
        weights.update(nn.state.safe_load(str(path)))
    return weights


def build_config(cfg: dict, max_context: int, has_qkv_bias: bool) -> DenseConfig:
    n_heads = cfg["num_attention_heads"]
    n_kv_heads = cfg.get("num_key_value_heads", n_heads)
    dim = cfg["hidden_size"]
    head_dim = cfg.get("head_dim") or dim // n_heads
    return DenseConfig(
        num_blocks=cfg["num_hidden_layers"],
        dim=dim,
        hidden_dim=cfg["intermediate_size"],
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        norm_eps=cfg.get("rms_norm_eps", 1e-6),
        vocab_size=cfg["vocab_size"],
        head_dim=head_dim,
        rope_theta=float(cfg.get("rope_theta", 10000.0)),
        rope_dim=head_dim,
        max_context=max_context,
        qkv_bias=has_qkv_bias,
    )


def load_transformer(model_dir: Path, max_context: int) -> tuple[Transformer, dict]:
    """Return (model, config dict). Raises UnsupportedArchitecture for anything else."""
    cfg = read_config(model_dir)
    arch = (cfg.get("architectures") or [""])[0]
    if arch not in SUPPORTED_ARCHITECTURES:
        raise UnsupportedArchitecture(f"architecture {arch or 'unknown'} is not supported for live serving yet")

    # Copy every tensor onto the compute device and realize it now. Lazy DISK-backed tensors
    # make the JIT produce kernels with the wrong shapes, and the weights must sit in memory anyway.
    raw = {k: v.to(Device.DEFAULT).realize() for k, v in _load_raw_weights(model_dir).items()}
    state: dict[str, Tensor] = {
        "token_embd.weight": raw["model.embed_tokens.weight"],
        "output_norm.weight": raw["model.norm.weight"],
    }
    if "lm_head.weight" in raw:
        state["output.weight"] = raw["lm_head.weight"]
    else:  # tied embeddings (SmolLM2-135M, Qwen2.5-0.5B)
        state["output.weight"] = raw["model.embed_tokens.weight"]

    n_layers = cfg["num_hidden_layers"]
    for i in range(n_layers):
        prefix = f"model.layers.{i}."
        for hf_suffix, gguf_suffix in _LAYER_MAP.items():
            key = prefix + hf_suffix
            if key in raw:
                state[f"blk.{i}.{gguf_suffix}"] = raw[key]

    has_bias = "model.layers.0.self_attn.q_proj.bias" in raw
    config = build_config(cfg, max_context, has_qkv_bias=has_bias)
    model = Transformer(config)
    # Same half-precision policy as tinygrad's GGUF loader.
    state = {k: v.cast(dtypes.float16) for k, v in state.items()}
    nn.state.load_state_dict(model, state, verbose=False, consume=True, realize=False)
    return model, cfg
