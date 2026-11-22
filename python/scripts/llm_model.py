"""Dense decoder-only transformer (Llama / Qwen2 layout) for tinygrad.

Adapted from tinygrad's `tinygrad/llm/model.py` (MIT License, tinygrad 0.14.0). That file
imports `tinygrad.llm.kernels`, which is missing from the published 0.14.0 wheel, so this
copy keeps only the dense attention/MLP path. MoE, MLA, state-space and AMD-only kernels are
left out because no supported model needs them.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

from tinygrad import Tensor, TinyJit, UOp, dtypes, nn
from tinygrad.uop.ops import resolve


@dataclass(frozen=True)
class DenseConfig:
    num_blocks: int
    dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int
    head_dim: int
    rope_theta: float
    rope_dim: int
    max_context: int
    qkv_bias: bool = False


@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device: str | None = None) -> Tensor:
    freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
    freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
    return freqs.cos().cat(freqs.sin(), dim=-1).clone(device)


def apply_rope(x: Tensor, freqs_cis: Tensor) -> Tensor:
    # "half-split" RoPE: the layout Hugging Face checkpoints already use
    assert x.shape[-1] % 2 == 0
    cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
    x1, x2 = x.chunk(2, dim=-1)
    return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)


class Block:
    def __init__(self, config: DenseConfig):
        self.config = config
        self.attn_norm = nn.RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.dim, config.norm_eps)
        q_out = config.head_dim * config.n_heads
        kv_out = config.head_dim * config.n_kv_heads
        self.attn_q = nn.Linear(config.dim, q_out, bias=config.qkv_bias)
        self.attn_k = nn.Linear(config.dim, kv_out, bias=config.qkv_bias)
        self.attn_v = nn.Linear(config.dim, kv_out, bias=config.qkv_bias)
        self.attn_output = nn.Linear(q_out, config.dim, bias=False)
        self.ffn_gate = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.ffn_up = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.ffn_down = nn.Linear(config.hidden_dim, config.dim, bias=False)

    def _attention(self, x: Tensor, start_pos: int | UOp) -> Tensor:
        c = self.config
        q, k, v = self.attn_q(x), self.attn_k(x), self.attn_v(x)
        B, T, _ = x.shape
        q = q.reshape(B, T, c.n_heads, c.head_dim).transpose(1, 2)
        k = k.reshape(B, T, c.n_kv_heads, c.head_dim).transpose(1, 2)
        v = v.reshape(B, T, c.n_kv_heads, c.head_dim).transpose(1, 2)

        freqs = self.freqs_cis[start_pos:start_pos + T]
        q = apply_rope(q[..., :c.rope_dim], freqs).cat(q[..., c.rope_dim:], dim=-1)
        k = apply_rope(k[..., :c.rope_dim], freqs).cat(k[..., c.rope_dim:], dim=-1)

        store = self.cache_kv[:, :, :, start_pos:start_pos + T, :].uop.store(Tensor.stack(k, v).cast(dtypes.half).uop)
        assigned_kv = Tensor(self.cache_kv.uop.after(store))
        k = assigned_kv[0, :, :, 0:start_pos + T, :]
        v = assigned_kv[1, :, :, 0:start_pos + T, :]

        mask = Tensor.full((1, 1, T, start_pos + T), float("-inf"), dtype=x.dtype, buffer=False).triu(start_pos + 1) \
            if resolve(T != 1) else None
        attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)
        return self.attn_output(attn.transpose(1, 2).reshape(B, T, -1))

    def _feed_forward(self, x: Tensor) -> Tensor:
        return self.ffn_down(self.ffn_gate(x).silu().contiguous() * self.ffn_up(x))

    def _init_state(self, x: Tensor) -> None:
        if not hasattr(self, "cache_kv"):
            c = self.config
            self.cache_kv = Tensor.zeros(2, x.shape[0], c.n_kv_heads, c.max_context, c.head_dim,
                                         dtype=dtypes.half, device=x.device)
            self.freqs_cis = precompute_freqs_cis(c.rope_dim, c.max_context, c.rope_theta, device=x.device)

    def __call__(self, x: Tensor, start_pos: int | UOp) -> Tensor:
        self._init_state(x)
        h = x + self._attention(self.attn_norm(x), start_pos)
        return (h + self._feed_forward(self.ffn_norm(h))).contiguous()


class Transformer:
    def __init__(self, config: DenseConfig):
        self.blk: list[Block] = [Block(config) for _ in range(config.num_blocks)]
        self.token_embd = nn.Embedding(config.vocab_size, config.dim)
        self.output_norm = nn.RMSNorm(config.dim, config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.max_context = config.max_context
        self._cached_tokens: list[int] = []
        self.prefill_jit = TinyJit(self.forward)
        self.rollout_jit = TinyJit(self.forward)

    def forward(self, tokens: Tensor, start_pos: int | UOp, temperature: Tensor) -> Tensor:
        x = self.token_embd(tokens).float()
        for block in self.blk:
            x = block(x, start_pos)
        logits = self.output(self.output_norm(x[:, -1:]))[:, -1, :]
        # Gumbel-max: argmax(logits/T - log(-log(U))) samples softmax(logits/T); T=0 is greedy.
        return (logits / temperature.maximum(1e-12) - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)

    def __call__(self, tokens: Tensor, start_pos: int | UOp, temperature: Tensor) -> Tensor:
        return (self.prefill_jit if resolve(tokens.shape[1] != 1) else self.rollout_jit)(tokens.contiguous(), start_pos, temperature)

    def get_start_pos(self, tokens: list[int]) -> int:
        """How many leading tokens already have valid KV entries from the previous call."""
        prefix = 0
        for a, b in zip(tokens[:-1], self._cached_tokens):
            if a != b:
                break
            prefix += 1
        return prefix

    def generate(self, tokens: list[int], chunk_size: int = 32, temperature: float = 0.0):
        """Yield one token id at a time. Stops when the context window is full."""
        v_start_pos = UOp.variable("start_pos", 0, self.max_context - 1)
        v_toks = UOp.variable("toks", 1, chunk_size)
        temp = Tensor([temperature])
        t = Tensor(tokens + [0] * (self.max_context - len(tokens)), dtype="int32").reshape(1, self.max_context)
        start_pos = self.get_start_pos(tokens)
        out, prompt_len = None, len(tokens)
        while len(tokens) < self.max_context:
            n_toks = min(chunk_size, len(tokens) - start_pos)
            sp, nt = v_start_pos.bind(start_pos), v_toks.bind(n_toks)
            out = self(t[:, sp:sp + nt] if start_pos < prompt_len or out is None else out, sp, temp).realize()
            start_pos += n_toks
            if start_pos < len(tokens):  # still prefilling the prompt
                continue
            tokens.append(int(out.item()))
            self._cached_tokens = tokens[:-1]
            yield tokens[-1]
