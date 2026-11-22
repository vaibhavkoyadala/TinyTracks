"""Turn flat tensor names into a list of blocks, stages and a loss.

Rules, in order:
1. Names are split on "." and put in a tree. Single wrapper levels such as "model." or
   "transformer." are skipped, so the blocks show the network rather than its wrapper.
2. A node whose children are numbered (layers.0 ... layers.23) is one stack block with a count.
3. Anything else becomes one block per child. A child with its own children becomes a stage,
   and each stage is drawn on its own row.
4. The loss comes from the output head's width. A vocabulary-sized output means cross-entropy
   over next tokens. Anything else means MSE. Pass `loss=` to override.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from math import log10

from tensorfile import TensorInfo

WRAPPERS = {"model", "transformer", "module", "net"}
HEAD_WORDS = ("lm_head", "output", "proj_out", "head", "classifier", "logits", "predictions")
INPUT_WORDS = ("embed", "wte", "wpe", "patch", "conv", "stem", "pos", "tok")
ATTN_WORDS = ("attn", "attention", "qkv")
MLP_WORDS = ("mlp", "fc1", "fc2", "gate_proj", "up_proj", "down_proj", "ffn")


@dataclass
class Node:
    name: str
    children: dict[str, "Node"] = field(default_factory=dict)
    tensors: list[TensorInfo] = field(default_factory=list)  # tensors that end at this node

    def params(self) -> int:
        return sum(t.numel for t in self.tensors) + sum(c.params() for c in self.children.values())

    def all_tensors(self) -> list[TensorInfo]:
        out = list(self.tensors)
        for c in self.children.values():
            out.extend(c.all_tensors())
        return out

    def names(self) -> list[str]:
        out = [self.name]
        for c in self.children.values():
            out.extend(c.names())
        return out


@dataclass
class Block:
    id: str
    label: str            # the name on the block
    path: str             # full dotted path
    kind: str             # input | transformer | attention | mlp | norm | head | other | loss
    params: int
    count: int = 1        # >1 for numbered stacks
    dtype: str = ""
    shape_hint: str = ""  # the largest matrix, e.g. "896 x 4864"
    stage: str = ""       # the stage this block belongs to ("" for the main row)
    detail: str = ""      # second line on the block

    @property
    def is_stack(self) -> bool:
        return self.count > 1


@dataclass
class Architecture:
    source: str
    stages: list[tuple[str, list[Block]]]   # (stage name, blocks)
    loss_name: str
    loss_detail: str
    total_params: int
    num_tensors: int
    dtypes: list[str]

    @property
    def blocks(self) -> list[Block]:
        return [b for _, bs in self.stages for b in bs]


def build_tree(infos: list[TensorInfo]) -> Node:
    root = Node("")
    for info in infos:
        node = root
        parts = info.name.split(".")
        for part in parts[:-1]:
            node = node.children.setdefault(part, Node(part))
        leaf = node.children.setdefault(parts[-1], Node(parts[-1]))
        leaf.tensors.append(info)
    return root


def _unwrap(node: Node) -> Node:
    """Skip wrapper levels like 'model' when they hold the whole network."""
    while len(node.children) == 1 and not node.tensors:
        (name, child), = node.children.items()
        if name in WRAPPERS or (len(child.children) > 1 and not child.tensors):
            node = child
        else:
            break
    return node


def _is_stack(node: Node) -> bool:
    kids = list(node.children)
    return len(kids) >= 2 and all(k.isdigit() for k in kids)


def _is_leaf_group(node: Node) -> bool:
    """A node whose children are only tensors, such as embed_tokens.weight: one block, not several."""
    return bool(node.children) and all(not c.children for c in node.children.values())


def _is_group(node: Node) -> bool:
    """A node with real sub-structure (layers, encoder...), not a stack and not just weights."""
    return bool(node.children) and not _is_stack(node) and not _is_leaf_group(node)


_KIND_ORDER = {"input": 0, "other": 1, "transformer": 2, "attention": 2, "mlp": 2, "norm": 3, "head": 4}


def _kind(path: str, node: Node) -> str:
    lower = path.lower()
    if any(w in lower for w in HEAD_WORDS):
        return "head"
    if re.search(r"(^|[._])(norm|ln|layer_?norm)([._0-9]|$)", lower):
        return "norm"
    if any(w in lower for w in INPUT_WORDS):
        return "input"
    below = " ".join(node.names()).lower()
    if _is_stack(node):
        if any(w in below for w in ATTN_WORDS) and any(w in below for w in MLP_WORDS):
            return "transformer"
        if any(w in below for w in ATTN_WORDS):
            return "attention"
        if any(w in below for w in MLP_WORDS):
            return "mlp"
        return "other"
    if any(w in lower for w in ATTN_WORDS):
        return "attention"
    if any(w in lower for w in MLP_WORDS):
        return "mlp"
    return "other"


def _largest_matrix(tensors: list[TensorInfo]) -> str:
    mats = [t for t in tensors if len(t.shape) >= 2]
    if not mats:
        return ""
    best = max(mats, key=lambda t: t.numel)
    return " × ".join(str(d) for d in best.shape[:3])


def _dtype(tensors: list[TensorInfo]) -> str:
    counts = Counter(t.dtype for t in tensors)
    return counts.most_common(1)[0][0] if counts else ""


def _make_block(path: str, node: Node, stage: str) -> Block:
    tensors = node.all_tensors()
    count = 1
    label = node.name
    detail_parts = []
    if _is_stack(node):
        count = len(node.children)
        label = f"{node.name} ×{count}"
        first = node.children[min(node.children, key=int)]
        per_layer = sum(t.numel for t in first.all_tensors())
        detail_parts.append(f"{_fmt(per_layer)} per layer")
    kind = _kind(path, node)
    shape = _largest_matrix(tensors if not count > 1 else node.children[min(node.children, key=int)].all_tensors())
    return Block(
        id=re.sub(r"[^a-zA-Z0-9]+", "-", path).strip("-") or "root",
        label=label,
        path=path,
        kind=kind,
        params=sum(t.numel for t in tensors),
        count=count,
        dtype=_dtype(tensors),
        shape_hint=shape,
        stage=stage,
        detail=" · ".join(detail_parts),
    )


def _blocks_under(node: Node, path: str, stage: str) -> list[Block]:
    """Blocks for the children of `node`. Flat children become blocks; deeper children are split further."""
    out: list[Block] = []
    for name, child in node.children.items():
        child_path = f"{path}.{name}" if path else name
        if not _is_group(child):
            out.append(_make_block(child_path, child, stage))
        else:
            # A named group such as "encoder.layers" with children is expanded into its own blocks.
            sub = _blocks_under(child, child_path, stage)
            if len(sub) <= 1:
                out.append(_make_block(child_path, child, stage))
            else:
                out.extend(sub)
    return _logical_order(out)


def _logical_order(blocks: list[Block]) -> list[Block]:
    """Inputs first, then layers, then norms and heads. Ties keep the file's order."""
    return sorted(blocks, key=lambda b: _KIND_ORDER.get(b.kind, 1))


def _fmt(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f} B"
    if n >= 1e6:
        return f"{n / 1e6:.1f} M"
    if n >= 1e3:
        return f"{n / 1e3:.1f} K"
    return str(n)


def _order_key(name: str) -> tuple[int, str]:
    lower = name.lower()
    if any(w in lower for w in INPUT_WORDS):
        return (0, lower)
    if "encoder" in lower:
        return (1, lower)
    if "decoder" in lower:
        return (2, lower)
    if any(w in lower for w in HEAD_WORDS):
        return (4, lower)
    return (3, lower)


def _loss(blocks: list[Block], loss: str | None) -> tuple[str, str]:
    heads = [b for b in blocks if b.kind == "head"]
    widths = [int(w) for b in blocks for w in re.findall(r"\d+", b.shape_hint)]
    vocab = max(widths, default=0) if heads or any(b.kind == "input" for b in blocks) else 0
    if loss and loss != "auto":
        return loss, "custom"
    if vocab >= 1000:
        return "cross-entropy", f"next-token over {vocab:,} classes"
    return "MSE", "regression against targets"


def build(infos: list[TensorInfo], source: str, loss: str | None = None) -> Architecture:
    root = _unwrap(build_tree(infos))
    stages: list[tuple[str, list[Block]]] = []

    # Children that are themselves multi-level groups become stages (encoder, decoder, ...).
    groups = [(n, c) for n, c in root.children.items() if _is_group(c)]
    flats = [(n, c) for n, c in root.children.items() if not _is_group(c)]
    if len(groups) >= 2:
        for name, child in sorted(groups, key=lambda item: _order_key(item[0])):
            stages.append((name, _blocks_under(child, name, name)))  # already in logical order
        loose = [(n, _make_block(n, c, "")) for n, c in sorted(flats, key=lambda item: _order_key(item[0]))]
        before = [b for n, b in loose if _order_key(n)[0] <= 1]   # inputs and encoder-side leftovers
        after = [b for n, b in loose if _order_key(n)[0] > 1]     # decoder-side, norms, heads
        if before:
            stages.insert(0, ("", before))
        if after:
            stages.append(("", after))
    else:
        blocks = []
        for name, child in sorted(root.children.items(), key=lambda item: _order_key(item[0])):
            if _is_group(child):
                blocks.extend(_blocks_under(child, name, ""))
            else:
                blocks.append(_make_block(name, child, ""))
        stages.append(("", _logical_order(blocks)))

    all_blocks = [b for _, bs in stages for b in bs]
    if not any(b.kind == "head" for b in all_blocks):
        # Tied output projection (no lm_head tensor): show it as a virtual head on the last stage.
        # The tied weight is the input block with the largest matrix (the vocabulary embedding).
        inputs = [b for b in all_blocks if b.kind == "input" and b.shape_hint]
        embed = max(inputs, key=lambda b: _matrix_size(b.shape_hint), default=None)
        if embed is not None:
            stages[-1][1].append(Block(
                id="lm-head-tied", label="lm head", path="lm_head (tied)", kind="head", params=0,
                shape_hint=embed.shape_hint, stage=stages[-1][0], detail="tied to " + embed.label,
            ))
            all_blocks = [b for _, bs in stages for b in bs]
    loss_name, loss_detail = _loss(all_blocks, loss)
    dtypes = sorted({t.dtype for t in infos})
    return Architecture(
        source=source,
        stages=stages,
        loss_name=loss_name,
        loss_detail=loss_detail,
        total_params=sum(t.numel for t in infos),
        num_tensors=len(infos),
        dtypes=dtypes,
    )


def _matrix_size(shape_hint: str) -> int:
    size = 1
    for n in re.findall(r"\d+", shape_hint):
        size *= int(n)
    return size


def block_scale(block: Block) -> float:
    """0..1 height scale: bigger blocks are taller, on a log scale."""
    return min(1.0, max(0.0, (log10(max(block.params, 1)) - 3.0) / 6.0))


def format_params(n: int) -> str:
    return _fmt(n)
