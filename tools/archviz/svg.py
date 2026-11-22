"""Draw an Architecture as a standalone SVG: isometric blocks, connectors, loss, and optional loops.

The SVG has stable ids and classes so it can be controlled after it is written:
  - <g class="block kind-..." id="block-<id>">  one per block; `hl` is added when highlighted
  - <g id="loops">                              gradient loops; hidden with the `l` key in a browser
  - <g id="activations">                        trace overlay; SMIL animates it, or one frame is fixed
Clicking a block in a browser toggles its highlight. Space pauses or resumes the trace animation.
"""
from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path

from arch import Architecture, Block, block_scale, format_params

# Front-face size and the oblique depth vector (up and to the right), matching the reference diagrams.
W = 210
DX, DY = 22, -14
ROW_GAP = 64
COL_GAP = 66
PER_ROW = 4
MARGIN = 28
FONT = "Inter, ui-sans-serif, system-ui, sans-serif"

KIND_COLORS = {
    "input": "#71717a",
    "transformer": "#2563eb",
    "attention": "#2563eb",
    "mlp": "#2563eb",
    "norm": "#64748b",
    "head": "#15803d",
    "other": "#7c3aed",
    "loss": "#c2410c",
}
CROSS_COLOR = "#7c3aed"
LOOP_COLOR = "#c2410c"
HIGHLIGHT = "#facc15"
TRAINABLE = {"input", "transformer", "attention", "mlp", "norm", "head", "other"}


@dataclass
class Trace:
    frames: list[dict]          # each: {"label": str, "active": {block key: 0..1}}
    frame_ms: int = 500

    @staticmethod
    def load(path: Path) -> "Trace":
        data = json.loads(path.read_text())
        frames = data.get("frames") if isinstance(data, dict) else data
        if not isinstance(frames, list) or not frames:
            raise ValueError("a trace needs a non-empty \"frames\" list")
        clean = []
        for i, f in enumerate(frames):
            active = f.get("active", {})
            if not isinstance(active, dict):
                raise ValueError(f"frame {i}: \"active\" must map block names to values")
            clean.append({"label": str(f.get("label", f"frame {i}")),
                          "active": {str(k): max(0.0, min(1.0, float(v))) for k, v in active.items()}})
        frame_ms = int(data.get("frame_ms", 500)) if isinstance(data, dict) else 500
        return Trace(clean, max(50, frame_ms))


@dataclass
class Options:
    loops: bool = False
    highlight: set[str] = field(default_factory=set)
    trace: Trace | None = None
    frame: int | None = None    # render one trace frame as a still image
    title: str | None = None


def matches(block: Block, key: str) -> bool:
    """A trace key can name a block by id, path, label, or the bare name without its ×N count."""
    bare = block.label.split(" ×")[0]
    if key in (block.id, block.path, block.label, bare, block.path.split(".")[-1]):
        return True
    return block.stage != "" and key == block.stage  # naming a stage selects every block in it


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def _block_height(b: Block) -> int:
    return int(78 + 26 * block_scale(b) + (8 if b.count > 1 else 0))


class _Canvas:
    def __init__(self):
        self.parts: list[str] = []

    def add(self, s: str) -> None:
        self.parts.append(s)


def _poly(points: list[tuple[float, float]], fill: str, opacity: float, extra: str = "") -> str:
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polygon points="{pts}" fill="{fill}" fill-opacity="{opacity}" {extra}/>'


def _block_svg(b: Block, x: float, y: float, color: str, loss: bool = False) -> tuple[str, float]:
    H = _block_height(b)
    out = []
    if b.count > 1:  # stacked slabs behind the front face: "this repeats"
        for k in (2, 1):
            ox, oy = x - k * 9, y - k * 7
            out.append(f'<rect x="{ox:.1f}" y="{oy:.1f}" width="{W}" height="{H}" fill="{color}" '
                       f'fill-opacity="{0.22 + 0.1 * (2 - k)}" stroke="{color}" stroke-width="1"/>')
    out.append(_poly([(x + W, y), (x + W + DX, y + DY), (x + W + DX, y + H + DY), (x + W, y + H)],
                     color, 0.45, f'stroke="{color}" stroke-width="1"'))
    out.append(_poly([(x, y), (x + W, y), (x + W + DX, y + DY), (x + DX, y + DY)],
                     color, 0.7, f'stroke="{color}" stroke-width="1"'))
    out.append(f'<rect class="front" x="{x:.1f}" y="{y:.1f}" width="{W}" height="{H}" fill="{color}" '
               f'fill-opacity="0.9" stroke="{color}" stroke-width="1"/>')
    out.append(f'<rect x="{x + 7:.1f}" y="{y + 7:.1f}" width="{W - 14}" height="{H - 14}" fill="#f4f4f5"/>')

    if loss:
        lines = [(b.label, 13, 500, "#18181b"), (b.detail, 11, 400, "#3f3f46")]
    else:
        lines = [(b.label, 13, 500, "#18181b"),
                 (f"{format_params(b.params)} params · {b.dtype}" if b.params else "no own parameters", 11, 400, "#3f3f46")]
        if b.shape_hint:
            lines.append((f"matrix {b.shape_hint}", 11, 400, "#3f3f46"))
        if b.detail:
            lines.append((b.detail, 11, 400, "#3f3f46"))
    lines.append((b.kind if not loss else "loss", 10, 500, color))
    for i, (text, size, weight, fill) in enumerate(lines[:5]):
        ty = y + 24 + (0 if i == 0 else 15 + (i - 1) * 14)
        out.append(f'<text x="{x + 15:.1f}" y="{ty:.1f}" fill="{fill}" font-family="{FONT}" '
                   f'font-size="{size}" font-weight="{weight}">{_esc(text[:34])}</text>')
    return "".join(out), H


def _arrow(x1, y1, x2, y2, color: str, dashed: bool = False, label: str | None = None) -> str:
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    marker = f"url(#arrow-{_marker_name(color)})"
    if abs(y1 - y2) < 1 or abs(x1 - x2) < 1:
        d = f"M {x1:.1f} {y1:.1f} L {x2:.1f} {y2:.1f}"
    else:  # elbow: down, across, down
        ym = (y1 + y2) / 2
        d = f"M {x1:.1f} {y1:.1f} L {x1:.1f} {ym:.1f} L {x2:.1f} {ym:.1f} L {x2:.1f} {y2:.1f}"
    text = ""
    if label:
        lx, ly = (x1 + x2) / 2, (y1 + y2) / 2 - 6
        text = (f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" fill="{color}" font-family="{FONT}" '
                f'font-size="11">{_esc(label)}</text>')
    return (f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"{dash} marker-end="{marker}"/>{text}')


def _marker_name(color: str) -> str:
    return {"#2563eb": "blue", "#7c3aed": "purple", "#c2410c": "orange", "#71717a": "gray"}.get(color, "gray")


def render(arch: Architecture, opts: Options | None = None) -> str:
    opts = opts or Options()
    blocks = arch.blocks
    # Layout: each stage starts a new row; rows hold up to PER_ROW blocks.
    rows: list[tuple[str, list[Block]]] = []
    for stage, bs in arch.stages:
        for i in range(0, len(bs), PER_ROW):
            rows.append((stage if i == 0 else "", bs[i:i + PER_ROW]))

    cell_w = W + DX + COL_GAP
    row_h = max(_block_height(b) for b in blocks) + ROW_GAP + 18
    top = 120
    left = MARGIN + 6
    pos: dict[str, tuple[float, float, int]] = {}
    stage_labels: list[tuple[float, float, str]] = []
    for r, (stage, bs) in enumerate(rows):
        y = top + r * row_h + (30 if stage else 0)
        if stage:
            stage_labels.append((left, y - 24, stage))
        for c, b in enumerate(bs):
            x = left + c * cell_w
            pos[b.id] = (x, y, _block_height(b))

    # The loss sits after the last block: same row when there is room, otherwise a new row.
    last_row_y = top + (len(rows) - 1) * row_h + (30 if rows[-1][0] else 0)
    last_row_len = len(rows[-1][1])
    loss_block = Block(id="loss", label=arch.loss_name, path="loss", kind="loss", params=0,
                       detail=arch.loss_detail)
    if last_row_len < PER_ROW:
        loss_x, loss_y = left + last_row_len * cell_w, last_row_y
    else:
        loss_x, loss_y = left, last_row_y + row_h
    pos[loss_block.id] = (loss_x, loss_y, _block_height(loss_block))

    width = int(left + PER_ROW * cell_w + MARGIN)
    bottom = max(y + h for x, y, h in pos.values())
    height = int(bottom + 100)

    body: list[str] = []
    arrows: list[str] = []
    loops: list[str] = []

    highlighted = {b.id for b in blocks if any(matches(b, k) for k in opts.highlight)}

    # Stage labels and blocks
    for x, y, text in stage_labels:
        body.append(f'<text x="{x:.1f}" y="{y:.1f}" fill="#52525b" font-family="{FONT}" font-size="12" '
                    f'font-weight="500">{_esc(text)}</text>')
    for stage, bs in arch.stages:
        for b in bs:
            x, y, _ = pos[b.id]
            svg, _ = _block_svg(b, x, y, KIND_COLORS.get(b.kind, "#7c3aed"))
            hl = " hl" if b.id in highlighted else ""
            body.append(f'<g class="block kind-{b.kind}{hl}" id="block-{b.id}" data-path="{_esc(b.path)}" '
                        f'data-kind="{b.kind}" data-params="{b.params}">{svg}</g>')
    lx, ly, lh = pos["loss"]
    svg, _ = _block_svg(loss_block, lx, ly, KIND_COLORS["loss"], loss=True)
    body.append(f'<g class="block kind-loss" id="block-loss" data-path="loss" data-kind="loss" data-params="0">{svg}</g>')

    # Forward connectors between consecutive blocks in the sequence
    seq = [b for _, bs in arch.stages for b in bs] + [loss_block]
    for prev, nxt in zip(seq, seq[1:]):
        px, py, ph = pos[prev.id]
        nx, ny, nh = pos[nxt.id]
        if abs(py - ny) < 1:
            arrows.append(_arrow(px + W + DX, py + ph / 2 + DY / 2, nx, ny + nh / 2, KIND_COLORS["transformer"]))
        else:
            arrows.append(_arrow(px + W / 2, py + ph, nx + W / 2, ny, KIND_COLORS["transformer"]))

    # Cross-attention: encoder output feeds every decoder layer
    enc = [bs for stage, bs in arch.stages if "encoder" in stage.lower()]
    dec = [bs for stage, bs in arch.stages if "decoder" in stage.lower()]
    if enc and dec and enc[0] and dec[0]:
        ex, ey, eh = pos[enc[0][-1].id]
        dx_, dy_, dh = pos[dec[0][0].id]
        arrows.append(_arrow(ex + W / 2, ey + eh, dx_ + W / 2, dy_, CROSS_COLOR, dashed=True,
                             label="cross-attention memory"))

    # Gradient loops: loss back to every trainable block
    if opts.loops:
        # Gradients leave the top of the loss and curve back to the underside of each trainable block.
        lx_c, ly_c = lx + W / 2, ly
        targets = [b for b in blocks if b.kind in TRAINABLE and b.params]
        for i, b in enumerate(targets):
            bx, by, bh = pos[b.id]
            bxc, byc = bx + W / 2, by + bh
            up = ly_c - 56 - i * 7
            below = byc + 56 + i * 7
            loops.append(
                f'<path d="M {lx_c:.1f} {ly_c:.1f} C {lx_c:.1f} {up:.1f} {bxc:.1f} {below:.1f} {bxc:.1f} {byc:.1f}" '
                f'fill="none" stroke="{LOOP_COLOR}" stroke-width="1.6" stroke-dasharray="6 4" '
                f'marker-end="url(#arrow-orange)" opacity="0.85"/>')
        loops.append(f'<text x="{lx + W + DX + 14:.1f}" y="{ly + 24:.1f}" fill="{LOOP_COLOR}" font-family="{FONT}" '
                     f'font-size="12" font-weight="500">∂loss/∂θ to {len(targets)} trainable blocks</text>'
                     f'<text x="{lx + W + DX + 14:.1f}" y="{ly + 40:.1f}" fill="{LOOP_COLOR}" font-family="{FONT}" '
                     f'font-size="11">then one optimizer step per block</text>')

    # Activation overlay (trace)
    overlay: list[str] = []
    if opts.trace is not None:
        overlay.extend(_trace_overlay(opts.trace, blocks + [loss_block], pos, opts.frame, width))

    title = _esc(opts.title or f"{Path(arch.source).name}: architecture")
    subtitle = _esc(f"{arch.num_tensors} tensors · {format_params(arch.total_params)} parameters · "
                    f"{', '.join(arch.dtypes) or 'no dtype'} · loss {arch.loss_name}")
    legend = _legend(opts, height)
    root_class = " dim" if highlighted else ""

    head = (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="{title}" class="archviz{root_class}" id="archviz">\n'
        f'<title>{title}</title>\n'
        f'<style>\n'
        f'  text {{ font-family: {FONT}; }}\n'
        f'  .block .front {{ transition: stroke-width 0.2s; }}\n'
        f'  .block.hl .front {{ stroke: {HIGHLIGHT}; stroke-width: 5; }}\n'
        f'  .block.hl {{ filter: drop-shadow(0 0 6px {HIGHLIGHT}); }}\n'
        f'  .archviz.dim .block {{ opacity: 0.3; }}\n'
        f'  .archviz.dim .block.hl {{ opacity: 1; }}\n'
        f'  .block {{ cursor: pointer; }}\n'
        f'</style>\n'
        f'<defs>'
        + "".join(f'<marker id="arrow-{n}" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">'
                  f'<path d="M 0 0 L 7 3 L 0 6 Z" fill="{c}"/></marker>'
                  for n, c in [("blue", KIND_COLORS["transformer"]), ("purple", CROSS_COLOR),
                               ("orange", LOOP_COLOR), ("gray", KIND_COLORS["input"])])
        + '</defs>\n'
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>\n'
        f'<text x="{MARGIN}" y="36" fill="#18181b" font-family="{FONT}" font-size="20" font-weight="500">{title}</text>\n'
        f'<text x="{MARGIN}" y="58" fill="#3f3f46" font-family="{FONT}" font-size="12">{subtitle}</text>\n'
        f'<text x="{MARGIN}" y="76" fill="#71717a" font-family="{FONT}" font-size="11">'
        f'Isometric blocks, one per module. A stack ×N repeats its block. Click a block to highlight it. '
        f'Press l to toggle loops and space to pause the trace.</text>\n'
    )
    parts = [head]
    parts.append('<g id="connectors">' + "".join(arrows) + "</g>\n")
    parts.append('<g id="blocks">' + "".join(body) + "</g>\n")
    parts.append(f'<g id="loops" style="display:{"" if opts.loops else "none"}">' + "".join(loops) + "</g>\n")
    parts.append('<g id="activations">' + "".join(overlay) + "</g>\n")
    parts.append(legend)
    parts.append(_script())
    parts.append("</svg>\n")
    return "".join(parts)


def _trace_overlay(trace: Trace, blocks: list[Block], pos: dict, frame: int | None, width: int) -> list[str]:
    n = len(trace.frames)
    out: list[str] = []
    total_ms = n * trace.frame_ms
    for b in blocks:
        x, y, h = pos[b.id]
        values = []
        for f in trace.frames:
            v = 0.0
            for key, val in f["active"].items():
                if matches(b, key):
                    v = max(v, val)
            values.append(v)
        if not any(values):
            continue
        rect_attrs = f'x="{x - 4:.1f}" y="{y - 4:.1f}" width="{W + 8}" height="{h + 8}" rx="4" fill="{HIGHLIGHT}" stroke="{HIGHLIGHT}"'
        if frame is not None:
            i = max(0, min(frame, n - 1))
            out.append(f'<rect {rect_attrs} fill-opacity="{values[i] * 0.85:.2f}" stroke-opacity="{min(1, values[i]):.2f}"/>')
        else:
            opac = ";".join(f"{v * 0.85:.2f}" for v in values) + f";{values[-1] * 0.85:.2f}"
            times = ";".join(f"{i / n:.4f}" for i in range(n)) + ";1"
            out.append(f'<rect {rect_attrs} fill-opacity="0">'
                       f'<animate attributeName="fill-opacity" values="{opac}" keyTimes="{times}" calcMode="discrete" '
                       f'dur="{total_ms}ms" repeatCount="indefinite"/></rect>')
    if frame is None:
        for i, f in enumerate(trace.frames):
            vals = ";".join("1" if j == i else "0" for j in range(n))
            times = ";".join(f"{j / n:.4f}" for j in range(n))
            out.append(f'<text x="{width - MARGIN}" y="36" text-anchor="end" fill="#18181b" font-family="{FONT}" '
                       f'font-size="16" opacity="0">{_esc(f["label"])}'
                       f'<animate attributeName="opacity" values="{vals}" keyTimes="{times}" calcMode="discrete" '
                       f'dur="{total_ms}ms" repeatCount="indefinite"/></text>')
    else:
        out.append(f'<text x="{width - MARGIN}" y="36" text-anchor="end" fill="#18181b" font-family="{FONT}" '
                   f'font-size="16">{_esc(trace.frames[max(0, min(frame, n - 1))]["label"])}</text>')
    return out


def _legend(opts: Options, height: int) -> str:
    y = height - 26
    items = [("Transformer, attention, MLP", KIND_COLORS["transformer"], False),
             ("Inputs and embeddings", KIND_COLORS["input"], False),
             ("Norms", KIND_COLORS["norm"], False),
             ("Head", KIND_COLORS["head"], False),
             ("Loss", KIND_COLORS["loss"], False),
             ("Cross-attention", CROSS_COLOR, True)]
    if opts.loops:
        items.append(("Gradient loops", LOOP_COLOR, True))
    x = MARGIN
    out = []
    for label, color, dashed in items:
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        out.append(f'<line x1="{x}" y1="{y - 4}" x2="{x + 26}" y2="{y - 4}" stroke="{color}" stroke-width="3"{dash}/>')
        out.append(f'<text x="{x + 32}" y="{y}" fill="#3f3f46" font-family="{FONT}" font-size="12">{_esc(label)}</text>')
        x += 40 + 7 * len(label)
    return "<g id=\"legend\">" + "".join(out) + "</g>\n"


def _script() -> str:
    return """<script><![CDATA[
(function () {
  var svg = document.getElementById('archviz');
  var loops = document.getElementById('loops');
  var paused = false;
  document.addEventListener('click', function (e) {
    var block = e.target.closest && e.target.closest('.block');
    if (!block) return;
    block.classList.toggle('hl');
    svg.classList.toggle('dim', !!document.querySelector('.block.hl'));
  });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'l' || e.key === 'L') {
      loops.style.display = loops.style.display === 'none' ? '' : 'none';
    } else if (e.key === ' ' && svg.pauseAnimations) {
      e.preventDefault();
      if (paused) { svg.unpauseAnimations(); } else { svg.pauseAnimations(); }
      paused = !paused;
    }
  });
})();
]]></script>
"""
