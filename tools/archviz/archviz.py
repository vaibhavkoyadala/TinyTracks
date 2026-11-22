#!/usr/bin/env python3
"""Draw a saved tensor file as a controllable SVG of its architecture.

    python archviz.py model.safetensors                      # blocks, connectors, loss
    python archviz.py model.safetensors --loops              # plus gradient loops to every trainable block
    python archviz.py model.safetensors --loss "MSE"         # override the inferred loss
    python archviz.py model.safetensors --highlight layers,norm
    python archviz.py model.safetensors --trace run.json     # animate block activations (SMIL)
    python archviz.py model.safetensors --trace run.json --frame 3   # freeze one frame, for export

A trace is JSON:
    {"frame_ms": 400,
     "frames": [{"label": "token 0", "active": {"layers": 0.9, "embed_tokens": 0.2}},
                {"label": "token 1", "active": {"layers": 0.8}}]}
Keys name a block by id, path, label, or bare name (e.g. "layers" for "layers ×24").
Values are 0..1. Use them for any per-block signal, such as normalized activation norms.

Only the tensor file's header is read. The weights are never loaded.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arch import build  # noqa: E402
from svg import Options, Trace, render  # noqa: E402
from tensorfile import UnsupportedTensorFile, read_tensor_infos  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draw a saved tensor file as a controllable architecture SVG.")
    parser.add_argument("tensor_file", type=Path, help=".safetensors or .npz file")
    parser.add_argument("-o", "--output", type=Path, help="output SVG (default: <file>.architecture.svg)")
    parser.add_argument("--loss", default="auto",
                        help="loss name to draw, or 'auto' (cross-entropy for vocabulary-sized outputs, else MSE)")
    parser.add_argument("--loops", action="store_true", help="draw gradient loops from the loss to every trainable block")
    parser.add_argument("--highlight", default="", help="comma-separated block names to highlight")
    parser.add_argument("--trace", type=Path, help="JSON file of per-block activations over frames")
    parser.add_argument("--frame", type=int, help="with --trace: render this frame as a still image")
    parser.add_argument("--title", help="title shown at the top of the diagram")
    args = parser.parse_args(argv)

    try:
        infos = read_tensor_infos(args.tensor_file)
    except (OSError, UnsupportedTensorFile, ValueError) as exc:
        print(f"archviz: {exc}", file=sys.stderr)
        return 2
    if not infos:
        print("archviz: the file contains no tensors", file=sys.stderr)
        return 2

    trace = None
    if args.trace is not None:
        try:
            trace = Trace.load(args.trace)
        except (OSError, ValueError) as exc:
            print(f"archviz: bad trace: {exc}", file=sys.stderr)
            return 2
        if args.frame is not None and not 0 <= args.frame < len(trace.frames):
            print(f"archviz: --frame must be between 0 and {len(trace.frames) - 1}", file=sys.stderr)
            return 2

    architecture = build(infos, str(args.tensor_file), loss=args.loss)
    options = Options(
        loops=args.loops,
        highlight={h.strip() for h in args.highlight.split(",") if h.strip()},
        trace=trace,
        frame=args.frame,
        title=args.title,
    )
    output = args.output or args.tensor_file.with_name(args.tensor_file.stem + ".architecture.svg")
    output.write_text(render(architecture, options))
    print(f"wrote {output} ({architecture.num_tensors} tensors, {len(architecture.blocks)} blocks, loss: {architecture.loss_name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
