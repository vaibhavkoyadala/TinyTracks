# archviz

Draws a saved tensor file as a controllable SVG of its architecture: isometric blocks for each
module, stacked blocks for repeated layers, connectors, the loss, and optional gradient loops.
The style follows `lossfunk/docs/*-architecture.svg`.

Standard library only. The file's header is read, and the weights are never loaded.

## Use

```sh
python tools/archviz/archviz.py model.safetensors                        # -> model.architecture.svg
python tools/archviz/archviz.py model.safetensors -o out.svg --loops     # gradient loops to every trainable block
python tools/archviz/archviz.py model.safetensors --loss MSE             # override the inferred loss
python tools/archviz/archviz.py model.safetensors --highlight layers,encoder
python tools/archviz/archviz.py model.safetensors --trace run.json       # animated activations
python tools/archviz/archviz.py model.safetensors --trace run.json --frame 3   # one frame, as a still
```

Inputs: `.safetensors` and `.npz`.

## How the diagram is built

- Tensor names are split on `.` and put in a tree. Wrapper levels (`model`, `transformer`) are skipped.
- A node whose children are numbered (`layers.0` … `layers.23`) becomes one block with `×24`.
- A node with its own sub-structure (`encoder`, `decoder`) becomes a stage, drawn on its own row.
- Blocks are colored by kind: input (gray), transformer/attention/MLP (blue), norm (slate), head (green), other (purple). Loss is orange.
- The loss comes from the output width. A vocabulary-sized output (≥ 1000 classes) gives cross-entropy. Anything else gives MSE.
- A model with no output tensor (tied embeddings, e.g. Qwen2.5-0.5B or Whisper) gets a virtual `lm head (tied)` block.
- Encoder → decoder pairs get a dashed cross-attention arrow.

## Controls

After the SVG is written, these work in a browser:

- Click a block to highlight it. Highlighted blocks stay lit and the rest dim.
- `l` shows or hides the gradient loops.
- Space pauses or resumes the trace animation.

The same controls are available from the command line through `--highlight`, `--loops` and `--trace`.
Highlight by block id, path, label, bare name (`layers`), or stage name (`encoder`).

## Trace format

A trace drives the activation overlay. Each frame sets per-block values between 0 and 1:

```json
{
  "frame_ms": 400,
  "frames": [
    {"label": "token 0 · embed",  "active": {"embed_tokens": 1.0}},
    {"label": "token 0 · layers", "active": {"layers": 0.9, "embed_tokens": 0.3}},
    {"label": "token 0 · loss",   "active": {"loss": 1.0}}
  ]
}
```

Keys name blocks the same way `--highlight` does. The values are yours to define. For example, normalized
activation norms from a forward pass, or attention entropy.

Animations are SMIL. They run in browsers, but not in macOS Quick Look, which shows the first frame.
