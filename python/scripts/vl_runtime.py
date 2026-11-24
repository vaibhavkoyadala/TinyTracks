"""A loaded vision-language model (Qwen2.5-VL or Qwen3-VL): text and images in, text out.

The checkpoint's architecture picks the model family (qwen25vl_model or qwen3vl_model). Both families
expose the same interface, so the prompt handling and decode loop here are shared.

It follows the LoadedModel interface used by serving.py, with one difference: `prepare_prompt` also
takes the decoded images, and `generate` takes the prepared prompt rather than a plain id list.
Callers must run `generate` on the GpuWorker thread.
"""
from __future__ import annotations

import json
from pathlib import Path

import jinja2
import numpy as np
from tinygrad import Tensor, TinyJit
from tokenizers import Tokenizer

import qwen3vl_model
import qwen25vl_model
from vl_image import ImageInput, decode_data_url, mrope_positions, preprocess

IMAGE_PAD = "<|image_pad|>"
MAX_IMAGES = 8
FAMILIES = (qwen25vl_model, qwen3vl_model)


class ImageNotSupported(ValueError):
    pass


def family_for(architecture: str):
    for family in FAMILIES:
        if architecture in family.ARCHITECTURES:
            return family
    raise ImageNotSupported(f"no vision runtime for architecture {architecture}")


class VLPrompt:
    """Token ids with the image placeholders expanded, plus the image data for the vision tower."""

    __slots__ = ("ids", "images", "positions")

    def __init__(self, ids: list[int], images: list[ImageInput], positions: np.ndarray):
        self.ids = ids
        self.images = images
        self.positions = positions


class LoadedVLM:
    kind = "chat"
    accepts_images = True

    def __init__(self, model_id: str, model_dir: Path, max_context: int):
        self.model_id = model_id
        self.max_context = max_context
        raw_config = json.loads((model_dir / "config.json").read_text())
        self.family = family_for(raw_config["architectures"][0])
        vision_cfg, text_cfg, self.config = self.family.load_configs(model_dir)
        visual, text = self.family.split_weights(self.family.load_weights(model_dir))
        self.spec = self.family.IMAGE_SPEC
        self.deepstack_levels = len(vision_cfg.deepstack)
        self.vision = self.family.VisionTower(vision_cfg, visual)
        self.text = self.family.TextModel(text_cfg, text)
        self._step = TinyJit(self.text.decode_body)   # recorded on the first tokens, then replayed
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.image_pad = self.tokenizer.token_to_id(IMAGE_PAD)

        tok_cfg = json.loads((model_dir / "tokenizer_config.json").read_text())
        template_src = tok_cfg.get("chat_template")
        if not template_src:
            template_src = json.loads((model_dir / "chat_template.json").read_text())["chat_template"]
        env = jinja2.Environment()
        env.globals["raise_exception"] = _raise
        self.template = env.from_string(template_src)

        gen = json.loads((model_dir / "generation_config.json").read_text())
        eos = gen.get("eos_token_id")
        self.stop_ids = set(eos if isinstance(eos, list) else [eos] if eos is not None else [])
        self.stop_ids.update(t for t in (self.tokenizer.token_to_id("<|im_end|>"),) if t is not None)

    # ----- prompts ----------------------------------------------------------

    def prepare_prompt(self, messages: list[dict], images: list[str]) -> VLPrompt:
        """`messages` may hold {"type": "image"} parts; `images` are data URLs in the same order."""
        if len(images) > MAX_IMAGES:
            raise ValueError(f"at most {MAX_IMAGES} images per request")
        decoded = [preprocess(decode_data_url(url), self.spec) for url in images]
        text = self.template.render(messages=messages, add_generation_prompt=True)
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        if self.image_pad is None:
            raise ValueError("this tokenizer has no image placeholder token")

        expanded: list[int] = []
        image_no = 0
        for tid in ids:
            if tid == self.image_pad:
                if image_no >= len(decoded):
                    raise ValueError("the prompt has more image placeholders than images")
                expanded.extend([self.image_pad] * decoded[image_no].llm_tokens)
                image_no += 1
            else:
                expanded.append(tid)
        if image_no != len(decoded):
            raise ValueError("the prompt has fewer image placeholders than images")
        if len(expanded) >= self.max_context:
            raise ValueError(f"prompt has {len(expanded)} tokens, but the model context is {self.max_context}")
        positions = mrope_positions(expanded, [img.grid for img in decoded], self.image_pad, self.spec.merge)
        return VLPrompt(expanded, decoded, positions)

    # ----- generation -------------------------------------------------------

    def _embed_prompt(self, prompt: VLPrompt) -> tuple[Tensor, list[Tensor] | None]:
        """Text embeddings with each run of image placeholders replaced by its vision features.

        Also returns the DeepStack levels: one (1, T, hidden) tensor per level, zero at text positions.
        It is None when the family has no DeepStack levels or the prompt has no images.
        """
        levels = self.deepstack_levels if prompt.images else 0
        dim = self.text.cfg.hidden
        main: list[Tensor] = []
        per_level: list[list[Tensor]] = [[] for _ in range(levels)]
        ids = prompt.ids
        image_no = 0
        i = 0
        while i < len(ids):
            if ids[i] == self.image_pad:
                img = prompt.images[image_no]
                feats = self.vision(img)
                main.append(feats[0].reshape(1, img.llm_tokens, -1))
                for k in range(levels):
                    per_level[k].append(feats[1 + k].reshape(1, img.llm_tokens, -1))
                image_no += 1
                i += img.llm_tokens
                continue
            j = i
            while j < len(ids) and ids[j] != self.image_pad:
                j += 1
            main.append(self.text.embed(ids[i:j]).reshape(1, j - i, -1))
            for k in range(levels):
                per_level[k].append(Tensor.zeros(1, j - i, dim))
            i = j
        x = main[0].cat(*main[1:], dim=1) if len(main) > 1 else main[0]
        deepstack = None
        if levels:
            deepstack = [p[0].cat(*p[1:], dim=1) if len(p) > 1 else p[0] for p in per_level]
        return x, deepstack

    def _rope(self, positions: np.ndarray) -> tuple[Tensor, Tensor]:
        cos, sin = self.family.mrope_cos_sin(positions, self.text.inv_freq, self.text.cfg.mrope_section)
        return Tensor(cos).realize(), Tensor(sin).realize()

    def generate(self, prompt, max_new_tokens: int, temperature: float, info: dict):
        """Yield token ids. `prompt` is a VLPrompt (or a plain id list for text-only use)."""
        if not isinstance(prompt, VLPrompt):
            prompt = VLPrompt(list(prompt), [], mrope_positions(list(prompt), [], self.image_pad, self.spec.merge))
        info.update(finish_reason="length", completion_tokens=0, prefill_s=None, decode_s=None)
        import time as _time

        start = _time.perf_counter()
        x, deepstack = self._embed_prompt(prompt)
        cos, sin = self._rope(prompt.positions)
        logits, cache = self.text.forward(x, cos, sin, None, deepstack)
        next_pos = int(prompt.positions.max()) + 1
        info["prefill_s"] = _time.perf_counter() - start
        first_at = _time.perf_counter()

        # Move the prefill cache into fixed-size buffers, so every decode step has the same shapes.
        cfg = self.text.cfg
        max_len = self.max_context
        length = len(prompt.ids)
        flat: list[Tensor] = []
        for k, v in cache:
            pad = Tensor.zeros(1, cfg.kv_heads, max_len - length, cfg.head_dim)
            flat += [k.cat(pad, dim=2).realize(), v.cat(pad, dim=2).realize()]
        pos = length - 1

        produced = 0
        while pos + 1 < max_len:
            token = _sample(logits.numpy(), temperature)
            if token in self.stop_ids:
                info["finish_reason"] = "stop"
                break
            yield token
            produced += 1
            info["completion_tokens"] = produced
            if produced >= max_new_tokens:
                break
            pos += 1
            x = self.text.embed([token])
            cos, sin = self._rope(np.full((3, 1), next_pos, dtype=np.int64))
            onehot = np.zeros((1, 1, max_len, 1), dtype=np.float32)
            onehot[0, 0, pos, 0] = 1.0
            allowed = np.where(np.arange(max_len) <= pos, 0.0, -np.inf).astype(np.float32).reshape(1, 1, 1, max_len)
            out = self._step(x, cos.reshape(1, 1, 1, cfg.head_dim), sin.reshape(1, 1, 1, cfg.head_dim),
                             Tensor(onehot), Tensor(allowed), *flat)
            logits = out[0]
            flat = list(out[1:])
            next_pos += 1
        info["decode_s"] = _time.perf_counter() - first_at
        info["total_s"] = _time.perf_counter() - start

    def warmup(self) -> None:
        """Record the decode step now, so the first real request does not pay for it."""
        prompt = self.prepare_prompt([{"role": "user", "content": "Hello"}], [])
        list(self.generate(prompt, 3, 0.0, {}))

    def close(self) -> None:
        self.vision = None
        self.text = None


def _sample(logits: np.ndarray, temperature: float) -> int:
    row = logits.reshape(-1).astype(np.float64)
    if temperature <= 0:
        return int(np.argmax(row))
    row = (row - row.max()) / max(temperature, 1e-6)
    probs = np.exp(row)
    probs /= probs.sum()
    return int(np.random.default_rng().choice(len(probs), p=probs))


def _raise(message: str):
    raise jinja2.TemplateError(message)
