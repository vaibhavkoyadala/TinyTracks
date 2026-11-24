"""Image input for Qwen2.5-VL and Qwen3-VL: decoding, resizing, patch layout and position indices.

Follows Hugging Face's image processors and the position rules of each family, so the inputs
match what the model was trained on. Everything here is host-side (numpy and Pillow).

Only `data:` URLs are accepted. The server never fetches a URL, so a request cannot make it
connect to anything else.
"""
from __future__ import annotations

import base64
import binascii
import io
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

TEMPORAL = 2
WINDOW = 112
MAX_IMAGE_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class ImageSpec:
    """How a model family turns pixels into patches. Each family's preprocessor_config.json sets these."""
    patch: int
    merge: int
    min_pixels: int
    max_pixels: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    @property
    def factor(self) -> int:
        return self.patch * self.merge      # images are resized to multiples of this


# Qwen2.5-VL. max_pixels is capped at about 768 LLM tokens; the model config allows far more.
QWEN25_SPEC = ImageSpec(
    patch=14, merge=2, min_pixels=56 * 56, max_pixels=768 * 28 * 28,
    mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711),
)
# Qwen3-VL. Its checkpoint allows 16M pixels; serving caps it at about 1024 LLM tokens to bound vision attention.
QWEN3_SPEC = ImageSpec(
    patch=16, merge=2, min_pixels=65536, max_pixels=1024 * 32 * 32,
    mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
)

# Defaults kept for the Qwen2.5-VL names that other modules import.
PATCH = QWEN25_SPEC.patch
MERGE = QWEN25_SPEC.merge
FACTOR = QWEN25_SPEC.factor
MIN_PIXELS = QWEN25_SPEC.min_pixels
MAX_PIXELS = QWEN25_SPEC.max_pixels
MEAN = np.array(QWEN25_SPEC.mean, dtype=np.float32)
STD = np.array(QWEN25_SPEC.std, dtype=np.float32)


class ImageError(ValueError):
    pass


def decode_data_url(url: str) -> Image.Image:
    if not url.startswith("data:"):
        raise ImageError("images must be sent as data: URLs (base64). Remote URLs are not fetched.")
    header, sep, payload = url.partition(",")
    if not sep or ";base64" not in header:
        raise ImageError("image data URLs must be base64 encoded")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageError("image data is not valid base64") from exc
    if len(raw) > MAX_IMAGE_BYTES:
        raise ImageError("an image is larger than 20 MB")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as exc:  # Pillow raises several types for bad files
        raise ImageError("could not decode the image; send PNG, JPEG or WebP") from exc
    return img.convert("RGB")


def smart_resize(height: int, width: int, spec: ImageSpec = QWEN25_SPEC) -> tuple[int, int]:
    """Round both sides to multiples of the family's factor, keeping the pixel count inside its limits."""
    factor = spec.factor
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > spec.max_pixels:
        beta = math.sqrt((height * width) / spec.max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < spec.min_pixels:
        beta = math.sqrt(spec.min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class ImageInput:
    """One image ready for the vision tower: patch rows plus the (t, h, w) grid."""

    def __init__(self, pixels: np.ndarray, grid: tuple[int, int, int], merge: int = MERGE):
        self.pixels = pixels    # (t*h*w, 3*2*patch*patch), rows in merge-block order
        self.grid = grid        # (t, h_patches, w_patches)
        self.merge = merge

    @property
    def llm_tokens(self) -> int:
        t, h, w = self.grid
        return t * (h // self.merge) * (w // self.merge)


def preprocess(img: Image.Image, spec: ImageSpec = QWEN25_SPEC) -> ImageInput:
    width, height = img.size
    new_h, new_w = smart_resize(height, width, spec)
    resized = img.resize((new_w, new_h), resample=Image.BICUBIC)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - np.array(spec.mean, dtype=np.float32)) / np.array(spec.std, dtype=np.float32)
    chw = arr.transpose(2, 0, 1)                      # (3, H, W)
    frames = np.stack([chw, chw], axis=0)             # a still image is repeated to fill the temporal patch
    p, m = spec.patch, spec.merge
    grid_t, grid_h, grid_w = 1, new_h // p, new_w // p
    patches = frames.reshape(TEMPORAL, 3, grid_h // m, m, p, grid_w // m, m, p)
    # -> (block_h, block_w, merge_h, merge_w, C, T, P, P): tokens in merge-block order, features in (C, T, P, P)
    patches = patches.transpose(2, 5, 3, 6, 1, 0, 4, 7)
    pixels = patches.reshape(grid_t * grid_h * grid_w, 3 * TEMPORAL * p * p)
    return ImageInput(np.ascontiguousarray(pixels, dtype=np.float32), (grid_t, grid_h, grid_w), m)


def vision_positions(grid: tuple[int, int, int], merge: int = MERGE) -> np.ndarray:
    """(h, w) position of every patch, in merge-block order, as the vision rotary embedding expects."""
    _, h, w = grid
    hpos = np.arange(h)[:, None].repeat(w, axis=1)
    wpos = np.arange(w)[None, :].repeat(h, axis=0)
    def block_order(a):
        return a.reshape(h // merge, merge, w // merge, merge).transpose(0, 2, 1, 3).reshape(-1)
    return np.stack([block_order(hpos), block_order(wpos)], axis=-1)


def window_index(grid: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Order that groups merged tokens into attention windows, and the window boundaries in patch tokens."""
    _, h, w = grid
    llm_h, llm_w = h // MERGE, w // MERGE
    vit_window = WINDOW // MERGE // PATCH          # 4 merged tokens per window side
    index = np.arange(llm_h * llm_w).reshape(llm_h, llm_w)
    pad_h = (vit_window - llm_h % vit_window) % vit_window
    pad_w = (vit_window - llm_w % vit_window) % vit_window
    padded = np.pad(index, ((0, pad_h), (0, pad_w)), constant_values=-100)
    nh, nw = (llm_h + pad_h) // vit_window, (llm_w + pad_w) // vit_window
    padded = padded.reshape(nh, vit_window, nw, vit_window).transpose(0, 2, 1, 3).reshape(nh * nw, vit_window * vit_window)
    seqlens = (padded != -100).sum(axis=1)
    order = padded.reshape(-1)
    order = order[order != -100]
    cu = np.concatenate([[0], np.cumsum(seqlens * MERGE * MERGE)]).astype(np.int64)
    cu = np.unique(cu)
    return order, cu


def mrope_positions(ids: list[int], grids: list[tuple[int, int, int]], image_pad: int, merge: int = MERGE) -> np.ndarray:
    """(3, T) temporal/height/width positions for a prompt that contains image placeholders.

    Text advances all three indices together. An image starts at the current position and advances
    the text position by max(llm_h, llm_w), not by the number of tokens.
    """
    out = np.zeros((3, len(ids)), dtype=np.int64)
    current = 0
    image_no = 0
    i = 0
    n = len(ids)
    while i < n:
        if ids[i] != image_pad:
            out[:, i] = current
            current += 1
            i += 1
            continue
        t, h, w = grids[image_no]
        image_no += 1
        llm_h, llm_w = h // merge, w // merge
        count = t * llm_h * llm_w
        ts = np.repeat(np.arange(t), llm_h * llm_w) + current
        hs = np.tile(np.repeat(np.arange(llm_h), llm_w), t) + current
        ws = np.tile(np.tile(np.arange(llm_w), llm_h), t) + current
        out[0, i:i + count] = ts
        out[1, i:i + count] = hs
        out[2, i:i + count] = ws
        current += max(llm_h, llm_w)
        i += count
    if image_no != len(grids):
        raise ImageError("the prompt does not contain every image placeholder")
    return out
