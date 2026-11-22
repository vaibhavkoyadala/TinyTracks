"""Read tensor names, dtypes and shapes from a saved tensor file without loading the data.

Supported: .safetensors (header only) and .npz (numpy archive, .npy headers only).
Standard library only.
"""
from __future__ import annotations

import ast
import json
import struct
import zipfile
from dataclasses import dataclass
from math import prod
from pathlib import Path

SAFETENSORS_DTYPES = {
    "F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16",
    "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8",
    "U8": "uint8", "BOOL": "bool",
}
NPY_DTYPES = {
    "f8": "float64", "f4": "float32", "f2": "float16",
    "i8": "int64", "i4": "int32", "i2": "int16", "i1": "int8",
    "u1": "uint8", "b1": "bool",
}


class UnsupportedTensorFile(ValueError):
    pass


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def numel(self) -> int:
        return prod(self.shape) if self.shape else 1


def read_tensor_infos(path: Path) -> list[TensorInfo]:
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        return _read_safetensors(path)
    if suffix == ".npz":
        return _read_npz(path)
    raise UnsupportedTensorFile(f"unsupported file type '{suffix}'; use .safetensors or .npz")


def _read_safetensors(path: Path) -> list[TensorInfo]:
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        if header_len > 100_000_000:
            raise UnsupportedTensorFile("the safetensors header is implausibly large")
        header = json.loads(f.read(header_len))
    infos = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype = SAFETENSORS_DTYPES.get(meta["dtype"], meta["dtype"].lower())
        infos.append(TensorInfo(name, dtype, tuple(int(d) for d in meta["shape"])))
    return infos


def _read_npz(path: Path) -> list[TensorInfo]:
    infos = []
    with zipfile.ZipFile(path) as zf:
        for member in zf.namelist():
            if not member.endswith(".npy"):
                continue
            with zf.open(member) as f:
                magic = f.read(6)
                if magic != b"\x93NUMPY":
                    raise UnsupportedTensorFile(f"{member} is not a .npy entry")
                major = f.read(1)[0]
                f.read(1)  # minor version
                if major == 1:
                    (hlen,) = struct.unpack("<H", f.read(2))
                else:
                    (hlen,) = struct.unpack("<I", f.read(4))
                meta = ast.literal_eval(f.read(hlen).decode("latin1"))
            descr = meta["descr"].lstrip("<>=|")
            dtype = NPY_DTYPES.get(descr, descr)
            infos.append(TensorInfo(member[: -len(".npy")], dtype, tuple(int(d) for d in meta["shape"])))
    return infos
