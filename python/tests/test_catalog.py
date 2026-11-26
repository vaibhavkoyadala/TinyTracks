"""Catalog entries that the serving code relies on.

Run with: python/venv/bin/python -m unittest discover -s python/tests -v
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from catalog import CATALOG, CATALOG_BY_ID  # noqa: E402
from serving import VISION_ARCHITECTURES  # noqa: E402

SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CatalogTest(unittest.TestCase):
    def test_ids_are_unique(self):
        ids = [m["id"] for m in CATALOG]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_file_is_pinned_to_its_revision_with_a_digest(self):
        for model in CATALOG:
            for f in model["files"]:
                with self.subTest(model=model["id"], file=f["filename"]):
                    self.assertIn(model["revision"], f["source_url"])
                    self.assertRegex(f["sha256"], SHA256)
                    self.assertGreater(f["size_bytes"], 0)

    def test_qwen3_vl_4b_is_a_servable_computer_use_vision_model(self):
        model = CATALOG_BY_ID["qwen3-vl-4b-instruct"]
        self.assertEqual(model["architecture"], "Qwen3VLForConditionalGeneration")
        self.assertIn(model["architecture"], VISION_ARCHITECTURES)
        self.assertIn("computer-use", model["description"])
        self.assertEqual(len([f for f in model["files"] if f["filename"].endswith(".safetensors")]), 2)


if __name__ == "__main__":
    unittest.main()
