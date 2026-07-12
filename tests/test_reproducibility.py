# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Reproducibility tests: the same seed must yield byte-identical output.

A benchmark others compare numbers on needs fixed datasets — with
``Config(seed=...)`` two runs of the same configuration must produce the
same bytes in every file, and different seeds must diverge.
"""

from __future__ import annotations

import hashlib
import os

from gen_fraud_graph.config import Config
from gen_fraud_graph.generator import FraudGraphGenerator


def _tree_digest(root: str) -> dict[str, str]:
    """Map each file's path relative to ``root`` to its content hash."""
    digests: dict[str, str] = {}
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root)
            with open(path, "rb") as fh:
                digests[rel] = hashlib.sha256(fh.read()).hexdigest()
    return digests


def _run(tmp_dir: str, sub: str, seed: int | None) -> dict[str, str]:
    out = os.path.join(tmp_dir, sub)
    cfg = Config(
        scale_factor=0.0001,
        seed=seed,
        embedding_provider="fake",
        embedding_dim=8,
        workers=1,
        batches_per_worker=1,
        output_dir=out,
    )
    FraudGraphGenerator(cfg).run()
    return _tree_digest(out)


class TestSeededGeneration:
    def test_same_seed_is_byte_identical(self, tmp_dir):
        first = _run(tmp_dir, "a", seed=42)
        second = _run(tmp_dir, "b", seed=42)
        assert first.keys() == second.keys()
        mismatched = [rel for rel in first if first[rel] != second[rel]]
        assert not mismatched, f"files differ despite identical seed: {mismatched}"

    def test_different_seed_diverges(self, tmp_dir):
        first = _run(tmp_dir, "a", seed=1)
        second = _run(tmp_dir, "b", seed=2)
        assert first.keys() == second.keys()
        assert any(first[rel] != second[rel] for rel in first)
