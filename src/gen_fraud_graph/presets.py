# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Canonical benchmark presets — datasets as code.

A preset pins every parameter that shapes the output bytes: scale, master
seed, worker/chunk layout, embedding dimensionality, and per-typology
pattern counts (tuned to ~1% fraud edges). Anyone running the same preset
on any machine regenerates the exact same files, so benchmark numbers are
comparable without shipping multi-gigabyte artifacts — a SHA-256 manifest
per preset (``presets/<name>.sha256``) lets consumers verify their copy.

Only ``output_dir``, ``output_format``, and ``compress`` remain free: they
change packaging, not content identity (the manifest covers the default
CSV format).
"""

from __future__ import annotations

from typing import Any

from gen_fraud_graph.config import Config

# Pattern counts scale linearly across presets so the fraud mix and rate
# (~1% of edges) stay comparable; seeds differ so the presets are
# independent draws, not nested subsets.
PRESETS: dict[str, dict[str, Any]] = {
    "momo-100k": {
        "scale_factor": 0.01,  # 100K wallets, 900K transactions
        "seed": 100_001,
        "workers": 4,
        "batches_per_worker": 1,
        "embedding_provider": "fake",
        "embedding_dim": 32,
        "num_fraud_rings": 300,
        "num_structuring_patterns": 200,
        "num_mobile_money_patterns": 300,
        "num_trade_based_ml_patterns": 120,
        "num_hawala_patterns": 160,
        "num_sim_swap_patterns": 240,
        "num_overdraft_mule_patterns": 140,
    },
    "momo-1m": {
        "scale_factor": 0.1,  # 1M wallets, 9M transactions
        "seed": 1_000_001,
        "workers": 8,
        "batches_per_worker": 1,
        "embedding_provider": "fake",
        "embedding_dim": 32,
        "num_fraud_rings": 3_000,
        "num_structuring_patterns": 2_000,
        "num_mobile_money_patterns": 3_000,
        "num_trade_based_ml_patterns": 1_200,
        "num_hawala_patterns": 1_600,
        "num_sim_swap_patterns": 2_400,
        "num_overdraft_mule_patterns": 1_400,
    },
    "momo-10m": {
        "scale_factor": 1.0,  # 10M wallets, 90M transactions
        "seed": 10_000_001,
        "workers": 16,
        "batches_per_worker": 2,
        "embedding_provider": "fake",
        "embedding_dim": 32,
        "num_fraud_rings": 30_000,
        "num_structuring_patterns": 20_000,
        "num_mobile_money_patterns": 30_000,
        "num_trade_based_ml_patterns": 12_000,
        "num_hawala_patterns": 16_000,
        "num_sim_swap_patterns": 24_000,
        "num_overdraft_mule_patterns": 14_000,
    },
}


def preset_config(
    name: str,
    *,
    output_dir: str = "data",
    output_format: str = "csv",
    compress: bool = False,
) -> Config:
    """Build the :class:`Config` for a named preset.

    Raises:
        KeyError: If ``name`` is not a known preset.
    """
    if name not in PRESETS:
        known = ", ".join(sorted(PRESETS))
        raise KeyError(f"unknown preset {name!r} — available: {known}")
    return Config(
        output_dir=output_dir,
        output_format=output_format,  # type: ignore[arg-type]
        compress=compress,
        **PRESETS[name],
    )
