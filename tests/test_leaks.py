# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for label leakage.

Injected fraud must not be separable from legitimate traffic on any single
column (description, timestamp, amount, or account-ID adjacency). These
tests pin the properties that keep the benchmark honest.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime

from gen_fraud_graph.embeddings import EmbeddingGenerator
from gen_fraud_graph.generator import FraudGraphGenerator
from gen_fraud_graph.schema import TRANSACTION_DESCRIPTIONS, TS_FMT
from gen_fraud_graph.typologies import FraudRingGenerator, StructuringGenerator


def _read_rows(path: str) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _all_fraud_rows(output_dir: str) -> list[dict]:
    return _read_rows(os.path.join(output_dir, "fraud", "transactions_fraud.csv"))


def _all_normal_rows(output_dir: str) -> list[dict]:
    rows: list[dict] = []
    tx_dir = os.path.join(output_dir, "transactions")
    for name in os.listdir(tx_dir):
        rows.extend(_read_rows(os.path.join(tx_dir, name)))
    return rows


class TestVocabularyLeak:
    def test_fraud_descriptions_use_shared_vocabulary(self, small_config):
        """No fraud-only wording: every injected description must come from
        the same pool legitimate transactions draw from."""
        FraudGraphGenerator(small_config).run()
        fraud_descs = {r["description"] for r in _all_fraud_rows(small_config.output_dir)}
        assert fraud_descs <= set(TRANSACTION_DESCRIPTIONS)

    def test_normal_descriptions_use_shared_vocabulary(self, small_config):
        FraudGraphGenerator(small_config).run()
        normal_descs = {r["description"] for r in _all_normal_rows(small_config.output_dir)}
        assert normal_descs <= set(TRANSACTION_DESCRIPTIONS)


class TestTimestampLeak:
    def test_normal_timestamps_vary_and_stay_in_window(self, small_config):
        """Legitimate traffic must form a real temporal background, not a
        single constant timestamp."""
        FraudGraphGenerator(small_config).run()
        stamps = {r["timestamp"] for r in _all_normal_rows(small_config.output_dir)}
        assert len(stamps) > 100

        start = datetime.strptime(small_config.sim_start_date, "%Y-%m-%d")
        for ts in stamps:
            dt = datetime.strptime(ts, TS_FMT)
            assert 0 <= (dt - start).days <= small_config.sim_days

    def test_fraud_timestamps_vary_and_stay_in_window(self, small_config):
        FraudGraphGenerator(small_config).run()
        stamps = {r["timestamp"] for r in _all_fraud_rows(small_config.output_dir)}
        assert len(stamps) > 100

        start = datetime.strptime(small_config.sim_start_date, "%Y-%m-%d")
        for ts in stamps:
            dt = datetime.strptime(ts, TS_FMT)
            # Multi-day pattern windows may run past the last start day but
            # must stay near the simulated window.
            assert 0 <= (dt - start).days <= small_config.sim_days + 30

    def test_fraud_cases_carry_time_windows(self, small_config):
        FraudGraphGenerator(small_config).run()
        cases = _read_rows(os.path.join(small_config.output_dir, "fraud", "fraud_cases.csv"))
        assert cases
        for case in cases:
            w_start = datetime.strptime(case["window_start"], TS_FMT)
            w_end = datetime.strptime(case["window_end"], TS_FMT)
            assert w_start <= w_end


class TestAmountLeak:
    def test_ring_amounts_are_not_constant(self, tmp_dir):
        """The old generator stamped 12,000,000.00 on every ring edge —
        a constant a single-feature classifier separates instantly."""
        emb = EmbeddingGenerator("fake", dim=8)
        gen = FraudRingGenerator(num_rings=10, depth_range=(4, 6))
        gen.generate(max_account_id=1_000, start_tx_id=0, embedder=emb, output_dir=tmp_dir)
        amounts = {r["amount"] for r in _all_fraud_rows(tmp_dir)}
        assert len(amounts) > 10

    def test_amounts_are_xof_integers(self, small_config):
        """XOF has no minor unit; every amount must be a whole number."""
        FraudGraphGenerator(small_config).run()
        for row in _all_fraud_rows(small_config.output_dir) + _all_normal_rows(
            small_config.output_dir
        ):
            assert float(row["amount"]) == int(float(row["amount"]))

    def test_fraud_and_normal_amount_ranges_overlap(self, small_config):
        """Marginal amount distributions must overlap: the normal maximum
        has to exceed the fraud minimum and vice versa."""
        FraudGraphGenerator(small_config).run()
        fraud = sorted(float(r["amount"]) for r in _all_fraud_rows(small_config.output_dir))
        normal = sorted(float(r["amount"]) for r in _all_normal_rows(small_config.output_dir))
        assert normal[-1] > fraud[0], "every fraud amount sits above the normal range"
        assert fraud[-1] > normal[0], "every fraud amount sits below the normal range"


class TestAccountAdjacencyLeak:
    def test_structuring_accounts_are_not_consecutive(self, tmp_dir):
        """Smurf IDs used to be coordinator+1..coordinator+n — detectable
        by integer adjacency alone."""
        emb = EmbeddingGenerator("fake", dim=8)
        gen = StructuringGenerator(num_patterns=10, smurfs_range=(5, 5))
        gen.generate(max_account_id=10_000, start_tx_id=0, embedder=emb, output_dir=tmp_dir)

        cases = _read_rows(os.path.join(tmp_dir, "fraud", "fraud_cases.csv"))
        consecutive_patterns = 0
        for case in cases:
            ids = sorted(int(a.removeprefix("acc_")) for a in case["involved_accounts"].split("|"))
            if all(b - a == 1 for a, b in zip(ids, ids[1:], strict=False)):
                consecutive_patterns += 1
        assert consecutive_patterns == 0
