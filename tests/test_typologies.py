# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Tests for gen_fraud_graph.typologies (fraud ring generation)."""

from __future__ import annotations

import csv
import os
import random
from datetime import datetime, timedelta

import pytest

from gen_fraud_graph.embeddings import EmbeddingGenerator
from gen_fraud_graph.typologies import (
    FraudRingGenerator,
    HawalaNetworkGenerator,
    OverdraftMuleGenerator,
    SIMSwapFraudGenerator,
    TradeBasedMLGenerator,
)


class TestFraudRings:
    def test_generate_creates_files(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=32)
        gen = FraudRingGenerator(num_rings=5, depth_range=(3, 5))
        n_tx, next_id = gen.generate(
            max_account_id=1000,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
            fmt="csv",
        )
        assert n_tx > 0
        assert os.path.exists(os.path.join(tmp_dir, "fraud", "transactions_fraud.csv"))
        assert os.path.exists(os.path.join(tmp_dir, "fraud", "fraud_cases.csv"))

    def test_fraud_cases_have_correct_columns(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=32)
        gen = FraudRingGenerator(num_rings=3, depth_range=(4, 4))
        gen.generate(
            max_account_id=100,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
        )
        with open(os.path.join(tmp_dir, "fraud", "fraud_cases.csv")) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 3
        assert "pattern_id" in rows[0]
        assert "involved_accounts" in rows[0]

    def test_rings_use_disjoint_accounts(self, tmp_dir):
        # Each ring must occupy its own accounts. Overlapping ranges merge two
        # rings into a single non-cycle component and make the per-ring
        # involved_accounts labels ambiguous.
        random.seed(0)
        emb = EmbeddingGenerator("fake", dim=8)
        gen = FraudRingGenerator(num_rings=15, depth_range=(4, 4))
        gen.generate(
            max_account_id=80,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
        )
        with open(os.path.join(tmp_dir, "fraud", "fraud_cases.csv")) as fh:
            rows = list(csv.DictReader(fh))
        seen: set[str] = set()
        for row in rows:
            accounts = row["involved_accounts"].split("|")
            assert seen.isdisjoint(
                accounts
            ), f"{row['pattern_id']} reuses accounts from an earlier ring"
            seen.update(accounts)


class TestTypologiesExtra:
    def test_neptune_format(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=16)
        gen = FraudRingGenerator(num_rings=2, depth_range=(3, 3))
        n_tx, _ = gen.generate(
            max_account_id=50,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
            fmt="neptune",
        )
        assert n_tx > 0
        path = os.path.join(tmp_dir, "fraud", "transactions_fraud.csv")
        with open(path) as fh:
            header = next(csv.reader(fh))
        assert "~from" in header

    def test_oversubscribed_rings_raise(self, tmp_dir):
        """When the rings need more distinct accounts than exist they can't be
        packed disjointly, so generate() must raise rather than emit rings that
        reference nonexistent accounts."""
        emb = EmbeddingGenerator("fake", dim=16)
        gen = FraudRingGenerator(num_rings=2, depth_range=(4, 4))
        with pytest.raises(ValueError, match="distinct"):
            gen.generate(
                max_account_id=3,
                start_tx_id=0,
                embedder=emb,
                output_dir=tmp_dir,
                fmt="csv",
            )

    def test_compress(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=16)
        gen = FraudRingGenerator(num_rings=2, depth_range=(3, 3))
        gen.generate(
            max_account_id=50,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
            fmt="csv",
            compress=True,
        )
        assert os.path.exists(os.path.join(tmp_dir, "fraud", "fraud_cases.csv.zip"))


class TestTradeBasedML:
    def test_generate_creates_files(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=32)
        gen = TradeBasedMLGenerator(num_patterns=5, intermediaries_range=(3, 5))
        n_tx, _ = gen.generate(
            max_account_id=1000,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
            fmt="csv",
        )
        assert n_tx > 0
        assert os.path.exists(os.path.join(tmp_dir, "fraud", "transactions_fraud.csv"))
        assert os.path.exists(os.path.join(tmp_dir, "fraud", "fraud_cases.csv"))

    def test_fraud_cases_have_correct_pattern_type(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=32)
        gen = TradeBasedMLGenerator(num_patterns=3, intermediaries_range=(3, 5))
        gen.generate(
            max_account_id=1000,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
        )
        with open(os.path.join(tmp_dir, "fraud", "fraud_cases.csv")) as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 3
        assert all(row["pattern_type"] == "trade_based_ml" for row in rows)

    def test_involved_accounts_positional_structure(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=8)
        gen = TradeBasedMLGenerator(num_patterns=1, intermediaries_range=(3, 3))
        gen.generate(
            max_account_id=1000,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
        )
        with open(os.path.join(tmp_dir, "fraud", "fraud_cases.csv")) as fh:
            row = next(csv.DictReader(fh))
        accounts = row["involved_accounts"].split("|")
        assert len(accounts) == 6  # exporter, shell_importer, 3 intermediaries, beneficiary
        assert int(row["depth"]) == 3

    def test_neptune_format(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=16)
        gen = TradeBasedMLGenerator(num_patterns=2, intermediaries_range=(3, 3))
        n_tx, _ = gen.generate(
            max_account_id=100,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
            fmt="neptune",
        )
        assert n_tx > 0
        path = os.path.join(tmp_dir, "fraud", "transactions_fraud.csv")
        with open(path) as fh:
            header = next(csv.reader(fh))
        assert "~from" in header

    def test_insufficient_accounts_returns_zero(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=8)
        gen = TradeBasedMLGenerator(num_patterns=5, intermediaries_range=(3, 5))
        n_tx, next_id = gen.generate(
            max_account_id=3,
            start_tx_id=42,
            embedder=emb,
            output_dir=tmp_dir,
        )
        assert (n_tx, next_id) == (0, 42)

    def test_compress_still_appends_plain_csv(self, tmp_dir):
        # TBML runs after FraudRingGenerator in the pipeline and must append
        # to the shared fraud files rather than fragment them into separate
        # zip files, so it ignores `compress` by design (see StructuringGenerator
        # and MobileMoneyFraudGenerator, which do the same).
        emb = EmbeddingGenerator("fake", dim=8)
        gen = TradeBasedMLGenerator(num_patterns=2, intermediaries_range=(3, 3))
        gen.generate(
            max_account_id=100,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
            compress=True,
        )
        assert os.path.exists(os.path.join(tmp_dir, "fraud", "fraud_cases.csv"))
        assert not os.path.exists(os.path.join(tmp_dir, "fraud", "fraud_cases.csv.zip"))


class TestHawalaNetwork:
    def test_generate_creates_files(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=32)
        gen = HawalaNetworkGenerator(num_patterns=5)
        n_tx, _ = gen.generate(
            max_account_id=1000,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
            fmt="csv",
        )
        assert n_tx > 0
        assert os.path.exists(os.path.join(tmp_dir, "fraud", "transactions_fraud.csv"))
        assert os.path.exists(os.path.join(tmp_dir, "fraud", "fraud_cases.csv"))

    def test_fraud_cases_pattern_type_and_positional_structure(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=32)
        gen = HawalaNetworkGenerator(num_patterns=10)
        gen.generate(
            max_account_id=1000,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
        )
        with open(os.path.join(tmp_dir, "fraud", "fraud_cases.csv")) as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 10
        for row in rows:
            assert row["pattern_type"] == "hawala_network"
            assert len(row["involved_accounts"].split("|")) == 4
            assert int(row["depth"]) in (3, 4)

    def test_settlement_probability_zero_never_emits_reverse_edge(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=8)
        gen = HawalaNetworkGenerator(num_patterns=10, settlement_probability=0.0)
        gen.generate(
            max_account_id=1000,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
        )
        with open(os.path.join(tmp_dir, "fraud", "fraud_cases.csv")) as fh:
            rows = list(csv.DictReader(fh))
        assert all(int(row["depth"]) == 3 for row in rows)

    def test_settlement_probability_one_always_emits_reverse_edge(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=8)
        gen = HawalaNetworkGenerator(num_patterns=10, settlement_probability=1.0)
        gen.generate(
            max_account_id=1000,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
        )
        with open(os.path.join(tmp_dir, "fraud", "fraud_cases.csv")) as fh:
            rows = list(csv.DictReader(fh))
        assert all(int(row["depth"]) == 4 for row in rows)

    def test_neptune_format(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=16)
        gen = HawalaNetworkGenerator(num_patterns=2)
        n_tx, _ = gen.generate(
            max_account_id=100,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
            fmt="neptune",
        )
        assert n_tx > 0
        path = os.path.join(tmp_dir, "fraud", "transactions_fraud.csv")
        with open(path) as fh:
            header = next(csv.reader(fh))
        assert "~from" in header

    def test_insufficient_accounts_returns_zero(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=8)
        gen = HawalaNetworkGenerator(num_patterns=5)
        n_tx, next_id = gen.generate(
            max_account_id=3,
            start_tx_id=42,
            embedder=emb,
            output_dir=tmp_dir,
        )
        assert (n_tx, next_id) == (0, 42)


class TestSIMSwap:
    def test_generate_creates_files(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=32)
        gen = SIMSwapFraudGenerator(num_patterns=5, num_agents_range=(3, 6))
        n_tx, _ = gen.generate(
            max_account_id=1000,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
            fmt="csv",
        )
        assert n_tx > 0
        assert os.path.exists(os.path.join(tmp_dir, "fraud", "transactions_fraud.csv"))
        assert os.path.exists(os.path.join(tmp_dir, "fraud", "fraud_cases.csv"))

    def test_fraud_cases_pattern_type_and_positional_structure(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=32)
        gen = SIMSwapFraudGenerator(num_patterns=10, num_agents_range=(3, 6))
        gen.generate(
            max_account_id=1000,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
        )
        with open(os.path.join(tmp_dir, "fraud", "fraud_cases.csv")) as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 10
        for row in rows:
            assert row["pattern_type"] == "sim_swap_takeover"
            accounts = row["involved_accounts"].split("|")
            assert len(accounts) == int(row["depth"]) + 1

    def test_amounts_are_near_identical(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=8)
        gen = SIMSwapFraudGenerator(
            num_patterns=1, num_agents_range=(4, 4), amount_jitter=0.05
        )
        gen.generate(
            max_account_id=1000,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
        )
        with open(os.path.join(tmp_dir, "fraud", "transactions_fraud.csv")) as fh:
            amounts = [float(r["amount"]) for r in csv.DictReader(fh)]
        assert len(amounts) == 4
        assert max(amounts) - min(amounts) <= 2 * gen.amount_jitter * max(amounts)

    def test_all_transactions_within_burst_window(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=8)
        gen = SIMSwapFraudGenerator(
            num_patterns=1, num_agents_range=(6, 6), burst_window_minutes=10
        )
        gen.generate(
            max_account_id=1000,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
        )
        with open(os.path.join(tmp_dir, "fraud", "transactions_fraud.csv")) as fh:
            timestamps = [
                datetime.strptime(r["timestamp"], "%Y-%m-%dT%H:%M:%S")
                for r in csv.DictReader(fh)
            ]
        span = max(timestamps) - min(timestamps)
        assert span <= timedelta(minutes=gen.burst_window_minutes)

    def test_neptune_format(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=16)
        gen = SIMSwapFraudGenerator(num_patterns=2, num_agents_range=(3, 3))
        n_tx, _ = gen.generate(
            max_account_id=100,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
            fmt="neptune",
        )
        assert n_tx > 0
        path = os.path.join(tmp_dir, "fraud", "transactions_fraud.csv")
        with open(path) as fh:
            header = next(csv.reader(fh))
        assert "~from" in header

    def test_insufficient_accounts_returns_zero(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=8)
        gen = SIMSwapFraudGenerator(num_patterns=5)
        n_tx, next_id = gen.generate(
            max_account_id=3,
            start_tx_id=42,
            embedder=emb,
            output_dir=tmp_dir,
        )
        assert (n_tx, next_id) == (0, 42)


class TestOverdraftMule:
    def test_generate_creates_files(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=32)
        gen = OverdraftMuleGenerator(num_patterns=5, num_mules_range=(5, 15))
        n_tx, _ = gen.generate(
            max_account_id=1000,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
            fmt="csv",
        )
        assert n_tx > 0
        assert os.path.exists(os.path.join(tmp_dir, "fraud", "transactions_fraud.csv"))
        assert os.path.exists(os.path.join(tmp_dir, "fraud", "fraud_cases.csv"))

    def test_fraud_cases_pattern_type_and_positional_structure(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=32)
        gen = OverdraftMuleGenerator(num_patterns=10, num_mules_range=(5, 15))
        gen.generate(
            max_account_id=1000,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
        )
        with open(os.path.join(tmp_dir, "fraud", "fraud_cases.csv")) as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 10
        for row in rows:
            assert row["pattern_type"] == "overdraft_mule_chain"
            accounts = row["involved_accounts"].split("|")
            assert len(accounts) == int(row["depth"]) + 2

    def test_consolidation_edge_amount_equals_sum_of_mule_amounts(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=8)
        gen = OverdraftMuleGenerator(num_patterns=1, num_mules_range=(5, 5))
        gen.generate(
            max_account_id=1000,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
        )
        with open(os.path.join(tmp_dir, "fraud", "fraud_cases.csv")) as fh:
            case = next(csv.DictReader(fh))
        accounts = case["involved_accounts"].split("|")
        collector, mules, agent = accounts[0], accounts[1:-1], accounts[-1]

        with open(os.path.join(tmp_dir, "fraud", "transactions_fraud.csv")) as fh:
            txs = list(csv.DictReader(fh))
        mule_total = sum(float(t["amount"]) for t in txs if t["src_id"] in mules and t["dst_id"] == collector)
        consolidation_amount = next(
            float(t["amount"]) for t in txs if t["src_id"] == collector and t["dst_id"] == agent
        )
        assert abs(consolidation_amount - mule_total) < 0.01

    def test_neptune_format(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=16)
        gen = OverdraftMuleGenerator(num_patterns=2, num_mules_range=(5, 5))
        n_tx, _ = gen.generate(
            max_account_id=100,
            start_tx_id=0,
            embedder=emb,
            output_dir=tmp_dir,
            fmt="neptune",
        )
        assert n_tx > 0
        path = os.path.join(tmp_dir, "fraud", "transactions_fraud.csv")
        with open(path) as fh:
            header = next(csv.reader(fh))
        assert "~from" in header

    def test_insufficient_accounts_returns_zero(self, tmp_dir):
        emb = EmbeddingGenerator("fake", dim=8)
        gen = OverdraftMuleGenerator(num_patterns=5)
        n_tx, next_id = gen.generate(
            max_account_id=6,
            start_tx_id=42,
            embedder=emb,
            output_dir=tmp_dir,
        )
        assert (n_tx, next_id) == (0, 42)
