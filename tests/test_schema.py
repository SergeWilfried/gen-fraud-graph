# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Tests for the mobile-money schema: role model, identity fields, tariffs,
and the extended account/transaction/sim-event outputs."""

from __future__ import annotations

import csv
import os
import random
from datetime import datetime

from gen_fraud_graph import schema
from gen_fraud_graph.generator import (
    FraudGraphGenerator,
    _generate_accounts_chunk,
    _generate_transactions_chunk,
)
from gen_fraud_graph.schema import (
    ACCOUNT_HEADERS,
    CHANNELS,
    KYC_TIERS,
    TRANSACTION_HEADERS,
    TS_FMT,
    TX_TYPES,
    account_type_for,
    commission_for,
    fee_for,
    msisdn_for,
    random_agent_uid,
    random_customer_uid,
    sample_agent_uids,
    sim_id_for,
)


def _read_rows(path: str) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


class TestRoleModel:
    def test_roles_are_deterministic(self):
        assert account_type_for(0) == "super_agent"
        assert account_type_for(2) == "agent"
        assert account_type_for(22) == "merchant"
        assert account_type_for(32) == "aggregator"
        assert account_type_for(35) == "customer"
        # Same slots in a later block.
        assert account_type_for(5_002) == "agent"
        assert account_type_for(5_500) == "customer"

    def test_role_mix_is_customer_heavy(self):
        counts: dict[str, int] = {}
        for uid in range(10_000):
            t = account_type_for(uid)
            counts[t] = counts.get(t, 0) + 1
        assert counts["customer"] / 10_000 > 0.9
        assert counts["agent"] / 10_000 == 0.02

    def test_random_agent_uid_returns_agents(self):
        rng = random.Random(7)
        for _ in range(50):
            assert account_type_for(random_agent_uid(5_000, rng)) == "agent"

    def test_random_customer_uid_returns_customers(self):
        rng = random.Random(7)
        for _ in range(50):
            assert account_type_for(random_customer_uid(5_000, rng)) == "customer"

    def test_sample_agent_uids_distinct(self):
        rng = random.Random(7)
        uids = sample_agent_uids(5_000, 6, rng)
        assert len(uids) == len(set(uids)) == 6

    def test_tiny_pool_falls_back_gracefully(self):
        rng = random.Random(7)
        assert 0 <= random_customer_uid(5, rng) < 5
        assert 0 <= random_agent_uid(1, rng) < 1


class TestIdentity:
    def test_msisdn_unique_and_senegalese(self):
        seen = {msisdn_for(uid) for uid in range(1_000)}
        assert len(seen) == 1_000
        assert all(m.startswith("+2217") for m in seen)

    def test_sim_id_unique(self):
        assert sim_id_for(1) != sim_id_for(2)


class TestTariffs:
    def test_fee_bands_never_negative_and_capped(self):
        for tx_type in TX_TYPES:
            for amount in (100, 5_000, 100_000, 19_000_000):
                assert 0 <= fee_for(tx_type, amount) <= 7_500

    def test_commission_only_on_cash_legs(self):
        assert commission_for("p2p", 100_000) == 0
        assert commission_for("cash_in", 100_000) > 0
        assert commission_for("cash_out", 100_000) > 0

    def test_splitting_deposits_farms_commission(self):
        """The banded schedule must make commission fraud economically
        real: many small cash-ins out-earn one large one."""
        total = 150_000
        split = sum(commission_for("cash_in", 15_000) for _ in range(10))
        single = commission_for("cash_in", total)
        assert split > single


class TestHeaders:
    def test_headers_derive_from_dataclasses(self):
        assert ACCOUNT_HEADERS[0] == "account_id"
        assert "msisdn" in ACCOUNT_HEADERS
        assert "kyc_tier" in ACCOUNT_HEADERS
        assert TRANSACTION_HEADERS[:3] == ["tx_id", "src_id", "dst_id"]
        for col in ("tx_type", "channel", "agent_id", "fee", "commission"):
            assert col in TRANSACTION_HEADERS

    def test_neptune_headers_typed(self):
        h = schema.neptune_account_headers()
        assert h[0] == "~id"
        assert "balance:Long" in h
        h = schema.neptune_transaction_headers()
        assert "amount:Long" in h
        assert "tx_type:String" in h


class TestAccountRows:
    def test_account_rows_carry_momo_fields(self, tmp_dir):
        _generate_accounts_chunk(0, 0, 0, 2_000, "fake", 8, tmp_dir, "csv")
        rows = _read_rows(os.path.join(tmp_dir, "accounts", "accounts_0_0.csv"))
        assert len(rows) == 2_000
        for row in rows:
            uid = int(row["account_id"].removeprefix("acc_"))
            assert row["account_type"] == account_type_for(uid)
            assert row["msisdn"] == msisdn_for(uid)
            assert row["kyc_tier"] in KYC_TIERS
            assert int(row["balance"]) >= 0

    def test_agents_hold_float_customers_do_not(self, tmp_dir):
        _generate_accounts_chunk(0, 0, 0, 2_000, "fake", 8, tmp_dir, "csv")
        rows = _read_rows(os.path.join(tmp_dir, "accounts", "accounts_0_0.csv"))
        for row in rows:
            if row["account_type"] in ("agent", "super_agent"):
                assert int(row["float_balance"]) > 0
            else:
                assert int(row["float_balance"]) == 0

    def test_customers_have_registration_agent(self, tmp_dir):
        _generate_accounts_chunk(0, 0, 0, 2_000, "fake", 8, tmp_dir, "csv")
        rows = _read_rows(os.path.join(tmp_dir, "accounts", "accounts_0_0.csv"))
        customers = [r for r in rows if r["account_type"] == "customer"]
        assert customers
        for row in customers:
            reg_uid = int(row["registration_agent_id"].removeprefix("acc_"))
            assert account_type_for(reg_uid) == "agent"

    def test_non_customers_are_fully_verified(self, tmp_dir):
        _generate_accounts_chunk(0, 0, 0, 2_000, "fake", 8, tmp_dir, "csv")
        rows = _read_rows(os.path.join(tmp_dir, "accounts", "accounts_0_0.csv"))
        for row in rows:
            if row["account_type"] != "customer":
                assert row["kyc_tier"] == "full"


class TestTransactionRows:
    def test_rows_carry_typed_fields(self, tmp_dir):
        _generate_transactions_chunk(0, 0, 0, 500, 5_000, "fake", 8, tmp_dir, "csv")
        rows = _read_rows(os.path.join(tmp_dir, "transactions", "transactions_0_0.csv"))
        assert len(rows) == 500
        for row in rows:
            assert row["tx_type"] in TX_TYPES
            assert row["channel"] in CHANNELS
            assert int(row["amount"]) > 0
            assert int(row["fee"]) >= 0
            assert int(row["commission"]) >= 0

    def test_cash_legs_have_agent_ids(self, tmp_dir):
        _generate_transactions_chunk(0, 0, 0, 500, 5_000, "fake", 8, tmp_dir, "csv")
        rows = _read_rows(os.path.join(tmp_dir, "transactions", "transactions_0_0.csv"))
        cash = [r for r in rows if r["tx_type"] in ("cash_in", "cash_out")]
        assert cash
        for row in cash:
            agent_uid = int(row["agent_id"].removeprefix("acc_"))
            assert account_type_for(agent_uid) == "agent"
            assert int(row["commission"]) > 0
        for row in rows:
            if row["tx_type"] not in ("cash_in", "cash_out"):
                assert row["agent_id"] == ""


class TestSimSwapEvents:
    def test_events_written_and_consistent(self, small_config):
        FraudGraphGenerator(small_config).run()
        fraud_dir = os.path.join(small_config.output_dir, "fraud")
        events = _read_rows(os.path.join(fraud_dir, "sim_events.csv"))
        cases = [
            c
            for c in _read_rows(os.path.join(fraud_dir, "fraud_cases.csv"))
            if c["pattern_type"] == "sim_swap_takeover"
        ]
        assert events
        assert len(events) == len(cases)
        for ev in events:
            uid = int(ev["account_id"].removeprefix("acc_"))
            assert ev["msisdn"] == msisdn_for(uid)
            assert ev["old_sim_id"] == sim_id_for(uid)
            assert ev["new_sim_id"] != ev["old_sim_id"]
            # The swap precedes the cash-out burst it enables.
            swap = datetime.strptime(ev["swap_ts"], TS_FMT)
            assert any(
                c["start_acc_id"] == ev["account_id"]
                and swap < datetime.strptime(c["window_start"], TS_FMT)
                for c in cases
            )

    def test_fraud_agent_roles_are_real(self, small_config):
        """SIM-swap cash-outs must fan out to agent-typed wallets — the
        role has to be visible in the graph, not only in ground truth."""
        FraudGraphGenerator(small_config).run()
        fraud_dir = os.path.join(small_config.output_dir, "fraud")
        for case in _read_rows(os.path.join(fraud_dir, "fraud_cases.csv")):
            if case["pattern_type"] != "sim_swap_takeover":
                continue
            for acc in case["involved_accounts"].split("|")[1:]:
                uid = int(acc.removeprefix("acc_"))
                assert account_type_for(uid) == "agent"
