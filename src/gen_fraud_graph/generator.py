# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Core generator — orchestrates account, transaction, and fraud generation."""

from __future__ import annotations

import csv
import os
import random
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta

import numpy as np
from tqdm import tqdm

from gen_fraud_graph.config import Config
from gen_fraud_graph.embeddings import EmbeddingGenerator
from gen_fraud_graph.exporters import get_headers
from gen_fraud_graph.schema import (
    ZONE_WEIGHTS,
    ZONES,
    account_type_for,
    channel_for,
    commission_for,
    description_for,
    fee_for,
    iso_ts,
    kyc_tier_for,
    msisdn_for,
    parse_date,
    random_agent_uid,
    random_aggregator_uid,
    random_customer_uid,
    random_merchant_uid,
    random_timestamp,
    sample_amount,
    sample_lognormal_xof,
    sample_tx_type,
    sim_id_for,
)
from gen_fraud_graph.typologies import (
    FraudRingGenerator,
    HawalaNetworkGenerator,
    MobileMoneyFraudGenerator,
    OverdraftMuleGenerator,
    SIMSwapFraudGenerator,
    StructuringGenerator,
    TradeBasedMLGenerator,
)

# Wallet balance shape (median XOF, log-normal sigma) by account type.
BALANCE_PARAMS: dict[str, tuple[int, float]] = {
    "customer": (15_000, 1.3),
    "merchant": (150_000, 1.2),
    "agent": (100_000, 1.0),
    "super_agent": (500_000, 1.0),
    "aggregator": (1_000_000, 1.0),
}

# Agent-side e-float (median XOF, sigma); zero for everyone else.
FLOAT_PARAMS: dict[str, tuple[int, float]] = {
    "agent": (750_000, 1.0),
    "super_agent": (5_000_000, 0.8),
}

# Fraction of wallets that share a device with the neighbouring account
# (family phones); mule farms exaggerate this pattern.
DEVICE_SHARE_RATE = 0.05


# ---------------------------------------------------------------------------
# Workload planning helpers
# ---------------------------------------------------------------------------


def _split_workload(total: int, num_shards: int) -> list[tuple[int, int]]:
    """Split ``total`` rows across ``num_shards`` shards without dropping rows."""
    if num_shards <= 0:
        raise ValueError("num_shards must be greater than zero")

    base, remainder = divmod(total, num_shards)
    shards: list[tuple[int, int]] = []
    start = 0

    for shard_idx in range(num_shards):
        count = base + (1 if shard_idx < remainder else 0)
        shards.append((start, count))
        start += count

    return shards


# ---------------------------------------------------------------------------
# Worker functions (must be top-level for multiprocessing)
# ---------------------------------------------------------------------------


def _generate_accounts_chunk(
    worker_id: int,
    batch_id: int,
    start_id: int,
    count: int,
    provider: str,
    dim: int,
    output_dir: str,
    fmt: str = "csv",
    sim_start_date: str = "2024-01-01",
    total_accounts: int | None = None,
) -> str:
    """Generate a chunk of account rows (called by ProcessPoolExecutor).

    ``total_accounts`` bounds the pool that registration agents are drawn
    from; it defaults to the end of this chunk so the worker stays
    self-contained when called directly.
    """
    rng = random.Random(start_id)
    embedder = EmbeddingGenerator(provider, dim=dim)  # type: ignore[arg-type]
    sim_start = parse_date(sim_start_date)
    pool_size = total_accounts if total_accounts is not None else max(start_id + count, 1)

    acc_dir = os.path.join(output_dir, "accounts")
    os.makedirs(acc_dir, exist_ok=True)

    headers = get_headers("account", fmt)  # type: ignore[arg-type]
    csv_path = os.path.join(acc_dir, f"accounts_{worker_id}_{batch_id}.csv")

    # Resume support
    existing_rows = 0
    file_exists = os.path.exists(csv_path)
    if file_exists:
        with open(csv_path) as fh:
            existing_rows = sum(1 for _ in fh) - 1
        if existing_rows >= count:
            return f"Worker {worker_id} Batch {batch_id}: Skipped (already complete)"
        print(f"  Worker {worker_id} Batch {batch_id}: Resuming from row {existing_rows}")

    batch_size = 5_000
    with open(csv_path, "a", newline="") as fh:
        writer = csv.writer(fh)
        if not file_exists:
            writer.writerow(headers)

        for i in range(max(0, existing_rows), count, batch_size):
            chunk_count = min(batch_size, count - i)
            batch_texts: list[str] = []
            batch_rows: list[list] = []

            for j in range(chunk_count):
                uid = start_id + i + j
                aid = f"acc_{uid}"
                name = f"Customer_{uid}"
                batch_texts.append(name)

                acc_type = account_type_for(uid)

                # Customers and merchants are onboarded by an agent;
                # agent-side wallets are provisioned by the operator.
                if acc_type in ("customer", "merchant") and pool_size > 2:
                    reg_agent_uid = random_agent_uid(pool_size, rng)
                    reg_agent = f"acc_{reg_agent_uid}" if reg_agent_uid != uid else ""
                else:
                    reg_agent = ""

                # A small share of wallets sit on a shared household device.
                if uid > 0 and rng.random() < DEVICE_SHARE_RATE:
                    device = f"dev_{uid - 1:08d}"
                else:
                    device = f"dev_{uid:08d}"

                bal_median, bal_sigma = BALANCE_PARAMS[acc_type]
                float_params = FLOAT_PARAMS.get(acc_type)
                float_balance = (
                    sample_lognormal_xof(rng, float_params[0], float_params[1], hi=100_000_000)
                    if float_params
                    else 0
                )

                # Accounts open over the three years leading up to the
                # simulated activity window.
                opened = sim_start - timedelta(days=rng.randint(1, 3 * 365))
                row: list = [
                    aid,
                    msisdn_for(uid),
                    name,
                    acc_type,
                    kyc_tier_for(acc_type, rng),
                    sim_id_for(uid),
                    device,
                    reg_agent,
                    rng.choices(ZONES, weights=ZONE_WEIGHTS, k=1)[0],
                    sample_lognormal_xof(rng, bal_median, bal_sigma, lo=0, hi=100_000_000),
                    float_balance,
                    round(rng.uniform(0, 1), 4),
                    opened.strftime("%Y-%m-%d"),
                ]
                if fmt == "neptune":
                    row.insert(1, "Account")
                batch_rows.append(row)

            if fmt == "neptune":
                embeddings = embedder.generate(batch_texts)
                final_rows = []
                for idx, r in enumerate(batch_rows):
                    vec = embeddings[idx]
                    if isinstance(vec, np.ndarray):
                        vec = vec.tolist()
                    final_rows.append(r + [";".join(map(str, vec))])
            else:
                final_rows = batch_rows

            writer.writerows(final_rows)
            if (i + chunk_count) % 50_000 == 0:
                print(f"  Worker {worker_id} Batch {batch_id}: {i + chunk_count} accounts written")

    return f"Worker {worker_id} Batch {batch_id}: Generated {count} accounts"


def _generate_transactions_chunk(
    worker_id: int,
    batch_id: int,
    start_tx_id: int,
    count: int,
    total_accounts: int,
    provider: str,
    dim: int,
    output_dir: str,
    fmt: str = "csv",
    sim_start_date: str = "2024-01-01",
    sim_days: int = 90,
) -> str:
    """Generate a chunk of transaction rows (called by ProcessPoolExecutor)."""
    rng = random.Random(start_tx_id)
    embedder = EmbeddingGenerator(provider, dim=dim)  # type: ignore[arg-type]
    sim_start = parse_date(sim_start_date)

    tx_dir = os.path.join(output_dir, "transactions")
    os.makedirs(tx_dir, exist_ok=True)

    headers = get_headers("transaction", fmt)  # type: ignore[arg-type]
    csv_path = os.path.join(tx_dir, f"transactions_{worker_id}_{batch_id}.csv")

    # Resume support
    existing_rows = 0
    file_exists = os.path.exists(csv_path)
    if file_exists:
        with open(csv_path) as fh:
            existing_rows = sum(1 for _ in fh) - 1
        if existing_rows >= count:
            return f"Worker {worker_id} Batch {batch_id}: Skipped (already complete)"
        print(f"  Worker {worker_id} Batch {batch_id}: Resuming from row {existing_rows}")

    embed_batch_size = 5_000
    with open(csv_path, "a", newline="") as fh:
        writer = csv.writer(fh)
        if not file_exists:
            writer.writerow(headers)

        for i in range(max(0, existing_rows), count, embed_batch_size):
            chunk_count = min(embed_batch_size, count - i)
            batch_texts: list[str] = []
            batch_rows: list[list] = []

            for j in range(chunk_count):
                tx_uid = start_tx_id + i + j
                tx_type = sample_tx_type(rng)

                # Endpoint roles follow the transaction type: cash legs run
                # through an agent, purchases hit merchants, airtime and
                # bills settle with aggregators.
                agent_id = ""
                if tx_type == "cash_in":
                    a = random_agent_uid(total_accounts, rng)
                    c = random_customer_uid(total_accounts, rng)
                    src_uid, dst_uid, agent_uid = a, c, a
                elif tx_type == "cash_out":
                    a = random_agent_uid(total_accounts, rng)
                    c = random_customer_uid(total_accounts, rng)
                    src_uid, dst_uid, agent_uid = c, a, a
                elif tx_type == "merchant_payment":
                    src_uid = random_customer_uid(total_accounts, rng)
                    dst_uid = random_merchant_uid(total_accounts, rng)
                    agent_uid = None
                elif tx_type in ("airtime", "bill_pay"):
                    src_uid = random_customer_uid(total_accounts, rng)
                    dst_uid = random_aggregator_uid(total_accounts, rng)
                    agent_uid = None
                elif tx_type == "bank_to_wallet":
                    src_uid = random_aggregator_uid(total_accounts, rng)
                    dst_uid = random_customer_uid(total_accounts, rng)
                    agent_uid = None
                else:  # p2p
                    src_uid = random_customer_uid(total_accounts, rng)
                    dst_uid = random_customer_uid(total_accounts, rng)
                    agent_uid = None
                while src_uid == dst_uid:
                    dst_uid = rng.randint(0, total_accounts - 1)
                if agent_uid is not None:
                    agent_id = f"acc_{agent_uid}"

                amount = sample_amount(rng, tx_type)
                desc = description_for(rng, tx_type)
                batch_texts.append(desc)

                row: list = [
                    f"tx_{tx_uid}",
                    f"acc_{src_uid}",
                    f"acc_{dst_uid}",
                    tx_type,
                    channel_for(rng, tx_type),
                    agent_id,
                    amount,
                    fee_for(tx_type, amount),
                    commission_for(tx_type, amount) if agent_id else 0,
                    iso_ts(random_timestamp(rng, sim_start, sim_days)),
                    desc,
                ]
                if fmt == "neptune":
                    row.insert(3, "TRANSFER")
                batch_rows.append(row)

            embeddings = embedder.generate(batch_texts)

            final_rows: list[list] = []
            for idx, r in enumerate(batch_rows):
                if fmt == "neptune":
                    final_rows.append(r)
                else:
                    vec = embeddings[idx]
                    if isinstance(vec, np.ndarray):
                        vec = vec.tolist()
                    final_rows.append(r + ["|".join(map(str, vec))])

            writer.writerows(final_rows)
            if (i + chunk_count) % 50_000 == 0:
                print(
                    f"  Worker {worker_id} Batch {batch_id}: {i + chunk_count} transactions written"
                )

    return f"Worker {worker_id} Batch {batch_id}: Generated {count} transactions"


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------


class FraudGraphGenerator:
    """Orchestrates the full synthetic fraud-graph generation pipeline.

    Usage::

        from gen_fraud_graph import FraudGraphGenerator, Config

        cfg = Config(scale_factor=0.01, embedding_provider="fake")
        gen = FraudGraphGenerator(cfg)
        gen.run()

    The output directory will contain:

    * ``accounts/``  — account node CSVs (one per worker × batch)
    * ``transactions/`` — legitimate transaction edge CSVs
    * ``fraud/`` — ``transactions_fraud.csv`` and ``fraud_cases.csv``
    """

    def __init__(self, config: Config) -> None:
        self.cfg = config

    def run(self, *, skip_accounts: bool = False) -> None:
        """Execute the three-phase generation pipeline.

        Args:
            skip_accounts: If *True*, skip Phase 1 (useful when resuming).
        """
        cfg = self.cfg
        os.makedirs(cfg.output_dir, exist_ok=True)

        print("=" * 50)
        print("gen_fraud_graph — Synthetic Fraud Graph Generator")
        print("=" * 50)
        print(f"  Scale factor : {cfg.scale_factor}")
        print(f"  Accounts     : {cfg.num_accounts:,}")
        print(f"  Transactions : {cfg.num_transactions:,}")
        print(f"  Fraud rings  : {cfg.num_fraud_rings:,}")
        print(f"  Format       : {cfg.output_format}")
        print(f"  Embedding    : {cfg.embedding_provider}")
        print(f"  Workers      : {cfg.workers}")
        print(f"  Compress     : {cfg.compress}")
        print(f"  Output       : {cfg.output_dir}")
        print("=" * 50)

        # Phase 1 — Accounts
        if not skip_accounts:
            self._generate_accounts()
        else:
            print("\n[Phase 1] Skipping accounts (--skip-accounts)")

        # Phase 2 — Transactions
        self._generate_transactions()

        # Phase 3 — Fraud rings
        self._generate_fraud()

        print("\nDone! All data generated.")

    # ------------------------------------------------------------------

    def _generate_accounts(self) -> None:
        cfg = self.cfg
        print("\n[Phase 1] Generating accounts...")

        shard_plan = _split_workload(cfg.num_accounts, cfg.workers * cfg.batches_per_worker)

        with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
            futures = []
            for w in range(cfg.workers):
                for b in range(cfg.batches_per_worker):
                    global_idx = w * cfg.batches_per_worker + b
                    start_id, count = shard_plan[global_idx]
                    futures.append(
                        pool.submit(
                            _generate_accounts_chunk,
                            w,
                            b,
                            start_id,
                            count,
                            cfg.embedding_provider,
                            cfg.embedding_dim,
                            cfg.output_dir,
                            cfg.output_format,
                            cfg.sim_start_date,
                            cfg.num_accounts,
                        )
                    )
            for f in tqdm(futures, total=len(futures), desc="Account batches"):
                f.result()

    def _generate_transactions(self) -> None:
        cfg = self.cfg
        print("\n[Phase 2] Generating transactions...")

        shard_plan = _split_workload(cfg.num_transactions, cfg.workers * cfg.batches_per_worker)

        with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
            futures = []
            for w in range(cfg.workers):
                for b in range(cfg.batches_per_worker):
                    global_idx = w * cfg.batches_per_worker + b
                    start_id, count = shard_plan[global_idx]
                    futures.append(
                        pool.submit(
                            _generate_transactions_chunk,
                            w,
                            b,
                            start_id,
                            count,
                            cfg.num_accounts,
                            cfg.embedding_provider,
                            cfg.embedding_dim,
                            cfg.output_dir,
                            cfg.output_format,
                            cfg.sim_start_date,
                            cfg.sim_days,
                        )
                    )
            for f in tqdm(futures, total=len(futures), desc="Transaction batches"):
                f.result()

    def _generate_fraud(self) -> None:
        cfg = self.cfg
        print("\n[Phase 3] Generating fraud patterns...")

        embedder = EmbeddingGenerator(cfg.embedding_provider, dim=cfg.embedding_dim)

        # --- cyclic money-laundering rings ---
        assert cfg.num_fraud_rings is not None
        ring_gen = FraudRingGenerator(
            num_rings=cfg.num_fraud_rings,
            depth_range=cfg.fraud_ring_depth_range,
            sim_start_date=cfg.sim_start_date,
            sim_days=cfg.sim_days,
        )
        n_ring_tx, next_tx_id = ring_gen.generate(
            max_account_id=cfg.num_accounts,
            start_tx_id=cfg.num_transactions,
            embedder=embedder,
            output_dir=cfg.output_dir,
            fmt=cfg.output_format,
            compress=cfg.compress,
        )
        print(f"  Injected {n_ring_tx:,} ring transactions across {cfg.num_fraud_rings:,} rings")

        # --- structuring / smurfing patterns ---
        assert cfg.num_structuring_patterns is not None
        struct_gen = StructuringGenerator(
            num_patterns=cfg.num_structuring_patterns,
            smurfs_range=cfg.structuring_smurfs_range,
            amount_range=cfg.structuring_amount_range,
            sim_start_date=cfg.sim_start_date,
            sim_days=cfg.sim_days,
        )
        n_struct_tx, next_tx_id_2 = struct_gen.generate(
            max_account_id=cfg.num_accounts,
            start_tx_id=next_tx_id,
            embedder=embedder,
            output_dir=cfg.output_dir,
            fmt=cfg.output_format,
            compress=cfg.compress,
        )
        print(
            f"  Injected {n_struct_tx:,} structuring transactions "
            f"across {cfg.num_structuring_patterns:,} patterns"
        )

        # --- mobile money fraud patterns ---
        assert cfg.num_mobile_money_patterns is not None
        mm_gen = MobileMoneyFraudGenerator(
            num_patterns=cfg.num_mobile_money_patterns,
            amount_range=cfg.mobile_money_amount_range,
            sim_start_date=cfg.sim_start_date,
            sim_days=cfg.sim_days,
        )
        n_mm_tx, next_tx_id_3 = mm_gen.generate(
            max_account_id=cfg.num_accounts,
            start_tx_id=next_tx_id_2,
            embedder=embedder,
            output_dir=cfg.output_dir,
            fmt=cfg.output_format,
            compress=cfg.compress,
        )
        print(
            f"  Injected {n_mm_tx:,} mobile money transactions "
            f"across {cfg.num_mobile_money_patterns:,} patterns"
        )

        # --- trade-based money laundering (TBML) patterns ---
        assert cfg.num_trade_based_ml_patterns is not None
        tbml_gen = TradeBasedMLGenerator(
            num_patterns=cfg.num_trade_based_ml_patterns,
            intermediaries_range=cfg.trade_based_ml_intermediaries_range,
            amount_range=cfg.trade_based_ml_amount_range,
            sim_start_date=cfg.sim_start_date,
            sim_days=cfg.sim_days,
        )
        n_tbml_tx, next_tx_id_4 = tbml_gen.generate(
            max_account_id=cfg.num_accounts,
            start_tx_id=next_tx_id_3,
            embedder=embedder,
            output_dir=cfg.output_dir,
            fmt=cfg.output_format,
            compress=cfg.compress,
        )
        print(
            f"  Injected {n_tbml_tx:,} TBML transactions "
            f"across {cfg.num_trade_based_ml_patterns:,} patterns"
        )

        # --- hawala / informal value transfer network patterns ---
        assert cfg.num_hawala_patterns is not None
        hawala_gen = HawalaNetworkGenerator(
            num_patterns=cfg.num_hawala_patterns,
            settlement_amount_range=cfg.hawala_settlement_amount_range,
            transfer_amount_range=cfg.hawala_transfer_amount_range,
            sim_start_date=cfg.sim_start_date,
            sim_days=cfg.sim_days,
        )
        n_hawala_tx, next_tx_id_5 = hawala_gen.generate(
            max_account_id=cfg.num_accounts,
            start_tx_id=next_tx_id_4,
            embedder=embedder,
            output_dir=cfg.output_dir,
            fmt=cfg.output_format,
            compress=cfg.compress,
        )
        print(
            f"  Injected {n_hawala_tx:,} hawala transactions "
            f"across {cfg.num_hawala_patterns:,} patterns"
        )

        # --- SIM-swap account takeover patterns ---
        assert cfg.num_sim_swap_patterns is not None
        simswap_gen = SIMSwapFraudGenerator(
            num_patterns=cfg.num_sim_swap_patterns,
            num_agents_range=cfg.sim_swap_agents_range,
            amount_range=cfg.sim_swap_amount_range,
            sim_start_date=cfg.sim_start_date,
            sim_days=cfg.sim_days,
        )
        n_simswap_tx, next_tx_id_6 = simswap_gen.generate(
            max_account_id=cfg.num_accounts,
            start_tx_id=next_tx_id_5,
            embedder=embedder,
            output_dir=cfg.output_dir,
            fmt=cfg.output_format,
            compress=cfg.compress,
        )
        print(
            f"  Injected {n_simswap_tx:,} SIM-swap transactions "
            f"across {cfg.num_sim_swap_patterns:,} patterns"
        )

        # --- overdraft / micro-loan mule chain patterns ---
        assert cfg.num_overdraft_mule_patterns is not None
        mule_gen = OverdraftMuleGenerator(
            num_patterns=cfg.num_overdraft_mule_patterns,
            num_mules_range=cfg.overdraft_mule_num_mules_range,
            loan_amount_range=cfg.overdraft_mule_loan_amount_range,
            sim_start_date=cfg.sim_start_date,
            sim_days=cfg.sim_days,
        )
        n_mule_tx, _ = mule_gen.generate(
            max_account_id=cfg.num_accounts,
            start_tx_id=next_tx_id_6,
            embedder=embedder,
            output_dir=cfg.output_dir,
            fmt=cfg.output_format,
            compress=cfg.compress,
        )
        print(
            f"  Injected {n_mule_tx:,} overdraft mule transactions "
            f"across {cfg.num_overdraft_mule_patterns:,} patterns"
        )
