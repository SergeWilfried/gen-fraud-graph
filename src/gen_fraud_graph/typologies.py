# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Fraud typology definitions for synthetic graph injection."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from gen_fraud_graph.embeddings import EmbeddingGenerator
from gen_fraud_graph.exporters import append_csv, get_headers, write_output
from gen_fraud_graph.schema import (
    FRAUD_CASE_HEADERS,
    SIM_EVENT_HEADERS,
    channel_for,
    commission_for,
    description_for,
    fee_for,
    iso_ts,
    msisdn_for,
    parse_date,
    random_agent_uid,
    random_customer_uid,
    random_timestamp,
    round_xof,
    sample_agent_uids,
    sample_lognormal_xof,
    sim_id_for,
)

# Shared unseeded RNG for the typology generators (the module-level
# ``random`` functions are used elsewhere; schema helpers want an
# explicit ``random.Random``).
_RNG = random.Random()


def _tx_row(
    tx_id: int,
    src: str,
    dst: str,
    tx_type: str,
    amount: int,
    ts_str: str,
    fmt: str,
    agent_id: str = "",
    rng: random.Random | None = None,
) -> tuple[list, str]:
    """Build one edge row (and its embedding text) in the shared schema.

    Injected rows reuse the per-type description vocabulary, channel mix,
    and fee/commission schedule of legitimate traffic — a fraud-only
    wording pool or tariff would leak the label through a single column.
    """
    rng = rng if rng is not None else _RNG
    desc = description_for(rng, tx_type)
    row: list = [f"tx_{tx_id}", src, dst]
    if fmt == "neptune":
        row.append("TRANSFER")
    row.extend(
        [
            tx_type,
            channel_for(rng, tx_type),
            agent_id,
            amount,
            fee_for(tx_type, amount),
            commission_for(tx_type, amount) if agent_id else 0,
            ts_str,
            desc,
        ]
    )
    return row, desc


# ---------------------------------------------------------------------------
# Fraud ring generator (cyclic money-laundering patterns)
# ---------------------------------------------------------------------------


@dataclass
class FraudRingGenerator:
    """Generate cyclic fraud-ring patterns.

    Each ring is a cycle of ``depth`` accounts connected by suspicious
    high-value transactions.

    Per-hop amounts are correlated, not constant: each ring draws one
    log-normal seed amount (overlapping the legitimate heavy tail) and
    passes it around the cycle, each hop skimming a small cut. Hops are
    spread hours-to-days apart across a multi-day window — laundering
    cycles are slow compared to cash-out bursts.

    Args:
        num_rings: How many rings to create.
        depth_range: ``(min_depth, max_depth)`` hops per ring.
        amount_median: Median of the log-normal seed amount, in FCFA.
        amount_sigma: Log-normal sigma of the seed amount.
        skim_range: Per-hop fractional skim applied as the sum moves.
        sim_start_date: First day of the simulated window (``YYYY-MM-DD``).
        sim_days: Length of the simulated window in days.
    """

    num_rings: int = 100
    depth_range: tuple[int, int] = (4, 7)
    amount_median: int = 600_000
    amount_sigma: float = 1.0
    skim_range: tuple[float, float] = (0.01, 0.05)
    sim_start_date: str = "2024-01-01"
    sim_days: int = 90

    def generate(
        self,
        max_account_id: int,
        start_tx_id: int,
        embedder: EmbeddingGenerator,
        output_dir: str,
        fmt: str = "csv",
        compress: bool = False,
        rng: random.Random | None = None,
    ) -> tuple[int, int]:
        """Generate fraud rings and write output files.

        Returns:
            ``(num_fraud_transactions, next_tx_id)``
        """
        import os

        from tqdm import tqdm

        rng = rng if rng is not None else _RNG
        fraud_dir = os.path.join(output_dir, "fraud")
        os.makedirs(fraud_dir, exist_ok=True)

        headers_tx = get_headers("transaction", fmt)  # type: ignore[arg-type]
        headers_cases = FRAUD_CASE_HEADERS

        tx_rows: list[list] = []
        case_rows: list[list] = []
        current_tx_id = start_tx_id
        sim_start = parse_date(self.sim_start_date)
        # Allocate every ring's accounts up front from one pool of distinct
        # ids, then give each ring its own slice. Overlapping ranges would
        # merge two rings into a single non-cycle component and make the
        # per-ring involved_accounts labels ambiguous.
        min_d, max_d = self.depth_range
        depths = [rng.randint(min_d, max_d) for _ in range(self.num_rings)]
        total_needed = sum(depths)
        if total_needed > max_account_id:
            raise ValueError(
                f"{self.num_rings} fraud rings need {total_needed} distinct "
                f"accounts but only {max_account_id} exist; lower the ring "
                f"count or raise the account scale"
            )
        account_pool = rng.sample(range(max_account_id), total_needed)
        pool_offset = 0

        for pattern_id in tqdm(range(self.num_rings), desc="Generating fraud rings"):
            depth = depths[pattern_id]
            ring_ids = account_pool[pool_offset : pool_offset + depth]
            pool_offset += depth

            accounts = [f"acc_{i}" for i in ring_ids]
            involved = "|".join(accounts)

            batch_texts: list[str] = []
            batch_rows: list[list] = []

            window_days = rng.randint(2, 14)
            start_day = rng.randint(0, max(0, self.sim_days - window_days - 1))
            ts = random_timestamp(rng, sim_start + timedelta(days=start_day), 1)
            window_start = ts
            hop_gap_cap = max(3_600, window_days * 86_400 // depth)
            amount = sample_lognormal_xof(rng, self.amount_median, self.amount_sigma, lo=50_000)

            for k in range(depth):
                src = accounts[k]
                dst = accounts[(k + 1) % depth]
                row, desc = _tx_row(
                    current_tx_id, src, dst, "p2p", amount, iso_ts(ts), fmt, rng=rng
                )
                batch_texts.append(desc)
                batch_rows.append(row)
                current_tx_id += 1

                skim = rng.uniform(*self.skim_range)
                amount = max(5_000, round_xof(amount * (1 - skim)))
                if k < depth - 1:
                    ts += timedelta(seconds=rng.randint(3_600, hop_gap_cap))
            window_end = ts

            embeddings = embedder.generate(batch_texts)

            for idx, r in enumerate(batch_rows):
                if fmt == "neptune":
                    tx_rows.append(r)
                else:
                    vec = embeddings[idx]
                    if isinstance(vec, np.ndarray):
                        vec = vec.tolist()
                    tx_rows.append(r + ["|".join(map(str, vec))])

            case_rows.append(
                [
                    f"pat_{pattern_id}",
                    accounts[0],
                    "cycle",
                    depth,
                    involved,
                    iso_ts(window_start),
                    iso_ts(window_end),
                ]
            )

        file_tx = os.path.join(fraud_dir, "transactions_fraud")
        file_cases = os.path.join(fraud_dir, "fraud_cases")
        write_output(file_tx, headers_tx, tx_rows, compress=compress)
        write_output(file_cases, headers_cases, case_rows, compress=compress)

        return len(tx_rows), current_tx_id


@dataclass
class StructuringGenerator:
    """Generate structuring (smurfing) fraud patterns.
    In a structuring scheme a single coordinator account receives funds from
    several "smurf" accounts, each sending amounts just below the BCEAO
    5,000,000 FCFA cash payment limit. The coordinator aggregates these
    deposits to move a larger sum without triggering a single reportable
    event.

    Graph shape::

        smurf_0 -> coordinator
        smurf_1 -> coordinator
            ...
        smurf_N -> coordinator
    Multiple sources converge on one node.
    This is a structurally distinct from the cyclic ring produced by
    :class:'FraudRingGenerator' and exercises different subgraph-detection
    algorithms.

    Args:
        num_patterns: How many structuring patterns to create.
        smurfs_range: ''(min_smurfs, mac_smurfs)'' - number of feeder
        accounts per pattern. Mirrors the real world practice of using
        3-10 smurfs to stay inconspicuous.
        amount_range: ''(min_amount, max_amount)'' - each smurf transfer is
        drawn uniformly form this range. Defaults to 4,000,000-4,900,000 FCFA,
        deliberately sub-threshold.
    """

    num_patterns: int = 100
    smurfs_range: tuple[int, int] = (3, 10)
    amount_range: tuple[int, int] = (4_000_000, 4_950_000)
    sim_start_date: str = "2024-01-01"
    sim_days: int = 90

    def generate(
        self,
        max_account_id: int,
        start_tx_id: int,
        embedder: EmbeddingGenerator,
        output_dir: str,
        fmt: str = "csv",
        compress: bool = False,
        rng: random.Random | None = None,
    ) -> tuple[int, int]:
        """Generate structuring patterns and append to fraud output files.

        Output files are appended to the same ``fraud/`` directory used by
        :class:`FraudRingGenerator` so a single pipeline run can inject both
        typologies into one dataset.

        Args:
            max_account_id: Upper bound of account IDs already generated.
            start_tx_id: First transaction ID to use (must not collide with
                IDs already written by the ring generator or normal txs).
            embedder: Embedding generator instance — same one used by the
                ring generator so embedding provenance is consistent.
            output_dir: Root output directory.
            fmt: ``"csv"`` or ``"neptune"``.
            compress: ZIP the output CSV files.

        Returns:
            ``(num_fraud_transactions, next_tx_id)``
        """
        import os

        from tqdm import tqdm

        rng = rng if rng is not None else _RNG
        fraud_dir = os.path.join(output_dir, "fraud")
        os.makedirs(fraud_dir, exist_ok=True)

        headers_tx = get_headers("transaction", fmt)  # type: ignore[arg-type]
        headers_cases = FRAUD_CASE_HEADERS

        tx_rows: list[list] = []
        case_rows: list[list] = []
        current_tx_id = start_tx_id
        sim_start = parse_date(self.sim_start_date)

        for pattern_id in tqdm(range(self.num_patterns), desc="Generating structuring patterns"):
            min_s, max_s = self.smurfs_range
            num_smurfs = rng.randint(min_s, max_s)

            # Draw the coordinator and smurfs anywhere in the account pool.
            # Consecutive IDs would let a detector cheat on ID adjacency
            # instead of the fan-in shape.
            needed = num_smurfs + 1
            if max_account_id < needed:
                num_smurfs = max(1, max_account_id - 1)
                needed = num_smurfs + 1

            idxs = rng.sample(range(max_account_id), needed)
            coordinator = f"acc_{idxs[0]}"
            smurfs = [f"acc_{idxs[1 + i]}" for i in range(num_smurfs)]
            involved = "|".join([coordinator] + smurfs)

            batch_texts: list[str] = []
            batch_rows: list[list] = []

            # Deposits land over one to three days.
            window_days = rng.randint(1, 3)
            start_day = rng.randint(0, max(0, self.sim_days - window_days - 1))
            base = sim_start + timedelta(days=start_day)
            ts_list = sorted(random_timestamp(rng, base, window_days) for _ in smurfs)

            for smurf, ts in zip(smurfs, ts_list, strict=True):
                amount = round_xof(rng.uniform(*self.amount_range))
                row, desc = _tx_row(
                    current_tx_id, smurf, coordinator, "p2p", amount, iso_ts(ts), fmt, rng=rng
                )
                batch_texts.append(desc)
                batch_rows.append(row)
                current_tx_id += 1

            embeddings = embedder.generate(batch_texts)

            for idx, r in enumerate(batch_rows):
                if fmt == "neptune":
                    tx_rows.append(r)
                else:
                    vec = embeddings[idx]
                    if isinstance(vec, np.ndarray):
                        vec = vec.tolist()
                    tx_rows.append(r + ["|".join(map(str, vec))])

            case_rows.append(
                [
                    f"struct_{pattern_id}",
                    coordinator,
                    "structuring",
                    num_smurfs,  # depth = number of feeder hops
                    involved,
                    iso_ts(ts_list[0]),
                    iso_ts(ts_list[-1]),
                ]
            )

        # Append to the same fraud files so both typologies land in one CSV.
        file_tx = os.path.join(fraud_dir, "transactions_fraud")
        file_cases = os.path.join(fraud_dir, "fraud_cases")
        append_csv(file_tx + ".csv", headers_tx, tx_rows)
        append_csv(file_cases + ".csv", headers_cases, case_rows)

        return len(tx_rows), current_tx_id


@dataclass
class MobileMoneyFraudGenerator:
    """Generate mobile money specific fraud patterns (Agent commission fraud).

    This models a split transaction scheme where an agent divides a single
    large deposit into a burst of small transfers to artificially inflate
    commissions.

    Graph shape::

        agent -> customer (tx 1)
        agent -> customer (tx 2)
        ...
        agent -> customer (tx N)
    """

    num_patterns: int = 100
    amount_range: tuple[int, int] = (5_000, 25_000)
    burst_window_minutes: int = 10  # max minutes between split transactions in a burst
    sim_start_date: str = "2024-01-01"
    sim_days: int = 90

    def generate(
        self,
        max_account_id: int,
        start_tx_id: int,
        embedder: EmbeddingGenerator,
        output_dir: str,
        fmt: str = "csv",
        compress: bool = False,
        rng: random.Random | None = None,
    ) -> tuple[int, int]:
        """Generate mobile money patterns and append to fraud output files."""
        import os

        from tqdm import tqdm

        # Pre-loop guard: need at least 2 distinct accounts to form an agent->customer edge.
        if max_account_id < 2:
            return 0, start_tx_id

        sim_start = parse_date(self.sim_start_date)

        rng = rng if rng is not None else _RNG
        fraud_dir = os.path.join(output_dir, "fraud")
        os.makedirs(fraud_dir, exist_ok=True)

        headers_tx = get_headers("transaction", fmt)  # type: ignore[arg-type]
        headers_cases = FRAUD_CASE_HEADERS

        tx_rows: list[list] = []
        case_rows: list[list] = []
        current_tx_id = start_tx_id

        for pattern_id in tqdm(range(self.num_patterns), desc="Generating mobile money fraud"):
            # The splitting party is a real agent-typed wallet: commission
            # farming is only expressible when the graph carries the role
            # and the per-transaction commission.
            agent_idx = random_agent_uid(max_account_id, rng)
            customer_idx = random_customer_uid(max_account_id, rng)
            while customer_idx == agent_idx:
                customer_idx = rng.randrange(max_account_id)

            agent = f"acc_{agent_idx}"
            customer = f"acc_{customer_idx}"
            involved = f"{agent}|{customer}"

            num_txs = rng.randint(4, 10)

            # Burst base time: random point in the simulated window, then
            # each split tx is offset by a small random increment to
            # simulate a rapid burst — essential for velocity detection.
            base_ts = random_timestamp(rng, sim_start, self.sim_days)

            batch_texts: list[str] = []
            batch_rows: list[list] = []
            elapsed_seconds = 0

            for _ in range(num_txs):
                amount = round_xof(rng.uniform(*self.amount_range))

                tx_ts = iso_ts(base_ts + timedelta(seconds=elapsed_seconds))
                last_offset = elapsed_seconds
                max_gap = max(30, self.burst_window_minutes * 60 // num_txs)
                elapsed_seconds += rng.randint(30, max_gap)

                row, desc = _tx_row(
                    current_tx_id,
                    agent,
                    customer,
                    "cash_in",
                    amount,
                    tx_ts,
                    fmt,
                    agent_id=agent,
                    rng=rng,
                )
                batch_texts.append(desc)
                batch_rows.append(row)
                current_tx_id += 1

            embeddings = embedder.generate(batch_texts)

            for idx, r in enumerate(batch_rows):
                if fmt == "neptune":
                    tx_rows.append(r)
                else:
                    vec = embeddings[idx]
                    if isinstance(vec, np.ndarray):
                        vec = vec.tolist()
                    tx_rows.append(r + ["|".join(map(str, vec))])

            case_rows.append(
                [
                    f"mm_{pattern_id}",
                    agent,
                    "mobile_money_split",
                    num_txs,
                    involved,
                    iso_ts(base_ts),
                    iso_ts(base_ts + timedelta(seconds=last_offset)),
                ]
            )

        file_tx = os.path.join(fraud_dir, "transactions_fraud")
        file_cases = os.path.join(fraud_dir, "fraud_cases")
        append_csv(file_tx + ".csv", headers_tx, tx_rows)
        append_csv(file_cases + ".csv", headers_cases, case_rows)

        return len(tx_rows), current_tx_id


@dataclass
class TradeBasedMLGenerator:
    """Generate trade-based money laundering (TBML) fraud patterns.

    GIABA (the regional FATF-style body for UEMOA) identifies TBML as a top
    typology given the region's large informal trade sector. Funds are
    laundered by manipulating invoices for imported goods, then layered
    through a chain of intermediary accounts before reaching a beneficiary.

    Graph shape::

        exporter -> shell_importer
        shell_importer -> intermediary_0
        shell_importer -> intermediary_1
            ...
        shell_importer -> intermediary_{k-1}
        intermediary_0 -> beneficiary
        intermediary_1 -> beneficiary
            ...
        intermediary_{k-1} -> beneficiary

    A single shell_importer fans out to k intermediaries (layering), which
    then fan back in to one beneficiary.

    involved_accounts positional convention (pipe-joined):
        ``exporter|shell_importer|intermediary_0|...|intermediary_{k-1}|beneficiary``
        i.e. ``accounts[0]`` = exporter, ``accounts[1]`` = shell_importer,
        ``accounts[2:2+k]`` = intermediaries, ``accounts[-1]`` = beneficiary.
        ``depth`` = k, the number of intermediaries.

    Args:
        num_patterns: How many TBML patterns to create.
        intermediaries_range: ``(min_intermediaries, max_intermediaries)`` -
            number of layering intermediary accounts per pattern. Defaults
            to ``(3, 5)``.
        amount_range: ``(min_amount, max_amount)`` - per-edge transaction
            amount range in FCFA, drawn uniformly per edge. Defaults to
            ``(20_000_000, 150_000_000)``, reflecting invoice-manipulation
            scale (large commercial trade payments).
    """

    num_patterns: int = 100
    intermediaries_range: tuple[int, int] = (3, 5)
    amount_range: tuple[int, int] = (2_000_000, 15_000_000)
    sim_start_date: str = "2024-01-01"
    sim_days: int = 90

    def generate(
        self,
        max_account_id: int,
        start_tx_id: int,
        embedder: EmbeddingGenerator,
        output_dir: str,
        fmt: str = "csv",
        compress: bool = False,
        rng: random.Random | None = None,
    ) -> tuple[int, int]:
        """Generate TBML patterns and append to fraud output files.

        Each pattern needs 3 + k distinct accounts (exporter, shell_importer,
        k intermediaries, beneficiary), where k is drawn from
        ``intermediaries_range``. If fewer than 5 accounts exist (the
        minimum for k=3), generation is skipped.

        Returns:
            ``(num_fraud_transactions, next_tx_id)``
        """
        import os

        from tqdm import tqdm

        if max_account_id < 5:
            return 0, start_tx_id

        rng = rng if rng is not None else _RNG
        fraud_dir = os.path.join(output_dir, "fraud")
        os.makedirs(fraud_dir, exist_ok=True)

        headers_tx = get_headers("transaction", fmt)  # type: ignore[arg-type]
        headers_cases = FRAUD_CASE_HEADERS

        tx_rows: list[list] = []
        case_rows: list[list] = []
        current_tx_id = start_tx_id
        sim_start = parse_date(self.sim_start_date)

        for pattern_id in tqdm(range(self.num_patterns), desc="Generating TBML patterns"):
            min_k, max_k = self.intermediaries_range
            k = rng.randint(min_k, max_k)
            needed = k + 3
            if max_account_id < needed:
                k = max(min_k, max_account_id - 3)
                needed = k + 3
                if max_account_id < needed:
                    continue

            idxs = rng.sample(range(max_account_id), needed)
            exporter = f"acc_{idxs[0]}"
            shell_importer = f"acc_{idxs[1]}"
            intermediaries = [f"acc_{idxs[2 + i]}" for i in range(k)]
            beneficiary = f"acc_{idxs[-1]}"
            involved = "|".join([exporter, shell_importer] + intermediaries + [beneficiary])

            batch_texts: list[str] = []
            batch_rows: list[list] = []

            edges = [(exporter, shell_importer)]
            edges.extend((shell_importer, inter) for inter in intermediaries)
            edges.extend((inter, beneficiary) for inter in intermediaries)

            # Invoice settlement then layering: hops land hours-to-days
            # apart over a multi-week window.
            window_days = rng.randint(3, 21)
            start_day = rng.randint(0, max(0, self.sim_days - window_days - 1))
            ts = random_timestamp(rng, sim_start + timedelta(days=start_day), 1)
            window_start = ts
            hop_gap_cap = max(21_600, window_days * 86_400 // len(edges))

            # The invoice settlement arrives as a bank transfer; the
            # layering hops move on as wallet-to-wallet transfers.
            for edge_idx, (src, dst) in enumerate(edges):
                amount = round_xof(rng.uniform(*self.amount_range))
                tx_type = "bank_to_wallet" if edge_idx == 0 else "p2p"
                row, desc = _tx_row(
                    current_tx_id, src, dst, tx_type, amount, iso_ts(ts), fmt, rng=rng
                )
                batch_texts.append(desc)
                batch_rows.append(row)
                current_tx_id += 1
                ts += timedelta(seconds=rng.randint(21_600, hop_gap_cap))
            window_end = ts

            embeddings = embedder.generate(batch_texts)

            for idx, r in enumerate(batch_rows):
                if fmt == "neptune":
                    tx_rows.append(r)
                else:
                    vec = embeddings[idx]
                    if isinstance(vec, np.ndarray):
                        vec = vec.tolist()
                    tx_rows.append(r + ["|".join(map(str, vec))])

            case_rows.append(
                [
                    f"tbml_{pattern_id}",
                    exporter,
                    "trade_based_ml",
                    k,
                    involved,
                    iso_ts(window_start),
                    iso_ts(window_end),
                ]
            )

        file_tx = os.path.join(fraud_dir, "transactions_fraud")
        file_cases = os.path.join(fraud_dir, "fraud_cases")
        append_csv(file_tx + ".csv", headers_tx, tx_rows)
        append_csv(file_cases + ".csv", headers_cases, case_rows)

        return len(tx_rows), current_tx_id


@dataclass
class HawalaNetworkGenerator:
    """Generate hawala / informal value transfer (IVT) fraud patterns.

    Models an informal remittance corridor operated by a pair of hawaladars
    (informal value transfer brokers). A sender deposits cash with
    hawaladar_A; hawaladar_A instructs hawaladar_B (in another location) to
    pay out to the beneficiary; the two hawaladars periodically net their
    mutual debt with a reverse bulk-settlement wire rather than moving funds
    for every individual transfer, the hallmark of hawala's off-the-books
    settlement mechanism.

    Graph shape::

        sender -> hawaladar_A         (cash deposit)
        hawaladar_A -> hawaladar_B    (settlement wire)
        hawaladar_B -> beneficiary    (payout)
        hawaladar_B -> hawaladar_A    (periodic reverse bulk-settlement,
                                        direction may flip to A -> B;
                                        emitted probabilistically, not
                                        on every pattern instance)

    involved_accounts positional convention (pipe-joined, fixed length 4):
        ``sender|hawaladar_A|hawaladar_B|beneficiary``
        i.e. ``accounts[0]`` = sender, ``accounts[1]`` = hawaladar_A,
        ``accounts[2]`` = hawaladar_B, ``accounts[3]`` = beneficiary.
        ``depth`` = number of edges actually emitted for this pattern
        instance: 3 if no reverse settlement fired, 4 if it did.

    Args:
        num_patterns: How many hawala corridors to create.
        transfer_amount_range: Sender->hawaladar_A and hawaladar_B->
            beneficiary leg amount range in FCFA. Defaults to
            ``(100_000, 2_000_000)`` - retail remittance scale, deliberately
            smaller than the bulk settlement legs.
        settlement_amount_range: hawaladar_A<->hawaladar_B settlement/netting
            amount range in FCFA. Defaults to ``(5_000_000, 50_000_000)``.
        settlement_probability: Probability that the periodic reverse
            bulk-settlement edge fires for a given pattern instance.
            Defaults to ``0.3``.
    """

    num_patterns: int = 100
    transfer_amount_range: tuple[int, int] = (50_000, 1_500_000)
    settlement_amount_range: tuple[int, int] = (2_000_000, 12_000_000)
    settlement_probability: float = 0.3
    sim_start_date: str = "2024-01-01"
    sim_days: int = 90

    def generate(
        self,
        max_account_id: int,
        start_tx_id: int,
        embedder: EmbeddingGenerator,
        output_dir: str,
        fmt: str = "csv",
        compress: bool = False,
        rng: random.Random | None = None,
    ) -> tuple[int, int]:
        """Generate hawala network patterns and append to fraud output files.

        Each pattern needs exactly 4 distinct accounts (sender, hawaladar_A,
        hawaladar_B, beneficiary). If fewer than 4 accounts exist,
        generation is skipped and ``(0, start_tx_id)`` is returned.

        Returns:
            ``(num_fraud_transactions, next_tx_id)``
        """
        import os

        from tqdm import tqdm

        if max_account_id < 4:
            return 0, start_tx_id

        rng = rng if rng is not None else _RNG
        fraud_dir = os.path.join(output_dir, "fraud")
        os.makedirs(fraud_dir, exist_ok=True)

        headers_tx = get_headers("transaction", fmt)  # type: ignore[arg-type]
        headers_cases = FRAUD_CASE_HEADERS

        tx_rows: list[list] = []
        case_rows: list[list] = []
        current_tx_id = start_tx_id
        sim_start = parse_date(self.sim_start_date)

        for pattern_id in tqdm(range(self.num_patterns), desc="Generating hawala patterns"):
            idxs = rng.sample(range(max_account_id), 4)
            sender = f"acc_{idxs[0]}"
            hawaladar_a = f"acc_{idxs[1]}"
            hawaladar_b = f"acc_{idxs[2]}"
            beneficiary = f"acc_{idxs[3]}"
            involved = "|".join([sender, hawaladar_a, hawaladar_b, beneficiary])

            batch_texts: list[str] = []
            batch_rows: list[list] = []

            # Deposit at t0, payout minutes-to-hours later, bulk
            # settlement wires days later — hawala moves value fast and
            # nets debt slowly.
            start_day = rng.randint(0, max(0, self.sim_days - 9))
            t_deposit = random_timestamp(rng, sim_start + timedelta(days=start_day), 1)
            t_payout = t_deposit + timedelta(minutes=rng.randint(10, 300))
            t_settle = t_deposit + timedelta(days=rng.randint(1, 7), minutes=rng.randint(0, 720))

            edges: list[tuple[str, str, tuple[int, int], datetime]] = [
                (sender, hawaladar_a, self.transfer_amount_range, t_deposit),
                (hawaladar_a, hawaladar_b, self.settlement_amount_range, t_settle),
                (hawaladar_b, beneficiary, self.transfer_amount_range, t_payout),
            ]
            if rng.random() < self.settlement_probability:
                t_settle_2 = t_settle + timedelta(minutes=rng.randint(30, 2_880))
                if rng.random() < 0.5:
                    edges.append(
                        (hawaladar_b, hawaladar_a, self.settlement_amount_range, t_settle_2)
                    )
                else:
                    edges.append(
                        (hawaladar_a, hawaladar_b, self.settlement_amount_range, t_settle_2)
                    )
            depth = len(edges)
            window_start = min(ts for _, _, _, ts in edges)
            window_end = max(ts for _, _, _, ts in edges)

            # Retail legs move like ordinary transfers; the hawaladar
            # netting wires settle bank-side.
            for src, dst, amount_range, ts in edges:
                amount = round_xof(rng.uniform(*amount_range))
                is_settlement = amount_range == self.settlement_amount_range
                tx_type = "bank_to_wallet" if is_settlement else "p2p"
                row, desc = _tx_row(
                    current_tx_id, src, dst, tx_type, amount, iso_ts(ts), fmt, rng=rng
                )
                batch_texts.append(desc)
                batch_rows.append(row)
                current_tx_id += 1

            embeddings = embedder.generate(batch_texts)

            for idx, r in enumerate(batch_rows):
                if fmt == "neptune":
                    tx_rows.append(r)
                else:
                    vec = embeddings[idx]
                    if isinstance(vec, np.ndarray):
                        vec = vec.tolist()
                    tx_rows.append(r + ["|".join(map(str, vec))])

            case_rows.append(
                [
                    f"hawala_{pattern_id}",
                    sender,
                    "hawala_network",
                    depth,
                    involved,
                    iso_ts(window_start),
                    iso_ts(window_end),
                ]
            )

        file_tx = os.path.join(fraud_dir, "transactions_fraud")
        file_cases = os.path.join(fraud_dir, "fraud_cases")
        append_csv(file_tx + ".csv", headers_tx, tx_rows)
        append_csv(file_cases + ".csv", headers_cases, case_rows)

        return len(tx_rows), current_tx_id


@dataclass
class SIMSwapFraudGenerator:
    """Generate SIM-swap account takeover fraud patterns.

    GSMA reports SIM-swap account takeover as an epidemic in the region: a
    fraudster bribes a telecom agent to swap a victim's SIM, then rapidly
    drains the victim's mobile money wallet by cashing out through several
    agents before the victim notices.

    Graph shape::

        victim -> cashout_agent_0
        victim -> cashout_agent_1
            ...
        victim -> cashout_agent_{n-1}

    A single victim account fans out to n cash-out agents. Unlike every
    other typology in this module, per-edge amounts are NOT drawn
    independently — each pattern draws one ``base_amount`` from
    ``amount_range`` and every cash-out edge amount is
    ``base_amount * uniform(1 - amount_jitter, 1 + amount_jitter)``, so
    amounts stay near-identical (fragments of one drained wallet balance),
    which is the detection signal this typology is meant to exercise. All
    edges also land within ``burst_window_minutes`` of each other.

    involved_accounts positional convention (pipe-joined):
        ``victim|cashout_agent_0|...|cashout_agent_{n-1}``
        i.e. ``accounts[0]`` = victim, ``accounts[1:]`` = cash-out agents.
        ``depth`` = n, the number of cash-out agents.

    Args:
        num_patterns: How many SIM-swap takeover patterns to create.
        num_agents_range: ``(min_agents, max_agents)`` - number of cash-out
            agents per pattern. Defaults to ``(3, 6)``.
        amount_range: Range the per-pattern ``base_amount`` is drawn from,
            in FCFA. Defaults to ``(20_000, 300_000)`` - wallet-balance
            scale.
        amount_jitter: Fractional jitter applied around ``base_amount`` for
            each cash-out edge. Defaults to ``0.05`` (±5%).
        burst_window_minutes: Max minutes between the first and last
            cash-out transaction in a pattern. Defaults to ``10``.
        benign_events_per_fraud: Ordinary phone-upgrade SIM swaps emitted
            into ``sim_events.csv`` per fraudulent one, so the presence of
            an event does not label the takeover. Defaults to ``20``.
    """

    num_patterns: int = 100
    num_agents_range: tuple[int, int] = (3, 6)
    amount_range: tuple[int, int] = (20_000, 300_000)
    amount_jitter: float = 0.05
    burst_window_minutes: int = 10
    benign_events_per_fraud: int = 20
    sim_start_date: str = "2024-01-01"
    sim_days: int = 90

    def generate(
        self,
        max_account_id: int,
        start_tx_id: int,
        embedder: EmbeddingGenerator,
        output_dir: str,
        fmt: str = "csv",
        compress: bool = False,
        rng: random.Random | None = None,
    ) -> tuple[int, int]:
        """Generate SIM-swap takeover patterns and append to fraud output files.

        Each pattern needs 1 + n distinct accounts (victim, n cash-out
        agents), where n is drawn from ``num_agents_range``. If fewer than
        4 accounts exist (the minimum for n=3), generation is skipped.

        Returns:
            ``(num_fraud_transactions, next_tx_id)``
        """
        import os

        from tqdm import tqdm

        if max_account_id < 4:
            return 0, start_tx_id

        rng = rng if rng is not None else _RNG
        fraud_dir = os.path.join(output_dir, "fraud")
        os.makedirs(fraud_dir, exist_ok=True)

        headers_tx = get_headers("transaction", fmt)  # type: ignore[arg-type]
        headers_cases = FRAUD_CASE_HEADERS

        tx_rows: list[list] = []
        case_rows: list[list] = []
        event_rows: list[list] = []
        current_tx_id = start_tx_id
        sim_start = parse_date(self.sim_start_date)

        for pattern_id in tqdm(range(self.num_patterns), desc="Generating SIM-swap patterns"):
            min_n, max_n = self.num_agents_range
            n = rng.randint(min_n, max_n)
            needed = n + 1
            if max_account_id < needed:
                n = max(min_n, max_account_id - 1)
                needed = n + 1
                if max_account_id < needed:
                    continue

            # The victim is a customer wallet; the drain fans out through
            # real agent-typed wallets (cash-out always has an agent leg).
            agent_uids = sample_agent_uids(max_account_id, n, rng)
            victim_uid = random_customer_uid(max_account_id, rng)
            if victim_uid in agent_uids:
                taken = set(agent_uids)
                victim_uid = next(u for u in range(max_account_id) if u not in taken)
            victim = f"acc_{victim_uid}"
            cashout_agents = [f"acc_{uid}" for uid in agent_uids]
            involved = "|".join([victim] + cashout_agents)

            base_amount = round_xof(rng.uniform(*self.amount_range))

            base_ts = random_timestamp(rng, sim_start, self.sim_days)
            max_gap = max(30, self.burst_window_minutes * 60 // n)
            elapsed_seconds = 0
            last_offset = 0

            # The takeover event itself: same msisdn and wallet, new SIM,
            # minutes before the first cash-out. IDs are assigned after the
            # benign decoys are mixed in, so nothing in the row identifies
            # the fraudulent swaps.
            swap_ts = base_ts - timedelta(minutes=rng.randint(2, 90))
            event_rows.append(
                ["", victim, msisdn_for(victim_uid), sim_id_for(victim_uid), "", iso_ts(swap_ts)]
            )

            batch_texts: list[str] = []
            batch_rows: list[list] = []

            for agent in cashout_agents:
                jitter = rng.uniform(1 - self.amount_jitter, 1 + self.amount_jitter)
                amount = round_xof(base_amount * jitter)

                tx_ts = iso_ts(base_ts + timedelta(seconds=elapsed_seconds))
                last_offset = elapsed_seconds
                elapsed_seconds += rng.randint(30, max_gap)

                row, desc = _tx_row(
                    current_tx_id,
                    victim,
                    agent,
                    "cash_out",
                    amount,
                    tx_ts,
                    fmt,
                    agent_id=agent,
                    rng=rng,
                )
                batch_texts.append(desc)
                batch_rows.append(row)
                current_tx_id += 1

            embeddings = embedder.generate(batch_texts)

            for idx, r in enumerate(batch_rows):
                if fmt == "neptune":
                    tx_rows.append(r)
                else:
                    vec = embeddings[idx]
                    if isinstance(vec, np.ndarray):
                        vec = vec.tolist()
                    tx_rows.append(r + ["|".join(map(str, vec))])

            case_rows.append(
                [
                    f"simswap_{pattern_id}",
                    victim,
                    "sim_swap_takeover",
                    n,
                    involved,
                    iso_ts(base_ts),
                    iso_ts(base_ts + timedelta(seconds=last_offset)),
                ]
            )

        # Benign decoy swaps: most SIM re-bindings are ordinary phone or
        # SIM upgrades. Without them the mere existence of a sim_events row
        # would label the takeovers (the tabular baseline exploited that).
        n_benign = len(event_rows) * self.benign_events_per_fraud
        for _ in range(n_benign):
            uid = random_customer_uid(max_account_id, rng)
            ts = random_timestamp(rng, sim_start, self.sim_days)
            event_rows.append(["", f"acc_{uid}", msisdn_for(uid), sim_id_for(uid), "", iso_ts(ts)])

        # Shuffle fraud and benign together, then assign uniform ids and
        # replacement-SIM serials so row order and naming carry no label.
        rng.shuffle(event_rows)
        for k, ev in enumerate(event_rows):
            ev[0] = f"simev_{start_tx_id}_{k}"
            ev[4] = f"sim_r{start_tx_id}_{k:06d}"

        file_tx = os.path.join(fraud_dir, "transactions_fraud")
        file_cases = os.path.join(fraud_dir, "fraud_cases")
        append_csv(file_tx + ".csv", headers_tx, tx_rows)
        append_csv(file_cases + ".csv", headers_cases, case_rows)
        append_csv(os.path.join(fraud_dir, "sim_events.csv"), SIM_EVENT_HEADERS, event_rows)

        return len(tx_rows), current_tx_id


@dataclass
class OverdraftMuleGenerator:
    """Generate overdraft / micro-loan mule chain fraud patterns.

    Fintech mobile lenders operating in UEMOA extend instant micro-loans
    through an app. Fraudsters open many one-time "mule" accounts, each
    takes out a small loan and immediately forwards the proceeds to a
    collector account, then defaults - the mule account is never reused.
    The collector consolidates and cashes out through an agent.

    Graph shape::

        mule_0 -> collector
        mule_1 -> collector
            ...
        mule_{n-1} -> collector
        collector -> agent          (consolidated cash-out)

    Unlike every other typology in this module, the final collector -> agent
    edge amount is NOT drawn independently — it equals the sum of all n
    mule loan amounts, modeling a real consolidated withdrawal.

    involved_accounts positional convention (pipe-joined):
        ``collector|mule_0|...|mule_{n-1}|agent``
        i.e. ``accounts[0]`` = collector, ``accounts[1:-1]`` = mules,
        ``accounts[-1]`` = cash-out agent.
        ``depth`` = n, the number of mule accounts.

    Args:
        num_patterns: How many mule-chain patterns to create.
        num_mules_range: ``(min_mules, max_mules)`` - number of one-time
            mule accounts per pattern. Defaults to ``(5, 15)``.
        loan_amount_range: Per-mule micro-loan amount range in FCFA, drawn
            independently per mule. Defaults to ``(25_000, 150_000)``.
    """

    num_patterns: int = 100
    num_mules_range: tuple[int, int] = (5, 15)
    loan_amount_range: tuple[int, int] = (25_000, 150_000)
    sim_start_date: str = "2024-01-01"
    sim_days: int = 90

    def generate(
        self,
        max_account_id: int,
        start_tx_id: int,
        embedder: EmbeddingGenerator,
        output_dir: str,
        fmt: str = "csv",
        compress: bool = False,
        rng: random.Random | None = None,
    ) -> tuple[int, int]:
        """Generate overdraft mule-chain patterns and append to fraud output files.

        Each pattern needs n + 2 distinct accounts (n mules, 1 collector,
        1 cash-out agent), where n is drawn from ``num_mules_range``. If
        fewer than 7 accounts exist (the minimum for n=5), generation is
        skipped.

        Returns:
            ``(num_fraud_transactions, next_tx_id)``
        """
        import os

        from tqdm import tqdm

        if max_account_id < 7:
            return 0, start_tx_id

        rng = rng if rng is not None else _RNG
        fraud_dir = os.path.join(output_dir, "fraud")
        os.makedirs(fraud_dir, exist_ok=True)

        headers_tx = get_headers("transaction", fmt)  # type: ignore[arg-type]
        headers_cases = FRAUD_CASE_HEADERS

        tx_rows: list[list] = []
        case_rows: list[list] = []
        current_tx_id = start_tx_id
        sim_start = parse_date(self.sim_start_date)

        for pattern_id in tqdm(range(self.num_patterns), desc="Generating overdraft mule patterns"):
            min_n, max_n = self.num_mules_range
            n = rng.randint(min_n, max_n)
            needed = n + 2
            if max_account_id < needed:
                n = max(min_n, max_account_id - 2)
                needed = n + 2
                if max_account_id < needed:
                    continue

            idxs = rng.sample(range(max_account_id), n + 1)
            collector = f"acc_{idxs[0]}"
            mules = [f"acc_{idxs[1 + i]}" for i in range(n)]
            # The consolidated withdrawal runs through a real agent wallet.
            agent_uid = random_agent_uid(max_account_id, rng)
            while agent_uid in idxs:
                agent_uid = rng.randrange(max_account_id)
            agent = f"acc_{agent_uid}"
            involved = "|".join([collector] + mules + [agent])

            batch_texts: list[str] = []
            batch_rows: list[list] = []

            # Loan disbursements are forwarded over a single day; the
            # collector cashes out a few hours after the last one.
            start_day = rng.randint(0, max(0, self.sim_days - 2))
            base = sim_start + timedelta(days=start_day)
            ts_list = sorted(random_timestamp(rng, base, 1) for _ in mules)

            mule_amounts: list[int] = []
            for mule, ts in zip(mules, ts_list, strict=True):
                amount = round_xof(rng.uniform(*self.loan_amount_range))
                mule_amounts.append(amount)
                row, desc = _tx_row(
                    current_tx_id, mule, collector, "p2p", amount, iso_ts(ts), fmt, rng=rng
                )
                batch_texts.append(desc)
                batch_rows.append(row)
                current_tx_id += 1

            consolidation_amount = sum(mule_amounts)
            t_cashout = ts_list[-1] + timedelta(minutes=rng.randint(60, 360))
            row, desc = _tx_row(
                current_tx_id,
                collector,
                agent,
                "cash_out",
                consolidation_amount,
                iso_ts(t_cashout),
                fmt,
                agent_id=agent,
                rng=rng,
            )
            batch_texts.append(desc)
            batch_rows.append(row)
            current_tx_id += 1

            embeddings = embedder.generate(batch_texts)

            for idx, r in enumerate(batch_rows):
                if fmt == "neptune":
                    tx_rows.append(r)
                else:
                    vec = embeddings[idx]
                    if isinstance(vec, np.ndarray):
                        vec = vec.tolist()
                    tx_rows.append(r + ["|".join(map(str, vec))])

            case_rows.append(
                [
                    f"mule_{pattern_id}",
                    collector,
                    "overdraft_mule_chain",
                    n,
                    involved,
                    iso_ts(ts_list[0]),
                    iso_ts(t_cashout),
                ]
            )

        file_tx = os.path.join(fraud_dir, "transactions_fraud")
        file_cases = os.path.join(fraud_dir, "fraud_cases")
        append_csv(file_tx + ".csv", headers_tx, tx_rows)
        append_csv(file_cases + ".csv", headers_cases, case_rows)

        return len(tx_rows), current_tx_id
