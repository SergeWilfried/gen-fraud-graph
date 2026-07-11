# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Fraud typology definitions for synthetic graph injection."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from gen_fraud_graph.embeddings import EmbeddingGenerator
from gen_fraud_graph.exporters import append_csv, get_headers, write_output

# ---------------------------------------------------------------------------
# Suspicious transaction descriptions used across typologies
# ---------------------------------------------------------------------------

SUSPICIOUS_DESCRIPTIONS: list[str] = [
    "transfert offshore vers paradis fiscal",
    "dépôt espèces sous le seuil de déclaration",
    "transfert rapide entre comptes",
    "paiement société écran",
    "transfert en cascade via intermédiaire",
    "transaction circulaire",
    "activité soudaine sur compte dormant",
    "virement transfrontalier de montant élevé",
]

# Description specific to structuring/smurfing patterns.
STRUCTURING_DESCRIPTIONS: list[str] = [
    "dépôt espèces sous le seuil de déclaration",
    "multiples petits dépôts le même jour",
    "paiement fractionné juste sous la limite",
    "dépôt espèces progressif au guichet",
    "dépôt espèces incrémental",
    "dépôt répété près de la limite",
    "transfert fragmenté pour échapper au contrôle",
]

# ---------------------------------------------------------------------------
# Fraud ring generator (cyclic money-laundering patterns)
# ---------------------------------------------------------------------------


@dataclass
class FraudRingGenerator:
    """Generate cyclic fraud-ring patterns.

    Each ring is a cycle of ``depth`` accounts connected by suspicious
    high-value transactions.

    Args:
        num_rings: How many rings to create.
        depth_range: ``(min_depth, max_depth)`` hops per ring.
        amount: Fixed transaction amount injected in fraud edges.
    """

    num_rings: int = 100
    depth_range: tuple[int, int] = (4, 7)
    amount: float = 12_000_000.00
    _descriptions: list[str] = field(default_factory=lambda: SUSPICIOUS_DESCRIPTIONS)

    def generate(
        self,
        max_account_id: int,
        start_tx_id: int,
        embedder: EmbeddingGenerator,
        output_dir: str,
        fmt: str = "csv",
        compress: bool = False,
    ) -> tuple[int, int]:
        """Generate fraud rings and write output files.

        Returns:
            ``(num_fraud_transactions, next_tx_id)``
        """
        import os

        from tqdm import tqdm

        fraud_dir = os.path.join(output_dir, "fraud")
        os.makedirs(fraud_dir, exist_ok=True)

        headers_tx = get_headers("transaction", fmt)  # type: ignore[arg-type]
        headers_cases = [
            "pattern_id",
            "start_acc_id",
            "pattern_type",
            "depth",
            "involved_accounts",
        ]

        tx_rows: list[list] = []
        case_rows: list[list] = []
        current_tx_id = start_tx_id
        # Allocate every ring's accounts up front from one pool of distinct
        # ids, then give each ring its own slice. Overlapping ranges would
        # merge two rings into a single non-cycle component and make the
        # per-ring involved_accounts labels ambiguous.
        min_d, max_d = self.depth_range
        depths = [random.randint(min_d, max_d) for _ in range(self.num_rings)]
        total_needed = sum(depths)
        if total_needed > max_account_id:
            raise ValueError(
                f"{self.num_rings} fraud rings need {total_needed} distinct "
                f"accounts but only {max_account_id} exist; lower the ring "
                f"count or raise the account scale"
            )
        account_pool = random.sample(range(max_account_id), total_needed)
        pool_offset = 0

        for pattern_id in tqdm(range(self.num_rings), desc="Generating fraud rings"):
            depth = depths[pattern_id]
            ring_ids = account_pool[pool_offset : pool_offset + depth]
            pool_offset += depth

            accounts = [f"acc_{i}" for i in ring_ids]
            involved = "|".join(accounts)

            batch_texts: list[str] = []
            batch_rows: list[list] = []

            for k in range(depth):
                src = accounts[k]
                dst = accounts[(k + 1) % depth]
                desc = random.choice(self._descriptions)
                batch_texts.append(desc)

                row: list = [f"tx_{current_tx_id}", src, dst]
                if fmt == "neptune":
                    row.append("TRANSFER")
                row.extend([self.amount, "2024-01-01T12:00:00", desc])
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
                    f"pat_{pattern_id}",
                    accounts[0],
                    "cycle",
                    depth,
                    involved,
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
    amount_range: tuple[float, float] = (4_000_000.00, 4_900_000.00)
    _descriptions: list[str] = field(default_factory=lambda: STRUCTURING_DESCRIPTIONS)

    def generate(
        self,
        max_account_id: int,
        start_tx_id: int,
        embedder: EmbeddingGenerator,
        output_dir: str,
        fmt: str = "csv",
        compress: bool = False,
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

        fraud_dir = os.path.join(output_dir, "fraud")
        os.makedirs(fraud_dir, exist_ok=True)

        headers_tx = get_headers("transaction", fmt)  # type: ignore[arg-type]
        headers_cases = [
            "pattern_id",
            "start_acc_id",
            "pattern_type",
            "depth",
            "involved_accounts",
        ]

        tx_rows: list[list] = []
        case_rows: list[list] = []
        current_tx_id = start_tx_id

        for pattern_id in tqdm(range(self.num_patterns), desc="Generating structuring patterns"):
            min_s, max_s = self.smurfs_range
            num_smurfs = random.randint(min_s, max_s)

            # The coordinator sits at a random offset; smurfs occupy the
            # num_smurfs slots immediately after it.  We need num_smurfs + 1
            # consecutive IDs so we guard against tiny account pools.
            needed = num_smurfs + 1
            if max_account_id < needed:
                coordinator_idx = 0
            else:
                coordinator_idx = random.randint(0, max_account_id - needed)

            coordinator = f"acc_{coordinator_idx}"
            smurfs = [f"acc_{coordinator_idx + 1 + i}" for i in range(num_smurfs)]
            involved = "|".join([coordinator] + smurfs)

            batch_texts: list[str] = []
            batch_rows: list[list] = []

            for smurf in smurfs:
                amount = round(random.uniform(*self.amount_range), 2)
                desc = random.choice(self._descriptions)
                batch_texts.append(desc)

                row: list = [f"tx_{current_tx_id}", smurf, coordinator]
                if fmt == "neptune":
                    row.append("TRANSFER")
                row.extend([amount, "2024-01-01T12:00:00", desc])
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
                ]
            )

        # Append to the same fraud files so both typologies land in one CSV.
        file_tx = os.path.join(fraud_dir, "transactions_fraud")
        file_cases = os.path.join(fraud_dir, "fraud_cases")
        append_csv(file_tx + ".csv", headers_tx, tx_rows)
        append_csv(file_cases + ".csv", headers_cases, case_rows)

        return len(tx_rows), current_tx_id


MOBILE_MONEY_DESCRIPTIONS: list[str] = [
    "dépôt cash wave",
    "retrait espèces agent",
    "transfert mobile money",
    "commission agent m-pesa/orange",
    "paiement marchand mobile",
]


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
    amount_range: tuple[float, float] = (10_000.00, 50_000.00)
    burst_window_minutes: int = 10  # max minutes between split transactions in a burst
    _descriptions: list[str] = field(default_factory=lambda: MOBILE_MONEY_DESCRIPTIONS)

    def generate(
        self,
        max_account_id: int,
        start_tx_id: int,
        embedder: EmbeddingGenerator,
        output_dir: str,
        fmt: str = "csv",
        compress: bool = False,
    ) -> tuple[int, int]:
        """Generate mobile money patterns and append to fraud output files."""
        import os
        from datetime import datetime, timedelta

        from tqdm import tqdm

        # Pre-loop guard: need at least 2 distinct accounts to form an agent->customer edge.
        if max_account_id < 2:
            return 0, start_tx_id

        fraud_dir = os.path.join(output_dir, "fraud")
        os.makedirs(fraud_dir, exist_ok=True)

        headers_tx = get_headers("transaction", fmt)  # type: ignore[arg-type]
        headers_cases = [
            "pattern_id",
            "start_acc_id",
            "pattern_type",
            "depth",
            "involved_accounts",
        ]

        tx_rows: list[list] = []
        case_rows: list[list] = []
        current_tx_id = start_tx_id

        for pattern_id in tqdm(range(self.num_patterns), desc="Generating mobile money fraud"):
            agent_idx, customer_idx = random.sample(range(max_account_id), 2)

            agent = f"acc_{agent_idx}"
            customer = f"acc_{customer_idx}"
            involved = f"{agent}|{customer}"

            num_txs = random.randint(4, 10)

            # Burst base time: random minute within a single day, then each
            # split tx is offset by a small random increment to simulate a
            # rapid burst — essential for time-based detection algorithms.
            base_ts = datetime(2024, 1, 1, random.randint(8, 20), random.randint(0, 59))

            batch_texts: list[str] = []
            batch_rows: list[list] = []
            elapsed_seconds = 0

            for _ in range(num_txs):
                amount = round(random.uniform(*self.amount_range), 2)
                desc = random.choice(self._descriptions)
                batch_texts.append(desc)

                tx_ts = (base_ts + timedelta(seconds=elapsed_seconds)).strftime("%Y-%m-%dT%H:%M:%S")
                max_gap = max(30, self.burst_window_minutes * 60 // num_txs)
                elapsed_seconds += random.randint(30, max_gap)

                row: list = [f"tx_{current_tx_id}", agent, customer]
                if fmt == "neptune":
                    row.append("TRANSFER")
                row.extend([amount, tx_ts, desc])
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
                ]
            )

        file_tx = os.path.join(fraud_dir, "transactions_fraud")
        file_cases = os.path.join(fraud_dir, "fraud_cases")
        append_csv(file_tx + ".csv", headers_tx, tx_rows)
        append_csv(file_cases + ".csv", headers_cases, case_rows)

        return len(tx_rows), current_tx_id


TBML_DESCRIPTIONS: list[str] = [
    "paiement facture import coton",
    "règlement fournisseur Chine",
    "sur-facturation équipements",
    "importation fantôme marchandises",
]


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
    amount_range: tuple[float, float] = (20_000_000.00, 150_000_000.00)
    _descriptions: list[str] = field(default_factory=lambda: TBML_DESCRIPTIONS)

    def generate(
        self,
        max_account_id: int,
        start_tx_id: int,
        embedder: EmbeddingGenerator,
        output_dir: str,
        fmt: str = "csv",
        compress: bool = False,
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

        fraud_dir = os.path.join(output_dir, "fraud")
        os.makedirs(fraud_dir, exist_ok=True)

        headers_tx = get_headers("transaction", fmt)  # type: ignore[arg-type]
        headers_cases = [
            "pattern_id",
            "start_acc_id",
            "pattern_type",
            "depth",
            "involved_accounts",
        ]

        tx_rows: list[list] = []
        case_rows: list[list] = []
        current_tx_id = start_tx_id

        for pattern_id in tqdm(range(self.num_patterns), desc="Generating TBML patterns"):
            min_k, max_k = self.intermediaries_range
            k = random.randint(min_k, max_k)
            needed = k + 3
            if max_account_id < needed:
                k = max(min_k, max_account_id - 3)
                needed = k + 3
                if max_account_id < needed:
                    continue

            idxs = random.sample(range(max_account_id), needed)
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

            for src, dst in edges:
                amount = round(random.uniform(*self.amount_range), 2)
                desc = random.choice(self._descriptions)
                batch_texts.append(desc)

                row: list = [f"tx_{current_tx_id}", src, dst]
                if fmt == "neptune":
                    row.append("TRANSFER")
                row.extend([amount, "2024-01-01T12:00:00", desc])
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
                    f"tbml_{pattern_id}",
                    exporter,
                    "trade_based_ml",
                    k,
                    involved,
                ]
            )

        file_tx = os.path.join(fraud_dir, "transactions_fraud")
        file_cases = os.path.join(fraud_dir, "fraud_cases")
        append_csv(file_tx + ".csv", headers_tx, tx_rows)
        append_csv(file_cases + ".csv", headers_cases, case_rows)

        return len(tx_rows), current_tx_id


HAWALA_DESCRIPTIONS: list[str] = [
    "transfert informel corridor Dakar-Abidjan",
    "règlement hawaladar",
    "compensation informelle",
    "envoi de fonds sous-régional",
]


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
    transfer_amount_range: tuple[float, float] = (100_000.00, 2_000_000.00)
    settlement_amount_range: tuple[float, float] = (5_000_000.00, 50_000_000.00)
    settlement_probability: float = 0.3
    _descriptions: list[str] = field(default_factory=lambda: HAWALA_DESCRIPTIONS)

    def generate(
        self,
        max_account_id: int,
        start_tx_id: int,
        embedder: EmbeddingGenerator,
        output_dir: str,
        fmt: str = "csv",
        compress: bool = False,
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

        fraud_dir = os.path.join(output_dir, "fraud")
        os.makedirs(fraud_dir, exist_ok=True)

        headers_tx = get_headers("transaction", fmt)  # type: ignore[arg-type]
        headers_cases = [
            "pattern_id",
            "start_acc_id",
            "pattern_type",
            "depth",
            "involved_accounts",
        ]

        tx_rows: list[list] = []
        case_rows: list[list] = []
        current_tx_id = start_tx_id

        for pattern_id in tqdm(range(self.num_patterns), desc="Generating hawala patterns"):
            idxs = random.sample(range(max_account_id), 4)
            sender = f"acc_{idxs[0]}"
            hawaladar_a = f"acc_{idxs[1]}"
            hawaladar_b = f"acc_{idxs[2]}"
            beneficiary = f"acc_{idxs[3]}"
            involved = "|".join([sender, hawaladar_a, hawaladar_b, beneficiary])

            batch_texts: list[str] = []
            batch_rows: list[list] = []

            edges: list[tuple[str, str, tuple[float, float]]] = [
                (sender, hawaladar_a, self.transfer_amount_range),
                (hawaladar_a, hawaladar_b, self.settlement_amount_range),
                (hawaladar_b, beneficiary, self.transfer_amount_range),
            ]
            if random.random() < self.settlement_probability:
                if random.random() < 0.5:
                    edges.append((hawaladar_b, hawaladar_a, self.settlement_amount_range))
                else:
                    edges.append((hawaladar_a, hawaladar_b, self.settlement_amount_range))
            depth = len(edges)

            for src, dst, amount_range in edges:
                amount = round(random.uniform(*amount_range), 2)
                desc = random.choice(self._descriptions)
                batch_texts.append(desc)

                row: list = [f"tx_{current_tx_id}", src, dst]
                if fmt == "neptune":
                    row.append("TRANSFER")
                row.extend([amount, "2024-01-01T12:00:00", desc])
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
                ]
            )

        file_tx = os.path.join(fraud_dir, "transactions_fraud")
        file_cases = os.path.join(fraud_dir, "fraud_cases")
        append_csv(file_tx + ".csv", headers_tx, tx_rows)
        append_csv(file_cases + ".csv", headers_cases, case_rows)

        return len(tx_rows), current_tx_id


SIM_SWAP_DESCRIPTIONS: list[str] = [
    "retrait immédiat après changement de carte SIM",
    "vidage de compte suite à piratage SIM",
    "retrait cash-out agent après prise de contrôle",
    "transfert suspect fenêtre courte post SIM swap",
]


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
    """

    num_patterns: int = 100
    num_agents_range: tuple[int, int] = (3, 6)
    amount_range: tuple[float, float] = (20_000.00, 300_000.00)
    amount_jitter: float = 0.05
    burst_window_minutes: int = 10
    _descriptions: list[str] = field(default_factory=lambda: SIM_SWAP_DESCRIPTIONS)

    def generate(
        self,
        max_account_id: int,
        start_tx_id: int,
        embedder: EmbeddingGenerator,
        output_dir: str,
        fmt: str = "csv",
        compress: bool = False,
    ) -> tuple[int, int]:
        """Generate SIM-swap takeover patterns and append to fraud output files.

        Each pattern needs 1 + n distinct accounts (victim, n cash-out
        agents), where n is drawn from ``num_agents_range``. If fewer than
        4 accounts exist (the minimum for n=3), generation is skipped.

        Returns:
            ``(num_fraud_transactions, next_tx_id)``
        """
        import os
        from datetime import datetime, timedelta

        from tqdm import tqdm

        if max_account_id < 4:
            return 0, start_tx_id

        fraud_dir = os.path.join(output_dir, "fraud")
        os.makedirs(fraud_dir, exist_ok=True)

        headers_tx = get_headers("transaction", fmt)  # type: ignore[arg-type]
        headers_cases = [
            "pattern_id",
            "start_acc_id",
            "pattern_type",
            "depth",
            "involved_accounts",
        ]

        tx_rows: list[list] = []
        case_rows: list[list] = []
        current_tx_id = start_tx_id

        for pattern_id in tqdm(range(self.num_patterns), desc="Generating SIM-swap patterns"):
            min_n, max_n = self.num_agents_range
            n = random.randint(min_n, max_n)
            needed = n + 1
            if max_account_id < needed:
                n = max(min_n, max_account_id - 1)
                needed = n + 1
                if max_account_id < needed:
                    continue

            idxs = random.sample(range(max_account_id), needed)
            victim = f"acc_{idxs[0]}"
            cashout_agents = [f"acc_{idxs[1 + i]}" for i in range(n)]
            involved = "|".join([victim] + cashout_agents)

            base_amount = round(random.uniform(*self.amount_range), 2)

            base_ts = datetime(2024, 1, 1, random.randint(8, 20), random.randint(0, 59))
            max_gap = max(30, self.burst_window_minutes * 60 // n)
            elapsed_seconds = 0

            batch_texts: list[str] = []
            batch_rows: list[list] = []

            for agent in cashout_agents:
                jitter = random.uniform(1 - self.amount_jitter, 1 + self.amount_jitter)
                amount = round(base_amount * jitter, 2)
                desc = random.choice(self._descriptions)
                batch_texts.append(desc)

                tx_ts = (base_ts + timedelta(seconds=elapsed_seconds)).strftime("%Y-%m-%dT%H:%M:%S")
                elapsed_seconds += random.randint(30, max_gap)

                row: list = [f"tx_{current_tx_id}", victim, agent]
                if fmt == "neptune":
                    row.append("TRANSFER")
                row.extend([amount, tx_ts, desc])
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
                ]
            )

        file_tx = os.path.join(fraud_dir, "transactions_fraud")
        file_cases = os.path.join(fraud_dir, "fraud_cases")
        append_csv(file_tx + ".csv", headers_tx, tx_rows)
        append_csv(file_cases + ".csv", headers_cases, case_rows)

        return len(tx_rows), current_tx_id


OVERDRAFT_MULE_DESCRIPTIONS: list[str] = [
    "décaissement prêt mobile microcrédit",
    "transfert produit de prêt vers collecteur",
    "retrait consolidé agent après décaissements",
    "compte mule prêt unique",
]


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
    loan_amount_range: tuple[float, float] = (25_000.00, 150_000.00)
    _descriptions: list[str] = field(default_factory=lambda: OVERDRAFT_MULE_DESCRIPTIONS)

    def generate(
        self,
        max_account_id: int,
        start_tx_id: int,
        embedder: EmbeddingGenerator,
        output_dir: str,
        fmt: str = "csv",
        compress: bool = False,
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

        fraud_dir = os.path.join(output_dir, "fraud")
        os.makedirs(fraud_dir, exist_ok=True)

        headers_tx = get_headers("transaction", fmt)  # type: ignore[arg-type]
        headers_cases = [
            "pattern_id",
            "start_acc_id",
            "pattern_type",
            "depth",
            "involved_accounts",
        ]

        tx_rows: list[list] = []
        case_rows: list[list] = []
        current_tx_id = start_tx_id

        for pattern_id in tqdm(range(self.num_patterns), desc="Generating overdraft mule patterns"):
            min_n, max_n = self.num_mules_range
            n = random.randint(min_n, max_n)
            needed = n + 2
            if max_account_id < needed:
                n = max(min_n, max_account_id - 2)
                needed = n + 2
                if max_account_id < needed:
                    continue

            idxs = random.sample(range(max_account_id), needed)
            collector = f"acc_{idxs[0]}"
            mules = [f"acc_{idxs[1 + i]}" for i in range(n)]
            agent = f"acc_{idxs[-1]}"
            involved = "|".join([collector] + mules + [agent])

            batch_texts: list[str] = []
            batch_rows: list[list] = []

            mule_amounts: list[float] = []
            for mule in mules:
                amount = round(random.uniform(*self.loan_amount_range), 2)
                mule_amounts.append(amount)
                desc = random.choice(self._descriptions)
                batch_texts.append(desc)

                row: list = [f"tx_{current_tx_id}", mule, collector]
                if fmt == "neptune":
                    row.append("TRANSFER")
                row.extend([amount, "2024-01-01T12:00:00", desc])
                batch_rows.append(row)
                current_tx_id += 1

            consolidation_amount = round(sum(mule_amounts), 2)
            desc = random.choice(self._descriptions)
            batch_texts.append(desc)
            row = [f"tx_{current_tx_id}", collector, agent]
            if fmt == "neptune":
                row.append("TRANSFER")
            row.extend([consolidation_amount, "2024-01-01T13:00:00", desc])
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
                ]
            )

        file_tx = os.path.join(fraud_dir, "transactions_fraud")
        file_cases = os.path.join(fraud_dir, "fraud_cases")
        append_csv(file_tx + ".csv", headers_tx, tx_rows)
        append_csv(file_cases + ".csv", headers_cases, case_rows)

        return len(tx_rows), current_tx_id
