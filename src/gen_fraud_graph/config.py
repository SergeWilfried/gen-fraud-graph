# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the synthetic fraud graph generator."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Config:
    """Generator configuration.

    Args:
        scale_factor: Multiplier over the base sizes.  ``1.0`` produces ~10 M
            accounts and ~90 M transactions.  Use ``0.01`` for ~100 K accounts.
        sim_start_date: First day of the simulated activity window
            (``YYYY-MM-DD``).  Every transaction timestamp — legitimate or
            injected — falls inside this window.
        sim_days: Length of the simulated activity window in days.
        seed: Master seed for reproducible generation. With the same seed
            and the same configuration (including ``workers`` and
            ``batches_per_worker``, which define the chunk layout), every
            output file is byte-identical across runs. ``None`` (the
            default) draws fresh entropy each run.
        num_fraud_rings: Number of cyclic fraud patterns to inject.  When
            *None* it is derived automatically from *scale_factor*.
        fraud_ring_depth_range: Min/max depth (hops) of each fraud ring.
        num_structuring_patterns: Number of smurfing/structuring patterns to
            inject.  When *None* it is derived from *scale_factor*.
        structuring_smurfs_range: Min/max smurf feeder accounts per pattern.
        structuring_amount_range: Per-smurf transfer amount range in FCFA.
            Defaults to ``(4_000_000, 4_950_000)`` — deliberately below the
            BCEAO 5,000,000 FCFA cash payment limit, and overlapping the
            legitimate high-value tail.
        num_mobile_money_patterns: Number of mobile money agent-commission
            fraud patterns to inject.  When *None* it is derived from
            *scale_factor*.
        mobile_money_amount_range: Per-split transaction amount range in FCFA.
            Defaults to ``(5_000, 25_000)`` — inside the bulk of the normal
            cash-in distribution so split bursts are only visible through
            velocity, not amount.
        num_trade_based_ml_patterns: Number of trade-based money laundering
            (TBML) patterns to inject.  When *None* it is derived from
            *scale_factor*.
        trade_based_ml_amount_range: Per-edge transaction amount range in
            FCFA for TBML patterns.  Defaults to ``(2_000_000, 15_000_000)``
            — inside the heavy right tail of the normal amount distribution
            so TBML edges are not separable on amount alone.
        trade_based_ml_intermediaries_range: Min/max number of layering
            intermediary accounts per TBML pattern.
        num_hawala_patterns: Number of hawala / informal value transfer
            network patterns to inject.  When *None* it is derived from
            *scale_factor*.
        hawala_settlement_amount_range: Hawaladar-to-hawaladar
            settlement/debt-netting amount range in FCFA.  Defaults to
            ``(2_000_000, 12_000_000)`` — overlapping the normal tail.
        hawala_transfer_amount_range: Sender/beneficiary leg (deposit and
            payout) amount range in FCFA.  Defaults to ``(50_000,
            1_500_000)`` — retail remittance scale, deliberately smaller
            than the settlement legs.
        num_sim_swap_patterns: Number of SIM-swap account takeover patterns
            to inject.  When *None* it is derived from *scale_factor*.
        sim_swap_amount_range: Range each pattern's base cash-out amount is
            drawn from, in FCFA.  Defaults to ``(20_000, 300_000)`` —
            wallet-balance scale.
        sim_swap_agents_range: Min/max number of cash-out agents per
            SIM-swap pattern.
        num_overdraft_mule_patterns: Number of overdraft/micro-loan mule
            chain patterns to inject.  When *None* it is derived from
            *scale_factor*.
        overdraft_mule_loan_amount_range: Per-mule micro-loan amount range
            in FCFA.  Defaults to ``(25_000, 150_000)``.
        overdraft_mule_num_mules_range: Min/max number of one-time mule
            accounts per pattern.
        embedding_provider: ``"fake"`` (random vectors, no deps), ``"local"``
            (SentenceTransformers), or ``"openai"`` (requires API key).
        embedding_dim: Dimensionality of generated embeddings.
        workers: Parallel processes for account/transaction generation.
        batches_per_worker: File chunks each worker produces.
        output_format: ``"csv"`` (generic) or ``"neptune"`` (AWS Neptune
            bulk-load headers).
        compress: Whether to ZIP the output CSV files.
        output_dir: Destination directory for generated files.
    """

    scale_factor: float = 1.0
    sim_start_date: str = "2024-01-01"
    sim_days: int = 90
    seed: int | None = None
    num_fraud_rings: int | None = None
    fraud_ring_depth_range: tuple[int, int] = (4, 7)
    num_structuring_patterns: int | None = None
    structuring_smurfs_range: tuple[int, int] = (3, 10)
    structuring_amount_range: tuple[int, int] = (4_000_000, 4_950_000)
    num_mobile_money_patterns: int | None = None
    mobile_money_amount_range: tuple[int, int] = (5_000, 25_000)
    num_trade_based_ml_patterns: int | None = None
    trade_based_ml_amount_range: tuple[int, int] = (2_000_000, 15_000_000)
    trade_based_ml_intermediaries_range: tuple[int, int] = (3, 5)
    num_hawala_patterns: int | None = None
    hawala_settlement_amount_range: tuple[int, int] = (2_000_000, 12_000_000)
    hawala_transfer_amount_range: tuple[int, int] = (50_000, 1_500_000)
    num_sim_swap_patterns: int | None = None
    sim_swap_amount_range: tuple[int, int] = (20_000, 300_000)
    sim_swap_agents_range: tuple[int, int] = (3, 6)
    num_overdraft_mule_patterns: int | None = None
    overdraft_mule_loan_amount_range: tuple[int, int] = (25_000, 150_000)
    overdraft_mule_num_mules_range: tuple[int, int] = (5, 15)
    embedding_provider: Literal["fake", "local", "openai"] = "fake"
    embedding_dim: int = 768
    workers: int = 1
    batches_per_worker: int = 1
    output_format: Literal["csv", "neptune"] = "csv"
    compress: bool = False
    output_dir: str = "data"

    # Derived — computed in __post_init__
    num_accounts: int = field(init=False)
    num_transactions: int = field(init=False)

    def __post_init__(self) -> None:
        self.num_accounts = int(10_000_000 * self.scale_factor)
        self.num_transactions = int(90_000_000 * self.scale_factor)
        if self.num_fraud_rings is None:
            self.num_fraud_rings = max(10, int(1000 * self.scale_factor))
        if self.num_structuring_patterns is None:
            self.num_structuring_patterns = max(10, int(500 * self.scale_factor))
        if self.num_mobile_money_patterns is None:
            self.num_mobile_money_patterns = max(10, int(800 * self.scale_factor))
        if self.num_trade_based_ml_patterns is None:
            self.num_trade_based_ml_patterns = max(10, int(300 * self.scale_factor))
        if self.num_hawala_patterns is None:
            self.num_hawala_patterns = max(10, int(400 * self.scale_factor))
        if self.num_sim_swap_patterns is None:
            self.num_sim_swap_patterns = max(10, int(600 * self.scale_factor))
        if self.num_overdraft_mule_patterns is None:
            self.num_overdraft_mule_patterns = max(10, int(350 * self.scale_factor))
