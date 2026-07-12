# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Mobile-money graph schema: node/edge specs, role model, and sampling.

This module is the single source of truth for what a wallet, a transaction,
and a fraud case look like. The CSV headers used by the exporters are
derived from the dataclasses below, so schema changes happen in one place.

Two design rules apply everywhere:

* **Roles live in the graph, not in the ground truth.** Agent, merchant,
  and aggregator wallets are first-class node attributes assigned
  deterministically from the account ID, so typologies that depend on the
  agent hierarchy (commission farming, SIM-swap cash-out, mule
  consolidation) are expressible in the data a model actually sees.
* **No single column separates fraud from background.** Amounts,
  timestamps, and descriptions are drawn from shared machinery so injected
  patterns are only detectable through graph structure, velocity, and the
  typed fields — never through a vocabulary or constant-value leak.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, fields
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

ACCOUNT_TYPES: tuple[str, ...] = (
    "customer",
    "agent",
    "super_agent",
    "merchant",
    "aggregator",
)

KYC_TIERS: tuple[str, ...] = ("unverified", "light", "full")

TX_TYPES: tuple[str, ...] = (
    "cash_in",
    "cash_out",
    "p2p",
    "merchant_payment",
    "airtime",
    "bill_pay",
    "bank_to_wallet",
)

CHANNELS: tuple[str, ...] = ("ussd", "app", "agent_pos", "api")

# BCEAO-style tiered e-money caps (XOF): the per-transaction ceiling a
# wallet of each KYC tier may move. Structuring patterns are defined by
# hugging these limits from below.
TIER_TX_CAPS: dict[str, int] = {
    "unverified": 200_000,
    "light": 2_000_000,
    "full": 10_000_000,
}


# ---------------------------------------------------------------------------
# Node / edge / ground-truth record specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WalletNode:
    """One mobile-money wallet (graph node).

    ``account_id`` stays the join key for edges; ``msisdn`` is the
    subscriber identity. Keeping both mirrors operator data models and
    lets SIM-swap events re-bind a SIM to the same msisdn.
    """

    account_id: str
    msisdn: str
    customer_name: str
    account_type: str  # one of ACCOUNT_TYPES
    kyc_tier: str  # one of KYC_TIERS
    sim_id: str
    device_id: str
    registration_agent_id: str  # agent that opened the wallet ("" if n/a)
    zone: str  # coarse geography (region/district)
    balance: int  # XOF
    float_balance: int  # XOF e-float; > 0 only for agent-side wallets
    risk_score: float
    creation_date: str


@dataclass(frozen=True)
class TransactionEdge:
    """One transaction (graph edge)."""

    tx_id: str
    src_id: str
    dst_id: str
    tx_type: str  # one of TX_TYPES
    channel: str  # one of CHANNELS
    agent_id: str  # facilitating agent on cash legs ("" otherwise)
    amount: int  # XOF
    fee: int  # XOF, paid by the sender
    commission: int  # XOF, operator -> agent on cash legs
    timestamp: str
    description: str


@dataclass(frozen=True)
class FraudCase:
    """Ground-truth record for one injected pattern."""

    pattern_id: str
    start_acc_id: str
    pattern_type: str
    depth: int
    involved_accounts: str  # pipe-joined, role encoded by position
    window_start: str
    window_end: str


@dataclass(frozen=True)
class SimSwapEvent:
    """A SIM re-binding: same msisdn and wallet, new physical SIM.

    Emitted alongside ``sim_swap_takeover`` patterns so the takeover is
    expressible in the data (join wallet -> event -> cash-out burst), not
    only in the fraud_cases ground truth.
    """

    event_id: str
    account_id: str
    msisdn: str
    old_sim_id: str
    new_sim_id: str
    swap_ts: str


def _headers(cls: type) -> list[str]:
    return [f.name for f in fields(cls)]


ACCOUNT_HEADERS: list[str] = _headers(WalletNode)
TRANSACTION_HEADERS: list[str] = _headers(TransactionEdge)
FRAUD_CASE_HEADERS: list[str] = _headers(FraudCase)
SIM_EVENT_HEADERS: list[str] = _headers(SimSwapEvent)

# Neptune bulk-load property types for the CSV columns above.
_NEPTUNE_TYPES: dict[str, str] = {
    "balance": "Long",
    "float_balance": "Long",
    "risk_score": "Double",
    "amount": "Long",
    "fee": "Long",
    "commission": "Long",
}


def neptune_account_headers() -> list[str]:
    cols = [f"{n}:{_NEPTUNE_TYPES.get(n, 'String')}" for n in ACCOUNT_HEADERS[1:]]
    return ["~id", "~label", *cols, "embedding:vector"]


def neptune_transaction_headers() -> list[str]:
    cols = [f"{n}:{_NEPTUNE_TYPES.get(n, 'String')}" for n in TRANSACTION_HEADERS[3:]]
    return ["~id", "~from", "~to", "~label", *cols]


# ---------------------------------------------------------------------------
# Role model — deterministic in the account ID
# ---------------------------------------------------------------------------

# Account roles are a pure function of the numeric account ID so that the
# parallel account workers and the fraud typology generators agree on who
# is an agent without any coordination. Within every block of 1000 IDs:
#   slots 0-1   super_agent (0.2%)
#   slots 2-21  agent       (2.0%)
#   slots 22-31 merchant    (1.0%)
#   slots 32-34 aggregator  (0.3%)
#   slots 35+   customer    (96.5%)
_DEFAULT_RNG = random.Random()

ROLE_MODULUS = 1000
_SUPER_AGENT_SLOTS = range(0, 2)
_AGENT_SLOTS = range(2, 22)
_MERCHANT_SLOTS = range(22, 32)
_AGGREGATOR_SLOTS = range(32, 35)


def account_type_for(uid: int) -> str:
    """Return the account type for a numeric account ID."""
    slot = uid % ROLE_MODULUS
    if slot in _SUPER_AGENT_SLOTS:
        return "super_agent"
    if slot in _AGENT_SLOTS:
        return "agent"
    if slot in _MERCHANT_SLOTS:
        return "merchant"
    if slot in _AGGREGATOR_SLOTS:
        return "aggregator"
    return "customer"


def _random_slot_uid(max_account_id: int, slots: range, rng: random.Random | None = None) -> int:
    """Draw a random uid whose role slot falls in ``slots``.

    Falls back to an arbitrary uid when the pool is too small to contain
    the requested role — tiny test pools lose role fidelity rather than
    failing.
    """
    r = rng if rng is not None else _DEFAULT_RNG
    if max_account_id <= slots.start:
        return r.randrange(max_account_id)
    num_blocks = (max_account_id + ROLE_MODULUS - 1) // ROLE_MODULUS
    for _ in range(64):
        uid = r.randrange(num_blocks) * ROLE_MODULUS + r.choice(slots)
        if uid < max_account_id:
            return uid
    return slots.start  # block 0 always holds every slot ≤ pool size


def random_agent_uid(max_account_id: int, rng: random.Random | None = None) -> int:
    return _random_slot_uid(max_account_id, _AGENT_SLOTS, rng)


def random_merchant_uid(max_account_id: int, rng: random.Random | None = None) -> int:
    return _random_slot_uid(max_account_id, _MERCHANT_SLOTS, rng)


def random_aggregator_uid(max_account_id: int, rng: random.Random | None = None) -> int:
    return _random_slot_uid(max_account_id, _AGGREGATOR_SLOTS, rng)


def random_customer_uid(max_account_id: int, rng: random.Random | None = None) -> int:
    """Draw a random customer uid (arbitrary uid on tiny role-less pools)."""
    r = rng if rng is not None else _DEFAULT_RNG
    for _ in range(200):
        uid = r.randrange(max_account_id)
        if account_type_for(uid) == "customer":
            return uid
    return r.randrange(max_account_id)


def sample_agent_uids(max_account_id: int, n: int, rng: random.Random | None = None) -> list[int]:
    """Draw ``n`` distinct agent uids, padding with arbitrary distinct uids
    when the pool holds fewer than ``n`` agents."""
    r = rng if rng is not None else _DEFAULT_RNG
    chosen: set[int] = set()
    for _ in range(64 * n):
        if len(chosen) == n:
            break
        chosen.add(random_agent_uid(max_account_id, r))
    while len(chosen) < n:
        chosen.add(r.randrange(max_account_id))
    return list(chosen)


# ---------------------------------------------------------------------------
# Identity attributes
# ---------------------------------------------------------------------------


def msisdn_for(uid: int) -> str:
    """Unique Senegalese-format MSISDN for a numeric account ID."""
    return f"+221{70_000_000 + uid}"


def sim_id_for(uid: int) -> str:
    """The SIM originally bound to the account (SIM swaps re-bind it)."""
    return f"sim_{uid:08d}"


# Region/district weights, urban-heavy like real wallet distribution.
ZONES: list[str] = [
    "DAKAR",
    "PIKINE",
    "GUEDIAWAYE",
    "RUFISQUE",
    "THIES",
    "TOUBA",
    "MBOUR",
    "SAINT-LOUIS",
    "KAOLACK",
    "ZIGUINCHOR",
    "DIOURBEL",
    "LOUGA",
    "TAMBACOUNDA",
    "KOLDA",
    "FATICK",
    "MATAM",
    "KAFFRINE",
    "SEDHIOU",
    "KEDOUGOU",
]
ZONE_WEIGHTS: list[int] = [25, 12, 8, 6, 8, 7, 5, 4, 4, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1]

# KYC tier mix for customer wallets; agent-side and business wallets are
# always fully verified.
_CUSTOMER_KYC_WEIGHTS = (25, 55, 20)


def kyc_tier_for(account_type: str, rng: random.Random) -> str:
    if account_type != "customer":
        return "full"
    return rng.choices(KYC_TIERS, weights=_CUSTOMER_KYC_WEIGHTS, k=1)[0]


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------

TS_FMT = "%Y-%m-%dT%H:%M:%S"

# Relative likelihood of a transaction landing in each hour of the day.
# Mobile-money traffic is diurnal: quiet overnight, peaks around midday and
# early evening.
HOUR_WEIGHTS: list[int] = [
    1, 1, 1, 1, 2, 4, 8, 14, 18, 20, 22, 24,  # 00h-11h
    22, 20, 20, 22, 24, 26, 24, 18, 12, 8, 4, 2,  # 12h-23h
]  # fmt: skip


def parse_date(date_str: str) -> datetime:
    """Parse a ``YYYY-MM-DD`` date string."""
    return datetime.strptime(date_str, "%Y-%m-%d")


def iso_ts(dt: datetime) -> str:
    """Format a datetime as the ISO timestamp used in all output rows."""
    return dt.strftime(TS_FMT)


def random_timestamp(rng: random.Random, start: datetime, days: int) -> datetime:
    """Draw a diurnally weighted timestamp within ``days`` days of ``start``."""
    day = rng.randrange(max(1, days))
    hour = rng.choices(range(24), weights=HOUR_WEIGHTS, k=1)[0]
    return start + timedelta(
        days=day, hours=hour, minutes=rng.randrange(60), seconds=rng.randrange(60)
    )


# ---------------------------------------------------------------------------
# Amounts (XOF integers)
# ---------------------------------------------------------------------------

# XOF has no minor unit; the smallest coin in circulation is 5 FCFA and
# mobile-money amounts are multiples of it.
XOF_GRANULARITY = 5

AMOUNT_FLOOR = 100
AMOUNT_CEILING = 20_000_000


def round_xof(amount: float) -> int:
    """Round to the nearest 5 FCFA, never below the granularity itself."""
    return max(XOF_GRANULARITY, int(XOF_GRANULARITY * round(amount / XOF_GRANULARITY)))


def sample_lognormal_xof(
    rng: random.Random,
    median: float,
    sigma: float,
    lo: int = AMOUNT_FLOOR,
    hi: int = AMOUNT_CEILING,
) -> int:
    """Draw a log-normal XOF amount clipped to ``[lo, hi]``.

    Log-normal marginals give the heavy right tail seen in real payment
    data: most transfers are small, a few are large. Fraud typologies clip
    or centre their draws inside this same support so that no amount band
    is fraud-only.
    """
    amount = rng.lognormvariate(0.0, sigma) * median
    return round_xof(min(max(amount, lo), hi))


# Per-type log-normal (median, sigma) for retail traffic.
AMOUNT_PARAMS: dict[str, tuple[int, float]] = {
    "cash_in": (25_000, 1.0),
    "cash_out": (30_000, 1.0),
    "p2p": (12_000, 1.2),
    "merchant_payment": (8_000, 1.0),
    "airtime": (1_000, 0.8),
    "bill_pay": (15_000, 0.9),
    "bank_to_wallet": (150_000, 1.3),
}

# Business segment: a small share of legitimate traffic (traders, informal
# import/export, payroll) moves large sums. Without it the >1M XOF band
# would be fraud-dominated and a plain amount threshold would separate the
# laundering typologies — the tabular baseline caught exactly that.
# tx_type -> (mixture probability, median, sigma)
BUSINESS_AMOUNT_PARAMS: dict[str, tuple[float, int, float]] = {
    "p2p": (0.03, 1_500_000, 0.9),
    "bank_to_wallet": (0.10, 2_500_000, 0.9),
    "cash_in": (0.02, 900_000, 0.8),
    "cash_out": (0.02, 900_000, 0.8),
    "merchant_payment": (0.02, 400_000, 0.9),
}


def sample_amount(rng: random.Random, tx_type: str) -> int:
    business = BUSINESS_AMOUNT_PARAMS.get(tx_type)
    if business is not None and rng.random() < business[0]:
        return sample_lognormal_xof(rng, business[1], business[2])
    median, sigma = AMOUNT_PARAMS[tx_type]
    return sample_lognormal_xof(rng, median, sigma)


# ---------------------------------------------------------------------------
# Traffic mix, channels, fees, commissions
# ---------------------------------------------------------------------------

TX_TYPE_WEIGHTS: list[int] = [24, 20, 30, 10, 8, 5, 3]  # aligned with TX_TYPES

_CHANNEL_MIX: dict[str, tuple[tuple[str, ...], tuple[int, ...]]] = {
    "cash_in": (("agent_pos", "ussd"), (85, 15)),
    "cash_out": (("agent_pos", "ussd"), (85, 15)),
    "p2p": (("ussd", "app"), (60, 40)),
    "merchant_payment": (("ussd", "app"), (55, 45)),
    "airtime": (("ussd", "app"), (65, 35)),
    "bill_pay": (("app", "ussd", "api"), (50, 40, 10)),
    "bank_to_wallet": (("api", "app"), (55, 45)),
}


def sample_tx_type(rng: random.Random) -> str:
    return rng.choices(TX_TYPES, weights=TX_TYPE_WEIGHTS, k=1)[0]


def channel_for(rng: random.Random, tx_type: str) -> str:
    channels, weights = _CHANNEL_MIX[tx_type]
    return rng.choices(channels, weights=weights, k=1)[0]


def _banded(bands: list[tuple[int, int]], amount: int) -> int:
    for ceiling, value in bands:
        if amount <= ceiling:
            return value
    return bands[-1][1]


_INF = 10**12

# Sender-paid fee bands (XOF), Orange-Money-style tariff ladders.
_P2P_FEE_BANDS = [
    (5_000, 50),
    (25_000, 150),
    (65_000, 350),
    (125_000, 750),
    (300_000, 1_500),
    (1_000_000, 3_000),
    (_INF, 5_000),
]
_CASH_OUT_FEE_BANDS = [
    (5_000, 100),
    (25_000, 250),
    (65_000, 500),
    (125_000, 1_000),
    (300_000, 2_000),
    (1_000_000, 4_000),
    (_INF, 7_500),
]

# Operator -> agent commission bands on cash legs. The flat per-band floor
# is what makes commission farming profitable: splitting one deposit into
# many small ones multiplies the per-transaction floor.
_COMMISSION_BANDS = [
    (5_000, 25),
    (15_000, 100),
    (50_000, 250),
    (150_000, 500),
    (500_000, 1_000),
    (_INF, 2_000),
]


def fee_for(tx_type: str, amount: int) -> int:
    """Sender-paid fee for a transaction (XOF)."""
    if tx_type == "p2p":
        return _banded(_P2P_FEE_BANDS, amount)
    if tx_type == "cash_out":
        return _banded(_CASH_OUT_FEE_BANDS, amount)
    if tx_type == "bill_pay":
        return 100
    return 0  # cash-in, merchant, airtime, bank-to-wallet are free to send


def commission_for(tx_type: str, amount: int) -> int:
    """Operator-paid agent commission on cash legs (XOF)."""
    if tx_type in ("cash_in", "cash_out"):
        return _banded(_COMMISSION_BANDS, amount)
    return 0


# ---------------------------------------------------------------------------
# Descriptions
# ---------------------------------------------------------------------------

# One vocabulary per transaction type, shared by every row — legitimate or
# injected. A disjoint "suspicious" wording pool (or an embedding derived
# from one) would hand the label to any bag-of-words baseline and make the
# graph benchmark worthless.
DESCRIPTIONS_BY_TYPE: dict[str, list[str]] = {
    "cash_in": [
        "dépôt espèces agent",
        "dépôt cash wave",
        "recharge portefeuille",
    ],
    "cash_out": [
        "retrait espèces agent",
        "retrait GAB",
        "retrait cash",
    ],
    "p2p": [
        "transfert mobile money",
        "envoi famille",
        "remboursement ami",
        "paiement loyer",
    ],
    "merchant_payment": [
        "paiement marchand mobile",
        "achat supermarché",
        "paiement restaurant",
    ],
    "airtime": [
        "achat crédit téléphonique",
        "recharge crédit",
    ],
    "bill_pay": [
        "paiement facture Senelec/CIE",
        "paiement facture SDE",
        "abonnement en ligne",
    ],
    "bank_to_wallet": [
        "virement banque vers wallet",
        "dépôt salaire",
        "décaissement microcrédit",
        "règlement fournisseur",
    ],
}

# Flat union, kept for callers that only need "a plausible description".
TRANSACTION_DESCRIPTIONS: list[str] = [d for descs in DESCRIPTIONS_BY_TYPE.values() for d in descs]


def description_for(rng: random.Random, tx_type: str) -> str:
    return rng.choice(DESCRIPTIONS_BY_TYPE[tx_type])
