# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Shared schema definitions: vocabularies, amount and timestamp sampling.

Everything that must be consistent between the normal-transaction workers
and the fraud typology generators lives here. Fraud rows draw amounts,
timestamps, and descriptions from the same machinery as legitimate rows so
that no single column separates the classes — injected patterns are meant
to be detectable through graph structure and velocity, not through a
vocabulary or constant-value leak.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------

TS_FMT = "%Y-%m-%dT%H:%M:%S"

# Relative likelihood of a transaction landing in each hour of the day.
# Mobile-money traffic is diurnal: quiet overnight, peaks around midday and
# early evening.
HOUR_WEIGHTS: list[int] = [
    1,
    1,
    1,
    1,
    2,
    4,
    8,
    14,
    18,
    20,
    22,
    24,  # 00h–11h
    22,
    20,
    20,
    22,
    24,
    26,
    24,
    18,
    12,
    8,
    4,
    2,  # 12h–23h
]


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


# ---------------------------------------------------------------------------
# Descriptions
# ---------------------------------------------------------------------------

# One vocabulary for every transaction row, legitimate or injected. Keeping
# the pools identical is deliberate: a disjoint "suspicious" vocabulary (or
# an embedding derived from one) would hand the label to any bag-of-words
# baseline and make the graph benchmark worthless.
TRANSACTION_DESCRIPTIONS: list[str] = [
    "achat supermarché",
    "dépôt salaire",
    "paiement facture Senelec/CIE",
    "abonnement en ligne",
    "paiement restaurant",
    "retrait GAB",
    "transfert mobile money",
    "prime assurance",
    "paiement loyer",
    "dépôt investissement",
    "envoi famille",
    "remboursement ami",
    "dépôt espèces agent",
    "retrait espèces agent",
    "achat crédit téléphonique",
    "paiement marchand mobile",
    "règlement fournisseur",
    "virement banque vers wallet",
]

# ---------------------------------------------------------------------------
# Fraud case ground-truth schema
# ---------------------------------------------------------------------------

FRAUD_CASE_HEADERS: list[str] = [
    "pattern_id",
    "start_acc_id",
    "pattern_type",
    "depth",
    "involved_accounts",
    "window_start",
    "window_end",
]
