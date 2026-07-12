# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""XGBoost baseline ladder for a gen-fraud-graph dataset.

The benchmark's thesis is that MoMo fraud is detectable through graph
structure + velocity + the typed schema fields, and never through any
single column. This script demonstrates it as a ladder of models on the
same data, one per cumulative feature tier:

* ``amounts``  — bank-style columns only (amount, time of day)
* ``velocity`` — + backward-looking per-wallet and per-agent history
* ``schema``   — + the MoMo fields (tx_type, channel, fees, commissions,
  account roles, KYC caps, SIM-swap events)
* ``graph``    — + edge-level topology (degrees, reciprocity, directed
  3-/4-cycle counts through the edge, PageRank)

The gaps between rungs are the benchmark's selling point; the per-typology
recall table shows which typology needs which signal. It also doubles as
the leak detector: a bottom-rung tier scoring near-perfect means the
generator leaks the label through a trivial column.

Labels are derived from provenance (rows in ``fraud/transactions_fraud.csv``
are fraud, rows in ``transactions/`` are not); nothing in the feature matrix
encodes the label directly. History features look strictly backward in time.
Graph features are transductive: they use the (label-free) topology of the
full dataset, the standard setting for graph fraud benchmarks.

Usage::

    gen-fraud-graph --scale 0.005 --output ./data
    python examples/baseline_xgb.py --data-dir ./data               # full ladder
    python examples/baseline_xgb.py --data-dir ./data --tier graph  # one tier, detailed

Requires the ``baseline`` extra: ``pip install 'gen-fraud-graph[baseline]'``.
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import average_precision_score, roc_curve
from xgboost import XGBClassifier

from gen_fraud_graph.schema import TIER_TX_CAPS

TIERS = ["amounts", "velocity", "schema", "graph"]

TRAIN_FRACTION = 0.70
VALID_FRACTION = 0.15  # remainder is the test slice

TX_COLUMNS = [
    "tx_id",
    "src_id",
    "dst_id",
    "tx_type",
    "channel",
    "agent_id",
    "amount",
    "fee",
    "commission",
    "timestamp",
]

NEW_WALLET_GAP = 86_400 * 30  # "never seen before" sentinel, seconds
NO_SWAP = 1e9  # "no SIM swap on record" sentinel, seconds


# ---------------------------------------------------------------------------
# Loading — label by provenance, join wallet attributes and SIM events
# ---------------------------------------------------------------------------


def load_dataset(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (transactions with labels and wallet joins, fraud_cases, sim_events)."""
    normal_files = sorted(glob.glob(os.path.join(data_dir, "transactions", "*.csv")))
    fraud_file = os.path.join(data_dir, "fraud", "transactions_fraud.csv")
    if not normal_files or not os.path.exists(fraud_file):
        raise SystemExit(f"no generated dataset under {data_dir!r} — run gen-fraud-graph first")

    normal = pd.concat(
        (pd.read_csv(f, usecols=TX_COLUMNS) for f in normal_files), ignore_index=True
    )
    fraud = pd.read_csv(fraud_file, usecols=TX_COLUMNS)
    normal["is_fraud"] = 0
    fraud["is_fraud"] = 1

    df = pd.concat([normal, fraud], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Temporal split needs a strict timeline; tx_id breaks timestamp ties
    # deterministically.
    df = df.sort_values(["timestamp", "tx_id"]).reset_index(drop=True)
    df["agent_id"] = df["agent_id"].fillna("")

    accounts = pd.concat(
        (
            pd.read_csv(f, usecols=["account_id", "account_type", "kyc_tier"])
            for f in sorted(glob.glob(os.path.join(data_dir, "accounts", "*.csv")))
        ),
        ignore_index=True,
    ).set_index("account_id")

    for side in ("src", "dst"):
        df[f"{side}_type"] = df[f"{side}_id"].map(accounts["account_type"]).fillna("unknown")
    df["src_kyc"] = df["src_id"].map(accounts["kyc_tier"]).fillna("full")

    cases = pd.read_csv(os.path.join(data_dir, "fraud", "fraud_cases.csv"))
    events_path = os.path.join(data_dir, "fraud", "sim_events.csv")
    events = (
        pd.read_csv(events_path, parse_dates=["swap_ts"])
        if os.path.exists(events_path)
        else pd.DataFrame(columns=["account_id", "swap_ts"])
    )

    print(
        f"Loaded {len(df):,} transactions ({df['is_fraud'].mean():.3%} fraud), "
        f"{len(cases):,} fraud cases, {len(events):,} SIM events"
    )
    return df, cases, events


# ---------------------------------------------------------------------------
# Feature tiers (cumulative)
# ---------------------------------------------------------------------------


def add_amount_features(f: pd.DataFrame, df: pd.DataFrame) -> None:
    """Tier 1 — what a bank-style schema exposes: amount and time."""
    ts = df["timestamp"]
    f["log_amount"] = np.log1p(df["amount"])
    f["hour"] = ts.dt.hour
    f["day_of_week"] = ts.dt.dayofweek
    f["is_night"] = ((f["hour"] >= 22) | (f["hour"] <= 5)).astype(int)
    # Every generated amount is a multiple of 5 XOF, so the classic
    # "round amount" tell only means something at note-sized multiples.
    f["is_round_amount"] = (df["amount"] % 25_000 == 0).astype(int)


def add_velocity_features(f: pd.DataFrame, df: pd.DataFrame) -> None:
    """Tier 2 — backward-looking wallet and agent history (no future info)."""
    for role, col in (("sender", "src_id"), ("receiver", "dst_id")):
        grp = df.groupby(col)
        count_so_far = grp.cumcount()
        f[f"{role}_tx_count_so_far"] = count_so_far

        cum_avg = (grp["amount"].cumsum() - df["amount"]) / count_so_far.replace(0, np.nan)
        f[f"{role}_amount_vs_own_avg"] = (df["amount"] / cum_avg).fillna(1.0)

        gap = grp["timestamp"].diff().dt.total_seconds()
        f[f"{role}_seconds_since_prev"] = gap.fillna(NEW_WALLET_GAP)

    # Fan-out: distinct counterparties the sender has paid so far.
    seen = df.groupby("src_id")["dst_id"].transform(lambda s: (~s.duplicated()).cumsum() - 1)
    f["sender_distinct_receivers_so_far"] = seen

    # Agent-dimension velocity: commission farming and takeover cash-outs
    # are bursts *at the agent*, whichever customers are involved.
    with_agent = df["agent_id"] != ""
    agent_grp = df[with_agent].groupby("agent_id")
    f["agent_tx_count_so_far"] = agent_grp.cumcount().reindex(df.index).fillna(0)
    agent_gap = agent_grp["timestamp"].diff().dt.total_seconds()
    f["agent_seconds_since_prev"] = agent_gap.reindex(df.index).fillna(NEW_WALLET_GAP)


def add_schema_features(f: pd.DataFrame, df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Tier 3 — the MoMo schema: types, tariffs, roles, KYC caps, SIM events."""
    f["fee"] = df["fee"]
    f["commission"] = df["commission"]
    f["fee_rate"] = df["fee"] / df["amount"]
    f["commission_rate"] = df["commission"] / df["amount"]

    # How close does this transfer sail to the sender's KYC-tier ceiling?
    # Structuring lives just under these caps.
    caps = df["src_kyc"].map(TIER_TX_CAPS).fillna(max(TIER_TX_CAPS.values()))
    f["amount_vs_tier_cap"] = df["amount"] / caps

    for col in ("tx_type", "channel", "src_type", "dst_type", "src_kyc"):
        f = pd.concat([f, pd.get_dummies(df[col], prefix=col)], axis=1)

    # Seconds since the sender's most recent SIM swap (if any). Most events
    # are benign upgrades, so this only works combined with burst features.
    if len(events):
        ev = events[["account_id", "swap_ts"]].sort_values("swap_ts")
        joined = pd.merge_asof(
            df[["timestamp", "src_id"]].reset_index().sort_values("timestamp"),
            ev,
            left_on="timestamp",
            right_on="swap_ts",
            left_by="src_id",
            right_by="account_id",
            direction="backward",
        ).set_index("index")
        delta = (joined["timestamp"] - joined["swap_ts"]).dt.total_seconds()
        f["seconds_since_sim_swap"] = delta.reindex(df.index).fillna(NO_SWAP)
    else:
        f["seconds_since_sim_swap"] = NO_SWAP
    return f


def add_graph_features(f: pd.DataFrame, df: pd.DataFrame) -> None:
    """Tier 4 — edge-level topology of the (label-free) transaction graph.

    Laundering rings are directed cycles: the count of short directed
    cycles closing through an edge, and how connected its endpoints are,
    are signals no per-row model can see.
    """
    out_sets: dict[str, set[str]] = defaultdict(set)
    in_sets: dict[str, set[str]] = defaultdict(set)
    pairs = df[["src_id", "dst_id"]].drop_duplicates()
    for u, v in pairs.itertuples(index=False):
        out_sets[u].add(v)
        in_sets[v].add(u)

    f["src_out_degree"] = df["src_id"].map(lambda a: len(out_sets[a])).astype(float)
    f["src_in_degree"] = df["src_id"].map(lambda a: len(in_sets[a])).astype(float)
    f["dst_out_degree"] = df["dst_id"].map(lambda a: len(out_sets[a])).astype(float)
    f["dst_in_degree"] = df["dst_id"].map(lambda a: len(in_sets[a])).astype(float)

    # Cycle participation through this edge (u -> v):
    #   reciprocal: v pays u back directly (2-cycle)
    #   cycles3:    v pays someone who pays u
    #   cycles4:    value returns to u two hops after v
    two_hop_out: dict[str, set[str]] = {}

    def out2(node: str) -> set[str]:
        cached = two_hop_out.get(node)
        if cached is None:
            cached = set()
            for w in out_sets[node]:
                cached |= out_sets[w]
            two_hop_out[node] = cached
        return cached

    recip, cyc3, cyc4, common = [], [], [], []
    for u, v in df[["src_id", "dst_id"]].itertuples(index=False):
        recip.append(1.0 if u in out_sets[v] else 0.0)
        cyc3.append(float(len(out_sets[v] & in_sets[u])))
        cyc4.append(float(len(out2(v) & in_sets[u])))
        common.append(float(len((out_sets[u] | in_sets[u]) & (out_sets[v] | in_sets[v]))))
    f["pair_reciprocal"] = recip
    f["cycles3_through_edge"] = cyc3
    f["cycles4_through_edge"] = cyc4
    f["common_neighbours"] = common

    # PageRank by power iteration on the unique-pair adjacency.
    nodes = sorted(out_sets.keys() | in_sets.keys())
    idx = {a: i for i, a in enumerate(nodes)}
    n = len(nodes)
    rows = [idx[u] for u, v in pairs.itertuples(index=False)]
    cols = [idx[v] for u, v in pairs.itertuples(index=False)]
    adj = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    out_deg = np.asarray(adj.sum(axis=1)).ravel()
    transition = csr_matrix(adj.multiply(1.0 / np.maximum(out_deg, 1.0)[:, None]))
    rank = np.full(n, 1.0 / n)
    for _ in range(30):
        rank = 0.15 / n + 0.85 * (transition.T @ rank)
    pagerank = dict(zip(nodes, rank * n, strict=True))
    f["src_pagerank"] = df["src_id"].map(pagerank).astype(float)
    f["dst_pagerank"] = df["dst_id"].map(pagerank).astype(float)


def build_features(df: pd.DataFrame, events: pd.DataFrame, tier: str) -> pd.DataFrame:
    """Build the cumulative feature matrix up to ``tier``."""
    level = TIERS.index(tier)
    f = pd.DataFrame(index=df.index)
    add_amount_features(f, df)
    if level >= 1:
        add_velocity_features(f, df)
    if level >= 2:
        f = add_schema_features(f, df, events)
    if level >= 3:
        add_graph_features(f, df)
    return f.astype(float)


# ---------------------------------------------------------------------------
# Split / train / evaluate
# ---------------------------------------------------------------------------


def temporal_split(n_rows: int) -> tuple[slice, slice, slice]:
    i_train = int(n_rows * TRAIN_FRACTION)
    i_valid = int(n_rows * (TRAIN_FRACTION + VALID_FRACTION))
    return slice(0, i_train), slice(i_train, i_valid), slice(i_valid, n_rows)


def train(x: pd.DataFrame, y: pd.Series, sl_train: slice, sl_valid: slice) -> XGBClassifier:
    n_fraud = int(y.iloc[sl_train].sum())
    weight = (len(y.iloc[sl_train]) - n_fraud) / max(n_fraud, 1)

    model = XGBClassifier(
        n_estimators=2000,
        early_stopping_rounds=50,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=weight,
        eval_metric="aucpr",
        n_jobs=-1,
    )
    model.fit(
        x.iloc[sl_train],
        y.iloc[sl_train],
        eval_set=[(x.iloc[sl_valid], y.iloc[sl_valid])],
        verbose=False,
    )
    return model


def recall_at_fpr(y_true: pd.Series, y_score: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.interp(target_fpr, fpr, tpr))


def threshold_at_fpr(y_true: pd.Series, y_score: np.ndarray, target_fpr: float) -> float:
    """Alert threshold calibrated on validation: the score above which the
    false-positive rate stays within ``target_fpr``."""
    fpr, _, thresholds = roc_curve(y_true, y_score)
    ok = np.where(fpr <= target_fpr)[0]
    return float(thresholds[ok[-1]]) if len(ok) else float("inf")


def pattern_recall(
    df: pd.DataFrame,
    cases: pd.DataFrame,
    sl_test: slice,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, tuple[int, int]]:
    """Per-typology recall: a pattern is caught if any of its edges inside
    the test slice is flagged at the alert threshold."""
    test = df.iloc[sl_test].copy()
    test["flagged"] = scores >= threshold
    flagged = test[test["flagged"] & (test["is_fraud"] == 1)]
    test_fraud = test[test["is_fraud"] == 1]

    result: dict[str, tuple[int, int]] = {}
    for ptype, group in cases.groupby("pattern_type"):
        caught = evaluable = 0
        for _, case in group.iterrows():
            w_start = pd.Timestamp(case["window_start"])
            w_end = pd.Timestamp(case["window_end"])
            involved = set(case["involved_accounts"].split("|"))

            def in_pattern(rows: pd.DataFrame, acc: set = involved, lo=w_start, hi=w_end) -> bool:
                return not rows[
                    rows["src_id"].isin(acc)
                    & rows["dst_id"].isin(acc)
                    & (rows["timestamp"] >= lo)
                    & (rows["timestamp"] <= hi)
                ].empty

            if not in_pattern(test_fraud):
                continue  # none of its edges landed in the test slice
            evaluable += 1
            if in_pattern(flagged):
                caught += 1
        if evaluable:
            result[str(ptype)] = (caught, evaluable)
    return result


def run_tier(
    tier: str,
    df: pd.DataFrame,
    cases: pd.DataFrame,
    events: pd.DataFrame,
    verbose: bool = False,
) -> dict:
    """Train and evaluate one cumulative feature tier."""
    features = build_features(df, events, tier)
    y = df["is_fraud"]
    sl_train, sl_valid, sl_test = temporal_split(len(df))

    model = train(features, y, sl_train, sl_valid)
    valid_scores = model.predict_proba(features.iloc[sl_valid])[:, 1]
    test_scores = model.predict_proba(features.iloc[sl_test])[:, 1]

    threshold = threshold_at_fpr(y.iloc[sl_valid], valid_scores, 0.01)
    result = {
        "tier": tier,
        "n_features": features.shape[1],
        "pr_auc": average_precision_score(y.iloc[sl_test], test_scores),
        "recall_1pct": recall_at_fpr(y.iloc[sl_test], test_scores, 0.01),
        "recall_01pct": recall_at_fpr(y.iloc[sl_test], test_scores, 0.001),
        "patterns": pattern_recall(df, cases, sl_test, test_scores, threshold),
    }

    if verbose:
        print(f"\n--- Tier '{tier}' ({result['n_features']} features) — test slice ---")
        print(f"PR-AUC:                  {result['pr_auc']:.3f}")
        print(f"Fraud caught @ 1%   FPR: {result['recall_1pct']:.1%}")
        print(f"Fraud caught @ 0.1% FPR: {result['recall_01pct']:.1%}")
        print("\nPer-typology pattern recall (alert threshold from validation):")
        for ptype, (caught, total) in sorted(result["patterns"].items()):
            print(f"  {ptype:<22} {caught:>3} / {total:<3} ({caught / total:.0%})")
        imp = pd.Series(model.feature_importances_, index=features.columns).sort_values(
            ascending=False
        )
        print("\nTop 15 features by importance:")
        for name, val in imp.head(15).items():
            print(f"  {name:38s} {val:.3f}")
    return result


def print_ladder(results: list[dict]) -> None:
    typologies = sorted({p for r in results for p in r["patterns"]})
    print("\n=== Baseline ladder (test slice) ===")
    header = f"{'tier':<10} {'feats':>5} {'PR-AUC':>7} {'R@1%':>6} {'R@0.1%':>7}"
    for t in typologies:
        header += f"  {t[:9]:>9}"
    print(header)
    for r in results:
        line = (
            f"{r['tier']:<10} {r['n_features']:>5} {r['pr_auc']:>7.3f} "
            f"{r['recall_1pct']:>6.1%} {r['recall_01pct']:>7.1%}"
        )
        for t in typologies:
            caught, total = r["patterns"].get(t, (0, 0))
            line += f"  {f'{caught}/{total}':>9}"
        print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", required=True, help="gen-fraud-graph output directory")
    ap.add_argument(
        "--tier",
        choices=TIERS,
        default=None,
        help="run a single cumulative tier with detailed output (default: full ladder)",
    )
    args = ap.parse_args()

    df, cases, events = load_dataset(args.data_dir)

    if args.tier:
        run_tier(args.tier, df, cases, events, verbose=True)
    else:
        results = [run_tier(tier, df, cases, events) for tier in TIERS]
        print_ladder(results)


if __name__ == "__main__":
    main()
