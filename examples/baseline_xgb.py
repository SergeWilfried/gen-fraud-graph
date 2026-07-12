# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""XGBoost tabular baseline for a gen-fraud-graph dataset.

This is the honesty check for the benchmark: a competent per-transaction
model with velocity features, trained on a generated dataset. It reads the
generator's output directly — no manual CSV wrangling:

    gen-fraud-graph --scale 0.005 --output ./data
    python examples/baseline_xgb.py --data-dir ./data

Labels are derived from provenance (rows in ``fraud/transactions_fraud.csv``
are fraud, rows in ``transactions/`` are not); nothing in the feature matrix
encodes the label directly. If a trivial feature (amount alone, an hour flag)
tops the importance list with a near-perfect score, the generator has a
leak — that is exactly what this script exists to catch.

Evaluation follows fraud-ops practice:

* temporal split — train on the past, test on the future;
* PR-AUC and recall at fixed false-positive rates (an operator reviews a
  bounded alert queue, so recall@FPR is the number that matters);
* per-typology pattern recall — a pattern counts as caught if at least one
  of its test-window edges is flagged at the alert threshold.

Requires the ``baseline`` extra: ``pip install 'gen-fraud-graph[baseline]'``.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_curve
from xgboost import XGBClassifier

from gen_fraud_graph.schema import TIER_TX_CAPS

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
# Features — behaviour, velocity, and the typed schema fields
# ---------------------------------------------------------------------------


def build_features(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    ts = df["timestamp"]

    # --- per-row facts ---
    f["log_amount"] = np.log1p(df["amount"])
    f["hour"] = ts.dt.hour
    f["day_of_week"] = ts.dt.dayofweek
    f["is_night"] = ((f["hour"] >= 22) | (f["hour"] <= 5)).astype(int)
    # Every generated amount is a multiple of 5 XOF, so the classic
    # "round amount" tell only means something at note-sized multiples.
    f["is_round_amount"] = (df["amount"] % 25_000 == 0).astype(int)
    f["fee"] = df["fee"]
    f["commission"] = df["commission"]
    f["fee_rate"] = df["fee"] / df["amount"]
    f["commission_rate"] = df["commission"] / df["amount"]

    # How close does this transfer sail to the sender's KYC-tier ceiling?
    # Structuring lives just under these caps.
    caps = df["src_kyc"].map(TIER_TX_CAPS).fillna(max(TIER_TX_CAPS.values()))
    f["amount_vs_tier_cap"] = df["amount"] / caps

    # --- categorical one-hots ---
    for col in ("tx_type", "channel", "src_type", "dst_type", "src_kyc"):
        f = pd.concat([f, pd.get_dummies(df[col], prefix=col)], axis=1)

    # --- velocity: backward-looking only (no future information) ---
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

    # Seconds since the sender's most recent SIM swap (if any) — the
    # takeover signal, available by joining fraud/sim_events.csv.
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

    f = f.astype(float)
    print(f"Built {f.shape[1]} features")
    return f


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
    print(
        f"Trained on {sl_train.stop:,} rows ({n_fraud:,} fraud); "
        f"{model.best_iteration + 1} trees"
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


def evaluate(name: str, y_true: pd.Series, y_score: np.ndarray) -> None:
    pr_auc = average_precision_score(y_true, y_score)
    print(f"\n--- {name} ---")
    print(f"PR-AUC:                  {pr_auc:.3f}  (random ~{y_true.mean():.4f})")
    print(f"Fraud caught @ 1%   FPR: {recall_at_fpr(y_true, y_score, 0.01):.1%}")
    print(f"Fraud caught @ 0.1% FPR: {recall_at_fpr(y_true, y_score, 0.001):.1%}")


def pattern_recall(
    df: pd.DataFrame,
    cases: pd.DataFrame,
    sl_test: slice,
    scores: np.ndarray,
    threshold: float,
) -> None:
    """Per-typology recall: a pattern is caught if any of its edges inside
    the test slice is flagged at the alert threshold."""
    test = df.iloc[sl_test].copy()
    test["flagged"] = scores >= threshold
    flagged = test[test["flagged"] & (test["is_fraud"] == 1)]
    test_fraud = test[test["is_fraud"] == 1]
    test_start = test["timestamp"].min()

    print("\n--- Per-typology pattern recall (test slice, alert threshold from validation) ---")
    print(f"{'pattern_type':<22} {'caught':>6} / {'in-test':>7}")
    for ptype, group in cases.groupby("pattern_type"):
        caught = evaluable = 0
        for _, case in group.iterrows():
            w_start = pd.Timestamp(case["window_start"])
            w_end = pd.Timestamp(case["window_end"])
            if w_end < test_start:
                continue  # pattern finished before the test window
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
            print(f"{ptype:<22} {caught:>6} / {evaluable:>7}  ({caught / evaluable:.0%})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", required=True, help="gen-fraud-graph output directory")
    ap.add_argument("--model-out", default="", help="optional path to save the model (json)")
    args = ap.parse_args()

    df, cases, events = load_dataset(args.data_dir)
    features = build_features(df, events)
    y = df["is_fraud"]

    sl_train, sl_valid, sl_test = temporal_split(len(df))
    model = train(features, y, sl_train, sl_valid)

    valid_scores = model.predict_proba(features.iloc[sl_valid])[:, 1]
    test_scores = model.predict_proba(features.iloc[sl_test])[:, 1]

    evaluate("VALIDATION (seen during early stopping)", y.iloc[sl_valid], valid_scores)
    evaluate("TEST (future data, never seen)", y.iloc[sl_test], test_scores)

    threshold = threshold_at_fpr(y.iloc[sl_valid], valid_scores, 0.01)
    pattern_recall(df, cases, sl_test, test_scores, threshold)

    imp = pd.Series(model.feature_importances_, index=features.columns).sort_values(ascending=False)
    print("\nTop 15 features by importance (sanity-check against fraud intuition):")
    for name, val in imp.head(15).items():
        print(f"  {name:38s} {val:.3f}")

    if args.model_out:
        model.save_model(args.model_out)
        print(f"\nModel saved to {args.model_out}")


if __name__ == "__main__":
    main()
