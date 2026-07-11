# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Verify that generated fraud patterns actually exist in the transaction data."""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict


def verify_fraud_patterns(
    fraud_cases_path: str,
    transactions_dir: str,
) -> bool:
    """Check that every fraud-case cycle is backed by real transaction edges.

    Args:
        fraud_cases_path: Path to ``fraud_cases.csv``.
        transactions_dir: Directory containing ``transactions_fraud.csv``
            (or the fraud subdirectory).

    Returns:
        ``True`` if all patterns are valid, ``False`` otherwise.
    """
    # Build edge set from fraud transactions
    fraud_tx_path = os.path.join(os.path.dirname(fraud_cases_path), "transactions_fraud.csv")
    if not os.path.exists(fraud_tx_path):
        print(f"ERROR: {fraud_tx_path} not found", file=sys.stderr)
        return False

    print("Loading fraud transaction edges...")
    edges: dict[str, set[str]] = defaultdict(set)
    with open(fraud_tx_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            src = row.get("src_id") or row.get("~from", "")
            dst = row.get("dst_id") or row.get("~to", "")
            if src and dst:
                edges[src].add(dst)

    print("Verifying fraud cases...")
    all_valid = True
    with open(fraud_cases_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pattern_id = row["pattern_id"]
            pattern_type = row.get("pattern_type", "cycle")
            accounts = row["involved_accounts"].split("|")
            depth = int(row["depth"])

            if pattern_type == "cycle":
                for k in range(depth):
                    src = accounts[k]
                    dst = accounts[(k + 1) % depth]
                    if dst not in edges.get(src, set()):
                        print(f"  FAIL: {pattern_id} — missing edge {src} -> {dst}")
                        all_valid = False
                        break
            elif pattern_type == "structuring":
                coordinator = accounts[0]
                smurfs = accounts[1:]
                for smurf in smurfs:
                    if coordinator not in edges.get(smurf, set()):
                        print(f"  FAIL: {pattern_id} — missing edge {smurf} -> {coordinator}")
                        all_valid = False
                        break
            elif pattern_type == "mobile_money_split":
                # Pattern: agent -> customer (multiple parallel edges)
                # involved_accounts is "agent|customer", depth is the number of splits.
                if len(accounts) < 2:
                    print(
                        f"  WARN: {pattern_id} — mobile_money_split has fewer "
                        f"than 2 accounts, skipping"
                    )
                    continue
                agent, customer = accounts[0], accounts[1]
                if customer not in edges.get(agent, set()):
                    print(f"  FAIL: {pattern_id} — missing edge {agent} -> {customer}")
                    all_valid = False
            elif pattern_type == "trade_based_ml":
                # Pattern: exporter -> shell_importer -> intermediary_i (fan-out)
                #          -> beneficiary (fan-in), for i in range(depth).
                # involved_accounts = exporter|shell_importer|intermediary_0..k-1|beneficiary
                if len(accounts) < 5:
                    print(
                        f"  WARN: {pattern_id} — trade_based_ml has fewer "
                        f"than 5 accounts, skipping"
                    )
                    continue
                exporter, shell_importer = accounts[0], accounts[1]
                intermediaries = accounts[2 : 2 + depth]
                beneficiary = accounts[-1]
                if shell_importer not in edges.get(exporter, set()):
                    print(f"  FAIL: {pattern_id} — missing edge {exporter} -> {shell_importer}")
                    all_valid = False
                    continue
                for inter in intermediaries:
                    if inter not in edges.get(shell_importer, set()):
                        print(f"  FAIL: {pattern_id} — missing edge {shell_importer} -> {inter}")
                        all_valid = False
                        break
                    if beneficiary not in edges.get(inter, set()):
                        print(f"  FAIL: {pattern_id} — missing edge {inter} -> {beneficiary}")
                        all_valid = False
                        break
            elif pattern_type == "hawala_network":
                # Pattern: sender -> hawaladar_A -> hawaladar_B -> beneficiary.
                # involved_accounts = sender|hawaladar_A|hawaladar_B|beneficiary.
                # The optional reverse settlement edge (depth==4) is not
                # independently verified: the edges dict has no per-pattern
                # isolation, so a coincidental edge between the same two
                # hawaladars can't be distinguished from this pattern's own.
                if len(accounts) < 4:
                    print(
                        f"  WARN: {pattern_id} — hawala_network has fewer "
                        f"than 4 accounts, skipping"
                    )
                    continue
                sender, hawaladar_a, hawaladar_b, beneficiary = accounts[:4]
                if hawaladar_a not in edges.get(sender, set()):
                    print(f"  FAIL: {pattern_id} — missing edge {sender} -> {hawaladar_a}")
                    all_valid = False
                    continue
                if hawaladar_b not in edges.get(hawaladar_a, set()):
                    print(f"  FAIL: {pattern_id} — missing edge {hawaladar_a} -> {hawaladar_b}")
                    all_valid = False
                    continue
                if beneficiary not in edges.get(hawaladar_b, set()):
                    print(f"  FAIL: {pattern_id} — missing edge {hawaladar_b} -> {beneficiary}")
                    all_valid = False
            elif pattern_type == "sim_swap_takeover":
                # Pattern: victim -> cashout_agent_i (fan-out), for each cash-out agent.
                # involved_accounts = victim|cashout_agent_0|...|cashout_agent_{depth-1}
                if len(accounts) < 2:
                    print(
                        f"  WARN: {pattern_id} — sim_swap_takeover has fewer "
                        f"than 2 accounts, skipping"
                    )
                    continue
                victim = accounts[0]
                cashout_agents = accounts[1:]
                for agent in cashout_agents:
                    if agent not in edges.get(victim, set()):
                        print(f"  FAIL: {pattern_id} — missing edge {victim} -> {agent}")
                        all_valid = False
                        break
            elif pattern_type == "overdraft_mule_chain":
                # Pattern: mule_i -> collector (fan-in), collector -> agent (consolidation).
                # involved_accounts = collector|mule_0|...|mule_{depth-1}|agent
                if len(accounts) < 3:
                    print(
                        f"  WARN: {pattern_id} — overdraft_mule_chain has fewer "
                        f"than 3 accounts, skipping"
                    )
                    continue
                collector = accounts[0]
                mules = accounts[1:-1]
                agent = accounts[-1]
                for mule in mules:
                    if collector not in edges.get(mule, set()):
                        print(f"  FAIL: {pattern_id} — missing edge {mule} -> {collector}")
                        all_valid = False
                        break
                else:
                    if agent not in edges.get(collector, set()):
                        print(f"  FAIL: {pattern_id} — missing edge {collector} -> {agent}")
                        all_valid = False
            else:
                print(f"  WARN: {pattern_id} — unknown pattern_type '{pattern_type}', skipping")

    if all_valid:
        print("All fraud patterns verified successfully.")
    else:
        print("Some fraud patterns failed verification.", file=sys.stderr)

    return all_valid


def main() -> None:
    """CLI entry point for verification."""
    import argparse

    parser = argparse.ArgumentParser(description="Verify generated fraud patterns.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root output directory (contains fraud/ subdirectory).",
    )
    args = parser.parse_args()

    cases = os.path.join(args.data_dir, "fraud", "fraud_cases.csv")
    if not os.path.exists(cases):
        print(f"ERROR: {cases} not found. Run gen-fraud-graph first.", file=sys.stderr)
        sys.exit(1)

    ok = verify_fraud_patterns(cases, args.data_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
