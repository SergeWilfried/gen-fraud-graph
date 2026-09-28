# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Render fraud typologies from a generated dataset as static images.

Each figure shows one injected pattern **embedded in its 1-hop
neighbourhood of legitimate traffic** — the point is not the textbook
shape but the needle-in-haystack property: fraud edges (red) thread
through ordinary transactions (grey) and only structure, velocity, and
the typed fields give them away.

Everything is drawn from the generator's own output: ``fraud_cases.csv``
supplies the pattern (roles by position, time window), the transaction
CSVs supply the background, ``accounts`` colour the nodes by role, and
``sim_events.csv`` annotates the SIM swap that precedes a takeover burst.

Usage::

    gen-fraud-graph --preset momo-100k --output ./momo-100k
    python examples/plot_typologies.py --data-dir ./momo-100k --out docs/images

Requires the ``viz`` extra: ``pip install 'gen-fraud-graph[viz]'``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import random
from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

from gen_fraud_graph.schema import TS_FMT, account_type_for

# Node colours by account role (red is reserved for fraud edges).
ROLE_COLOURS = {
    "customer": "#93a1c4",
    "agent": "#f28e2b",
    "super_agent": "#edc948",
    "merchant": "#59a14f",
    "aggregator": "#b07aa1",
}
NORMAL_EDGE = "#c4c9d4"
FRAUD_EDGE = "#d62728"

# Which typologies to render and how to pick a photogenic instance:
# (pattern_type, preferred depth, filename stem, title)
SHOWCASE = [
    ("cycle", 6, "typology_cycle", "Money-laundering cycle"),
    ("structuring", 9, "typology_structuring", "Structuring (smurfing) fan-in"),
    ("sim_swap_takeover", 6, "typology_sim_swap", "SIM-swap takeover cash-out burst"),
    ("overdraft_mule_chain", 10, "typology_mule_chain", "Micro-loan mule chain"),
]

MAX_BACKGROUND_NEIGHBOURS = 7  # per pattern node, keeps figures readable


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------


def read_rows(path: str) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def pick_case(cases: list[dict], pattern_type: str, preferred_depth: int) -> dict:
    """Deterministically pick the showcased instance of a typology."""
    candidates = [c for c in cases if c["pattern_type"] == pattern_type]
    if not candidates:
        raise SystemExit(f"dataset has no {pattern_type!r} patterns")
    return min(
        candidates,
        key=lambda c: (abs(int(c["depth"]) - preferred_depth), c["pattern_id"]),
    )


def load_fraud_edges(data_dir: str, case: dict) -> list[dict]:
    """The case's own transactions: endpoints in the pattern, inside its window."""
    involved = set(case["involved_accounts"].split("|"))
    w_start = datetime.strptime(case["window_start"], TS_FMT)
    w_end = datetime.strptime(case["window_end"], TS_FMT)
    edges = []
    for row in read_rows(os.path.join(data_dir, "fraud", "transactions_fraud.csv")):
        if row["src_id"] in involved and row["dst_id"] in involved:
            ts = datetime.strptime(row["timestamp"], TS_FMT)
            if w_start <= ts <= w_end:
                row["_ts"] = ts
                edges.append(row)
    return edges


def load_background(data_dir: str, node_union: set[str]) -> list[dict]:
    """Legitimate transactions touching any showcased pattern node."""
    edges = []
    for path in sorted(glob.glob(os.path.join(data_dir, "transactions", "*.csv"))):
        with open(path) as fh:
            for row in csv.DictReader(fh):
                if row["src_id"] in node_union or row["dst_id"] in node_union:
                    edges.append({k: row[k] for k in ("src_id", "dst_id", "tx_type", "amount")})
    return edges


def latest_swap_before(data_dir: str, account_id: str, before: datetime) -> dict | None:
    path = os.path.join(data_dir, "fraud", "sim_events.csv")
    if not os.path.exists(path):
        return None
    best = None
    for ev in read_rows(path):
        if ev["account_id"] != account_id:
            continue
        ts = datetime.strptime(ev["swap_ts"], TS_FMT)
        if ts < before and (best is None or ts > best["_ts"]):
            ev["_ts"] = ts
            best = ev
    return best


# ---------------------------------------------------------------------------
# Layout — pattern nodes are pinned per typology, background springs around
# ---------------------------------------------------------------------------


PATTERN_SCALE = 2.3  # how much canvas the pinned pattern claims vs the background


def arc_positions(n: int, x: float, spread: float = 4.6) -> list[tuple[float, float]]:
    if n == 1:
        return [(x, 0.0)]
    return [(x, spread * (i / (n - 1) - 0.5)) for i in range(n)]


def pattern_positions(pattern_type: str, accounts: list[str]) -> dict[str, tuple[float, float]]:
    n = len(accounts)
    k = PATTERN_SCALE
    if pattern_type == "cycle":
        return {
            acc: (
                1.7 * k * math.cos(2 * math.pi * i / n) / 2.3,
                1.7 * k * math.sin(2 * math.pi * i / n) / 2.3,
            )
            for i, acc in enumerate(accounts)
        }
    if pattern_type == "structuring":  # coordinator | smurfs...
        pos = {accounts[0]: (1.1 * k, 0.0)}
        pos.update(zip(accounts[1:], arc_positions(n - 1, -1.1 * k), strict=True))
        return pos
    if pattern_type == "sim_swap_takeover":  # victim | agents...
        pos = {accounts[0]: (-1.1 * k, 0.0)}
        pos.update(zip(accounts[1:], arc_positions(n - 1, 1.1 * k), strict=True))
        return pos
    if pattern_type == "overdraft_mule_chain":  # collector | mules... | agent
        pos = {accounts[0]: (0.5 * k, 0.0), accounts[-1]: (1.6 * k, 0.0)}
        pos.update(zip(accounts[1:-1], arc_positions(n - 2, -1.1 * k), strict=True))
        return pos
    return {acc: p for acc, p in zip(accounts, arc_positions(n, 0.0), strict=True)}


def pattern_labels(pattern_type: str, accounts: list[str]) -> dict[str, str]:
    if pattern_type == "cycle":
        return {acc: f"W{i + 1}" for i, acc in enumerate(accounts)}
    if pattern_type == "structuring":
        return {accounts[0]: "C", **{a: f"S{i + 1}" for i, a in enumerate(accounts[1:])}}
    if pattern_type == "sim_swap_takeover":
        return {accounts[0]: "V", **{a: f"A{i + 1}" for i, a in enumerate(accounts[1:])}}
    if pattern_type == "overdraft_mule_chain":
        labels = {accounts[0]: "C", accounts[-1]: "A"}
        labels.update({a: f"M{i + 1}" for i, a in enumerate(accounts[1:-1])})
        return labels
    return {a: str(i + 1) for i, a in enumerate(accounts)}


def offset_label(seconds: float) -> str:
    if seconds < 90:
        return f"+{seconds:.0f}s"
    if seconds < 7_200:
        return f"+{seconds / 60:.0f}m"
    if seconds < 172_800:
        return f"+{seconds / 3_600:.1f}h"
    return f"+{seconds / 86_400:.1f}d"


def window_label(case: dict) -> str:
    start = datetime.strptime(case["window_start"], TS_FMT)
    end = datetime.strptime(case["window_end"], TS_FMT)
    seconds = (end - start).total_seconds()
    if seconds < 3_600:
        return f"{seconds / 60:.0f} minutes"
    if seconds < 172_800:
        return f"{seconds / 3_600:.1f} hours"
    return f"{seconds / 86_400:.1f} days"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_case(
    case: dict,
    title: str,
    fraud_edges: list[dict],
    background: list[dict],
    out_path: str,
    layout_seed: int,
    caption: str,
    annotation: str | None = None,
) -> None:
    accounts = case["involved_accounts"].split("|")
    pattern_nodes = set(accounts)
    rng = random.Random(layout_seed)

    graph = nx.MultiDiGraph()
    graph.add_nodes_from(accounts)
    for e in fraud_edges:
        graph.add_edge(e["src_id"], e["dst_id"], fraud=True, amount=int(e["amount"]))

    # Background: a deterministic handful of legitimate counterparties per
    # pattern node — enough haystack to make the point, not a hairball.
    per_node: dict[str, list[dict]] = {n: [] for n in pattern_nodes}
    for e in background:
        for endpoint in (e["src_id"], e["dst_id"]):
            if endpoint in per_node:
                per_node[endpoint].append(e)
    for _node, edges in per_node.items():
        edges.sort(key=lambda e: (e["src_id"], e["dst_id"], e["amount"]))
        for e in rng.sample(edges, min(MAX_BACKGROUND_NEIGHBOURS, len(edges))):
            graph.add_edge(e["src_id"], e["dst_id"], fraud=False, amount=int(e["amount"]))

    pinned = pattern_positions(case["pattern_type"], accounts)
    pos = nx.spring_layout(graph, pos=pinned, fixed=list(pattern_nodes), seed=layout_seed, k=0.5)
    # Clamp the background halo to a fixed radius so outliers can't stretch
    # the axes and shrink the pattern into a corner of the canvas.
    max_r = 1.9 * PATTERN_SCALE
    for node, (x, y) in pos.items():
        if node not in pattern_nodes:
            r = math.hypot(x, y)
            if r > max_r:
                pos[node] = (x * max_r / r, y * max_r / r)

    fig, ax = plt.subplots(figsize=(10, 7.2), dpi=200)
    ax.set_axis_off()

    def node_colour(node: str) -> str:
        uid = int(node.removeprefix("acc_"))
        return ROLE_COLOURS[account_type_for(uid)]

    background_nodes = [n for n in graph.nodes if n not in pattern_nodes]
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=background_nodes,
        node_size=70,
        node_color=[node_colour(n) for n in background_nodes],
        alpha=0.45,
        linewidths=0,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=accounts,
        node_size=430,
        node_color=[node_colour(n) for n in accounts],
        edgecolors="#20222a",
        linewidths=1.5,
        ax=ax,
    )

    normal = [(u, v) for u, v, d in graph.edges(data=True) if not d["fraud"]]
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=normal,
        edge_color=NORMAL_EDGE,
        width=0.8,
        alpha=0.6,
        arrows=False,
        ax=ax,
    )
    fraud = [(u, v, d) for u, v, d in graph.edges(data=True) if d["fraud"]]
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=[(u, v) for u, v, _ in fraud],
        edge_color=FRAUD_EDGE,
        width=[0.5 + 0.34 * math.log10(max(d["amount"], 10)) for _, _, d in fraud],
        arrows=True,
        arrowsize=13,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.08",
        ax=ax,
    )

    nx.draw_networkx_labels(
        graph,
        pos,
        labels=pattern_labels(case["pattern_type"], accounts),
        font_size=8,
        font_weight="bold",
        font_color="#111111",
        ax=ax,
    )

    # Per-edge timing offsets along the fraud path, staggered along each
    # edge so fan-in/fan-out labels don't pile up at the midpoints.
    w_start = datetime.strptime(case["window_start"], TS_FMT)
    timed = sorted(fraud_edges, key=lambda e: e["_ts"])
    seen: set[tuple[str, str]] = set()
    for i, e in enumerate(timed):
        key = (e["src_id"], e["dst_id"])
        if key in seen:
            continue
        seen.add(key)
        (x1, y1), (x2, y2) = pos[e["src_id"]], pos[e["dst_id"]]
        t = (0.38, 0.56, 0.74)[i % 3]
        ax.text(
            x1 + t * (x2 - x1),
            y1 + t * (y2 - y1) + 0.09,
            offset_label((e["_ts"] - w_start).total_seconds()),
            fontsize=6.5,
            color="#8b1a1a",
            ha="center",
            va="bottom",
        )

    if annotation:
        ax.text(
            0.015,
            0.985,
            annotation,
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            family="monospace",
            bbox={"boxstyle": "round,pad=0.45", "fc": "#fff7e6", "ec": "#f28e2b"},
        )

    n_edges = len(fraud_edges)
    ax.set_title(
        f"{title} — {case['pattern_id']} " f"({n_edges} transactions over {window_label(case)})",
        fontsize=12,
        pad=12,
    )
    legend = [
        Line2D([], [], color=FRAUD_EDGE, lw=2.2, label="injected fraud transaction"),
        Line2D([], [], color=NORMAL_EDGE, lw=1.4, label="legitimate background (1-hop)"),
        mpatches.Patch(color=ROLE_COLOURS["customer"], label="customer wallet"),
        mpatches.Patch(color=ROLE_COLOURS["agent"], label="agent wallet"),
        mpatches.Patch(color=ROLE_COLOURS["merchant"], label="merchant"),
        mpatches.Patch(color=ROLE_COLOURS["aggregator"], label="aggregator"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=7.5, framealpha=0.9)
    fig.text(0.99, 0.012, caption, ha="right", fontsize=6.5, color="#9a9a9a")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", required=True, help="gen-fraud-graph output directory")
    ap.add_argument("--out", default="docs/images", help="directory for the PNGs")
    ap.add_argument("--layout-seed", type=int, default=11, help="layout determinism seed")
    ap.add_argument(
        "--caption",
        default="generated by gen-fraud-graph — momo-100k preset",
        help="footer caption stamped on each image",
    )
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cases = read_rows(os.path.join(args.data_dir, "fraud", "fraud_cases.csv"))
    picks = [
        (pick_case(cases, ptype, depth), stem, title) for ptype, depth, stem, title in SHOWCASE
    ]

    node_union = {acc for case, _, _ in picks for acc in case["involved_accounts"].split("|")}
    print(f"scanning background traffic for {len(node_union)} pattern wallets...")
    background = load_background(args.data_dir, node_union)

    for case, stem, title in picks:
        fraud_edges = load_fraud_edges(args.data_dir, case)
        annotation = None
        if case["pattern_type"] == "sim_swap_takeover":
            burst_start = datetime.strptime(case["window_start"], TS_FMT)
            swap = latest_swap_before(args.data_dir, case["start_acc_id"], burst_start)
            if swap:
                minutes = (burst_start - swap["_ts"]).total_seconds() / 60
                annotation = (
                    f"sim_events.csv: victim's SIM re-bound\n"
                    f"{swap['old_sim_id']} → {swap['new_sim_id']}\n"
                    f"{minutes:.0f} min before the first cash-out"
                )
        involved = set(case["involved_accounts"].split("|"))
        local_bg = [e for e in background if e["src_id"] in involved or e["dst_id"] in involved]
        render_case(
            case,
            title,
            fraud_edges,
            local_bg,
            os.path.join(args.out, f"{stem}.png"),
            args.layout_seed,
            args.caption,
            annotation,
        )


if __name__ == "__main__":
    main()
