# gen_fraud_graph

> Synthetic fraud graph generator for training and benchmarking graph-based fraud detection models in financial services.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/gen-fraud-graph.svg)](https://pypi.org/project/gen-fraud-graph/)
[![CI](https://github.com/SantanderAI/gen-fraud-graph/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SantanderAI/gen-fraud-graph/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/SantanderAI/gen-fraud-graph/branch/main/graph/badge.svg)](https://codecov.io/gh/SantanderAI/gen-fraud-graph)
[![CodeQL](https://github.com/SantanderAI/gen-fraud-graph/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/SantanderAI/gen-fraud-graph/actions/workflows/codeql.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/SantanderAI/gen-fraud-graph/badge)](https://scorecard.dev/viewer/?uri=github.com/SantanderAI/gen-fraud-graph)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg)](https://conventionalcommits.org)
[![GitHub last commit](https://img.shields.io/github/last-commit/SantanderAI/gen-fraud-graph)](https://github.com/SantanderAI/gen-fraud-graph/commits/main)

---

## Overview

**gen_fraud_graph** is an open-source Python tool that generates massive synthetic financial transaction graphs with injected fraud patterns and optional vector embeddings. It produces CSV datasets ready for ingestion into graph databases (TigerGraph, Neptune, Neo4j, JanusGraph) or for training graph neural networks (GNN).

The generator creates three types of data:
- **Wallet nodes** — synthetic mobile-money wallets (customer/agent/merchant hierarchy) with MSISDN, SIM and device identity, KYC tier, balances, and optional embedding vectors
- **Transaction edges** — typed mobile-money transactions (cash-in/out, P2P, merchant, airtime, bill pay, bank-to-wallet) with channel, fees, and agent commissions
- **Fraud typologies** — cyclic money-laundering rings, structuring/smurfing, mobile money agent-commission fraud, trade-based money laundering (TBML), hawala/informal value transfer networks, SIM-swap account takeover, and overdraft/micro-loan mule chains

### Key Features

- **Massive scale** — Generate from 1K to 100M+ accounts with configurable scale factor
- **Fraud pattern injection** — Cyclic money-laundering rings (4–7 hops), structuring, mobile money splits, TBML layering chains, hawala corridors, SIM-swap takeover cascades, and micro-loan mule chains
- **Parallel generation** — Multi-process workers for fast generation on high-core machines
- **Vector embeddings** — Three providers: `fake` (random, fast), `local` (SentenceTransformers), `openai` (API)
- **Multiple formats** — Generic CSV or AWS Neptune bulk-load format
- **Resume support** — Interrupted generation can resume from where it left off
- **Privacy by design** — All data is 100% synthetic; no real financial data is used

### Use Cases

- Training and evaluating **graph neural networks (GNN)** for fraud detection
- Benchmarking **anti-money laundering (AML)** detection algorithms
- Load-testing graph databases (TigerGraph, Neptune, JanusGraph, NebulaGraph, FalkorDB)
- Research in **financial crime detection** and **anomaly detection** on graphs
- Generating labeled datasets for **deep learning** on graph-structured data

---

## Quick Start

### Installation

> **Note:** `gen-fraud-graph` is not yet published to PyPI, so `pip install gen-fraud-graph` will fail with `No matching distribution found`. Until the first PyPI release, install from source as shown below. The PyPI badge above is pre-provisioned for the planned release.

Install from source using [uv](https://github.com/astral-sh/uv):
```bash
git clone https://github.com/SantanderAI/gen-fraud-graph.git
cd gen-fraud-graph
uv venv && source .venv/bin/activate
uv pip install -e '.[dev]'
```

With optional embedding providers (from the cloned source directory):
```bash
uv pip install -e '.[local]'    # SentenceTransformers (local model)
uv pip install -e '.[openai]'   # OpenAI API embeddings
uv pip install -e '.[all]'      # Everything including dev tools
```

If you prefer plain `pip` over `uv`, the source install works the same way:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
```

Once the package is published, `pip install gen-fraud-graph` will be the recommended path.

### CLI Usage

```bash
# Quick test (~1K accounts, ~9K transactions, fake embeddings)
gen-fraud-graph --scale 0.0001 --provider fake --output ./data

# Medium scale (~100K accounts, parallelized)
gen-fraud-graph --scale 0.01 --workers 4 --output ./data

# Full benchmark (~10M accounts, ~90M transactions)
gen-fraud-graph --scale 1.0 --workers 24 --output ./data

# Neptune bulk-load format
gen-fraud-graph --scale 0.01 --format neptune --output ./neptune_data

# Resume interrupted generation (skips completed files)
gen-fraud-graph --scale 1.0 --workers 24 --skip-accounts --output ./data
```

### CLI Arguments

| Flag | Default | Description |
|:---|:---|:---|
| `--scale` | `1.0` | Scale factor. `1.0` = ~10M accounts / ~90M transactions. `0.01` = ~100K accounts. |
| `--provider` | `fake` | Embedding provider: `fake` (random vectors), `local` (SentenceTransformers), `openai`. |
| `--output` | `data` | Output directory for generated CSV files. |
| `--workers` | `1` | Number of parallel worker processes. |
| `--batches` | `1` | Number of file chunks per worker. |
| `--format` | `csv` | Output format: `csv` (generic) or `neptune` (AWS Neptune bulk-load). |
| `--fraud-rings` | auto | Number of fraud rings. Default: auto-scaled from `--scale`. |
| `--trade-based-ml-patterns` | auto | Number of TBML patterns. Default: auto-scaled from `--scale`. |
| `--hawala-patterns` | auto | Number of hawala network patterns. Default: auto-scaled from `--scale`. |
| `--sim-swap-patterns` | auto | Number of SIM-swap account takeover patterns. Default: auto-scaled from `--scale`. |
| `--overdraft-mule-patterns` | auto | Number of overdraft/micro-loan mule chain patterns. Default: auto-scaled from `--scale`. |
| `--compress` | off | ZIP-compress output CSV files. |
| `--skip-accounts` | off | Skip account generation (useful when resuming). |

### Python API

```python
# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

from gen_fraud_graph import Config, FraudGraphGenerator

config = Config(
    scale_factor=0.001,         # ~10K accounts, ~90K transactions
    num_fraud_rings=50,         # 50 cyclic fraud patterns
    embedding_provider="fake",  # random vectors (fast, no model needed)
    workers=2,                  # 2 parallel processes
    output_dir="./output",
)

generator = FraudGraphGenerator(config)
generator.run()
```

### Verify Generated Patterns

```bash
python -m gen_fraud_graph.verify --data-dir ./data
```

### Measure Benchmark Difficulty

A baseline ladder ([`examples/baseline_xgb.py`](examples/baseline_xgb.py))
closes the loop: it reads the generated output directly, derives labels from
provenance, and trains one XGBoost model per cumulative feature tier —
bank-style columns, + velocity, + the MoMo schema, + graph topology. The
gaps between rungs are the point: they show that no single column solves
the benchmark and quantify what each signal layer is worth. Run it after
any generator change — a near-perfect bottom rung means a label leak.

```bash
pip install 'gen-fraud-graph[baseline]'
python examples/baseline_xgb.py --data-dir ./data              # full ladder
python examples/baseline_xgb.py --data-dir ./data --tier graph # one tier, detailed
```

Reference ladder on a 455K-transaction dataset (50K wallets, 730 injected
patterns, ~1% fraud edges; unseeded generation, so expect run-to-run
variation of a few points). Metrics are on the test slice of a temporal
split; the right-hand columns are per-typology pattern recall at a 1%-FPR
alert threshold:

| Tier | PR-AUC | R@1% FPR | R@0.1% FPR | Commission splits | SIM-swap | Mule chains | Cycles |
|:---|---:|---:|---:|---:|---:|---:|---:|
| `amounts` (bank-style) | 0.23 | 37% | 13% | 0/22 | 2/19 | 10/10 | 11/12 |
| `velocity` | 0.76 | 79% | 61% | 22/22 | 17/19 | 9/10 | 11/12 |
| `schema` (MoMo fields) | 0.88 | 91% | 76% | 22/22 | 19/19 | 10/10 | 11/12 |
| `graph` (topology) | 0.91 | 93% | 84% | 22/22 | 19/19 | 10/10 | 11/12 |

How to read it: amount/time columns alone are nearly useless against the
burst typologies (0/22 commission splits, 2/19 SIM-swaps) — the leak fixes
did their job. Velocity features recover the bursts; the MoMo schema fields
close SIM-swap takeovers (the `sim_events.csv` join only works combined
with burst features, since 95% of events are benign decoys) and add tariff
and KYC-cap signal; graph topology (directed cycle counts through each
edge, degrees, PageRank) adds the final margin, most visibly at the strict
0.1%-FPR operating point (76% → 84%). Slow laundering cycles stay the
hardest typology at every rung.

---

## Output Structure

```
data/
├── accounts/
│   ├── accounts_0_0.csv       # Account nodes (worker 0, batch 0)
│   └── accounts_1_0.csv       # Account nodes (worker 1, batch 0)
├── transactions/
│   ├── transactions_0_0.csv   # Transaction edges (worker 0, batch 0)
│   └── transactions_1_0.csv   # Transaction edges (worker 1, batch 0)
└── fraud/
    ├── transactions_fraud.csv  # Injected fraud transaction edges (all typologies)
    ├── fraud_cases.csv         # Ground truth (pattern type, accounts, time window)
    └── sim_events.csv          # SIM re-binding events behind SIM-swap takeovers
```

### CSV Schema

**accounts** (`accounts_*.csv`) — mobile-money wallets

| Column | Type | Description |
|:---|:---|:---|
| `account_id` | string | Wallet identifier and edge join key (`acc_0`, `acc_1`, ...) |
| `msisdn` | string | Subscriber phone number (`+2217...`), unique per wallet |
| `customer_name` | string | Synthetic customer name |
| `account_type` | string | `customer`, `agent`, `super_agent`, `merchant`, or `aggregator` — deterministic in the account ID (see `schema.py`) |
| `kyc_tier` | string | `unverified`, `light`, or `full` (BCEAO-style tiers; agent-side and business wallets are always `full`) |
| `sim_id` | string | SIM originally bound to the wallet (SIM swaps re-bind it via `sim_events.csv`) |
| `device_id` | string | Handset identifier; a small share of wallets share a household device |
| `registration_agent_id` | string | Agent that onboarded the wallet (empty for agent-side wallets) |
| `zone` | string | Coarse geography (region/district), urban-weighted |
| `balance` | int | Wallet balance in XOF (log-normal by account type) |
| `float_balance` | int | Agent e-float in XOF; `0` for non-agent wallets |
| `risk_score` | float | Risk score (0.0 – 1.0) |
| `creation_date` | string | Wallet opening date |

**transactions** (`transactions_*.csv`)

| Column | Type | Description |
|:---|:---|:---|
| `tx_id` | string | Unique transaction identifier |
| `src_id` | string | Source wallet |
| `dst_id` | string | Destination wallet |
| `tx_type` | string | `cash_in`, `cash_out`, `p2p`, `merchant_payment`, `airtime`, `bill_pay`, or `bank_to_wallet` |
| `channel` | string | `ussd`, `app`, `agent_pos`, or `api` |
| `agent_id` | string | Facilitating agent wallet on cash legs (empty otherwise) |
| `amount` | int | XOF amount (5 FCFA granularity); log-normal per `tx_type`, heavy-tailed — injected fraud overlaps this distribution by design |
| `fee` | int | Sender-paid fee in XOF (banded tariff, see `schema.py`) |
| `commission` | int | Operator-to-agent commission in XOF on cash legs |
| `timestamp` | string | Diurnally weighted timestamp inside the simulated window (`sim_start_date` + `sim_days`) |
| `description` | string | Drawn from a per-`tx_type` vocabulary shared by legitimate and injected rows |
| `embedding` | string | Pipe-separated embedding vector |

**sim_events** (`fraud/sim_events.csv`)

| Column | Type | Description |
|:---|:---|:---|
| `event_id` | string | Event identifier |
| `account_id` | string | Wallet whose SIM was swapped |
| `msisdn` | string | Unchanged subscriber number |
| `old_sim_id` / `new_sim_id` | string | SIM binding before / after the swap |
| `swap_ts` | string | Swap time — minutes before the cash-out burst it enables |

**fraud_cases** (`fraud/fraud_cases.csv`)

| Column | Type | Description |
|:---|:---|:---|
| `pattern_id` | string | Pattern identifier (`pat_0`, `struct_0`, `mm_0`, `tbml_0`, `hawala_0`, `simswap_0`, ...) |
| `start_acc_id` | string | Anchor account for the pattern (meaning varies by `pattern_type`) |
| `pattern_type` | string | One of `cycle`, `structuring`, `mobile_money_split`, `trade_based_ml`, `hawala_network`, `sim_swap_takeover`, `overdraft_mule_chain` |
| `depth` | int | Meaning varies by `pattern_type` — see below |
| `involved_accounts` | string | Pipe-separated list of accounts; role encoded by position — see below |
| `window_start` / `window_end` | string | Time window of the pattern's transactions (bursts span minutes; laundering cycles span days) |

`involved_accounts` positional convention and `depth` meaning, by `pattern_type`:
- `cycle`: ordered ring `acc_0\|acc_1\|...\|acc_{depth-1}` (edges wrap: last -> first). `depth` = ring length (4–7 hops).
- `structuring`: `coordinator\|smurf_0\|...\|smurf_{depth-1}` (`accounts[0]` = coordinator). `depth` = number of smurfs.
- `mobile_money_split`: `agent\|customer` (`accounts[0]` = agent). `depth` = number of split transactions.
- `trade_based_ml`: `exporter\|shell_importer\|intermediary_0\|...\|intermediary_{depth-1}\|beneficiary`. `depth` = number of layering intermediaries.
- `hawala_network`: `sender\|hawaladar_A\|hawaladar_B\|beneficiary` (fixed length 4). `depth` = number of edges emitted (3, or 4 if the periodic reverse settlement fired).
- `sim_swap_takeover`: `victim\|cashout_agent_0\|...\|cashout_agent_{depth-1}` (`accounts[0]` = victim). `depth` = number of cash-out agents.
- `overdraft_mule_chain`: `collector\|mule_0\|...\|mule_{depth-1}\|agent` (`accounts[0]` = collector). `depth` = number of mule accounts. The final collector -> agent edge amount equals the sum of all mule loan amounts.

---

## Scale Reference

| Scale | Accounts | Transactions | Fraud Rings | Approx. Size |
|:---|:---|:---|:---|:---|
| `0.0001` | 1,000 | 9,000 | 10 | ~2 MB |
| `0.001` | 10,000 | 90,000 | 10 | ~20 MB |
| `0.01` | 100,000 | 900,000 | 10 | ~200 MB |
| `0.1` | 1,000,000 | 9,000,000 | 100 | ~2 GB |
| `1.0` | 10,000,000 | 90,000,000 | 1,000 | ~20 GB |

---

## Project Structure

```
gen_fraud_graph/
├── src/gen_fraud_graph/
│   ├── __init__.py       # Package entry point
│   ├── cli.py            # CLI (gen-fraud-graph command)
│   ├── config.py         # Configuration dataclass
│   ├── embeddings.py     # Embedding providers (fake/local/openai)
│   ├── exporters.py      # CSV/ZIP output writers
│   ├── generator.py      # Core 3-phase pipeline orchestrator
│   ├── typologies.py     # Fraud ring generator
│   └── verify.py         # Pattern verification utility
├── tests/
│   └── test_generator.py # Unit and integration tests
├── examples/
│   └── basic_usage.py    # Minimal Python API example
├── .github/
│   ├── workflows/        # CI (ci, codeql, dep-scan, license-check,
│   │                     #     pattern-check, cla, stale, release)
│   ├── ISSUE_TEMPLATE/   # Bug + feature templates
│   ├── PULL_REQUEST_TEMPLATE.md
│   ├── dependabot.yml    # Weekly Python + Actions updates
│   └── pattern-check-allowlist.txt
├── pyproject.toml        # Package metadata and tool config
├── LICENSE               # Apache 2.0
├── NOTICE                # Apache 2.0 attribution
├── CONTRIBUTING.md       # Contribution guidelines
├── CODE_OF_CONDUCT.md    # Contributor Covenant v2.1
├── SECURITY.md           # Vulnerability disclosure policy
├── CODEOWNERS            # Maintainer approvals
└── CHANGELOG.md          # Release history
```

---

## Requirements

Core (always installed):
- Python >= 3.10
- NumPy >= 1.24
- Pandas >= 2.0
- tqdm >= 4.65

Optional:
- `sentence-transformers >= 2.2` — for `--provider local`
- `openai >= 1.0` — for `--provider openai`

---

## Contributing

We welcome contributions from the community. Please read our [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a pull request.

By contributing, you agree to the terms of our Contributor License Agreement (CLA).

---

## Security

To report a security vulnerability, please follow the process described in [SECURITY.md](SECURITY.md). **Do not open a public issue for security vulnerabilities.**

---

## Disclaimer

This software is an open source project from the **Santander AI Lab**, provided **"as is"** under its [license](LICENSE), without warranties or conditions of any kind. It is **not an official Banco Santander product or service**, carries no commitment of production support, and does not constitute financial, legal or professional advice.

"Santander" and its logo are registered trademarks of **Banco Santander, S.A.** The project license does not grant any right to use them beyond factual attribution.

If you believe you have found a security vulnerability, follow our [security policy](https://github.com/SantanderAI/.github/blob/main/SECURITY.md) — do not open a public issue. You are responsible for assessing the suitability of this software for your use case and for keeping your own deployments up to date.

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.

```
Copyright (c) 2026 Santander Group
SPDX-License-Identifier: Apache-2.0
```

---

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{gen_fraud_graph,
  title     = {gen\_fraud\_graph: Synthetic Fraud Graph Generator},
  author    = {Santander AI Lab},
  year      = {2026},
  url       = {https://github.com/SantanderAI/gen-fraud-graph},
  license   = {Apache-2.0}
}
```

---

<!-- GitHub repository metadata (for reference — configured via GitHub UI/API):
  description: "Synthetic fraud graph generator for benchmarking graph-based fraud detection models"
  topics: machine-learning, artificial-intelligence, fraud-detection, graph-neural-network,
          deep-learning, synthetic-data, financial-crime, anti-money-laundering, gnn,
          anomaly-detection, finance, python
  visibility: public
  license: Apache-2.0
  custom_properties:
    category: tool
    track: fast
    status: active
    team: ai-labs
-->
