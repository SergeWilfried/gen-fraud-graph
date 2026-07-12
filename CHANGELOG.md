# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Reproducible generation: `Config(seed=...)` / `--seed` makes every output
  file byte-identical across runs of the same configuration. The master
  seed derives independent child streams for account workers, transaction
  workers, the fraud phase, and the fake embedding vectors
- `examples/baseline_xgb.py` — XGBoost baseline ladder that reads generated
  output directly (provenance labels, wallet/KYC joins, SIM-event join)
  and trains one model per cumulative feature tier: bank-style amounts,
  + velocity, + MoMo schema, + graph topology (degrees, reciprocity,
  directed 3-/4-cycle counts through each edge, PageRank). Reports PR-AUC,
  recall@FPR, and per-typology pattern recall per tier; doubles as the
  benchmark's leak detector. New optional dependency extra:
  `gen-fraud-graph[baseline]`
- Legitimate business segment in transaction amounts (`BUSINESS_AMOUNT_PARAMS`)
  so the >1M XOF band is not fraud-dominated
- Benign decoy SIM-swap events (default 20 per fraudulent swap) with
  uniform, unlabeled event IDs — the presence of a `sim_events.csv` row no
  longer identifies a takeover
- Mobile-money schema (`schema.py`, single source of truth for CSV specs):
  wallet nodes carry `msisdn`, `account_type` (customer/agent/super-agent/
  merchant/aggregator), `kyc_tier`, `sim_id`, `device_id`,
  `registration_agent_id`, `zone`, and agent `float_balance`; transaction
  edges carry `tx_type`, `channel`, `agent_id`, `fee`, and `commission`
  (banded tariffs that make commission-split fraud economically real)
- `fraud/sim_events.csv` — SIM re-binding events (same msisdn, new SIM)
  emitted minutes before each SIM-swap cash-out burst, so the takeover is
  expressible in the data rather than only in the ground truth
- `fraud_cases.csv` gains `window_start`/`window_end` ground-truth columns
- Simulated activity window (`Config.sim_start_date`, `Config.sim_days`)
  with diurnally weighted timestamps for all traffic
- Structuring, mobile-money split, SIM-swap, and mule typologies now bind
  their agent-side roles to real agent-typed wallets

### Changed
- Amounts are XOF integers (5 FCFA granularity); legitimate traffic is
  log-normal per transaction type and fraud amount defaults were revised
  to overlap the legitimate distribution
- Injected rows draw descriptions, channels, and tariffs from the same
  machinery as legitimate rows

### Fixed
- Removed label leakage that made injected fraud trivially separable:
  disjoint "suspicious" description vocabulary, class-constant timestamps,
  fixed 12,000,000 ring amounts, and consecutive smurf account IDs

## [0.1.0] - 2026-07-06

### Added
- Core 3-phase generation pipeline: accounts → transactions → fraud rings
- `Config` dataclass with scale factor, embedding provider, output format, workers
- `FraudGraphGenerator` orchestrator with parallel `ProcessPoolExecutor` workers
- `EmbeddingGenerator` with three backends: `fake` (random), `local` (SentenceTransformers), `openai`
- `FraudRingGenerator` — cyclic money-laundering patterns with configurable depth (4–7 hops)
- CSV and AWS Neptune bulk-load output formats
- Resume support for interrupted generation (incremental file append)
- ZIP compression option for output files
- `gen-fraud-graph` CLI with `--scale`, `--workers`, `--provider`, `--format` flags
- Python API: `from gen_fraud_graph import Config, FraudGraphGenerator`
- `verify` module to validate fraud patterns against generated transaction edges
- Full test suite covering config, embeddings, exporters, typologies, and end-to-end pipeline
- GitHub Actions workflows (all third-party actions pinned to SHA digests):
  - `ci.yml` — ruff + black + mypy + pytest matrix (3.10/3.11/3.12) with Codecov
  - `codeql.yml` — CodeQL SAST (push, PR, weekly cron)
  - `dep-scan.yml` — `pip-audit` (push, PR, daily cron)
  - `license-check.yml` — dependency-license allowlist + SPDX header verification
  - `pattern-check.yml` — internal-pattern scan with allowlist
  - `cla.yml` — CLA Assistant Lite
  - `stale.yml` — stale issues/PRs automation
  - `release.yml` — PyPI publish via OIDC trusted publishing on GitHub Release
- `.github/dependabot.yml` — weekly Python and GitHub Actions updates
- Issue templates (bug, feature) and PR template
- Apache 2.0 LICENSE + NOTICE, CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md, CODEOWNERS

### Fixed
- Fraud rings no longer draw overlapping account ranges. Each ring picked a contiguous block of accounts without excluding accounts already used by earlier rings, so ring ranges could overlap: two rings merged into a single non-cycle component and their `involved_accounts` labels shared accounts. Rings are now placed on disjoint ranges.
- Corrected the README installation section: `pip install gen-fraud-graph` fails because the package is not yet published to PyPI, so the docs now lead with the from-source install (`uv pip install -e`) and note that the PyPI release is pending.
- Preserve all generated account and transaction rows when the requested totals do not divide evenly across worker batches.

[Unreleased]: https://github.com/SantanderAI/gen-fraud-graph/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/SantanderAI/gen-fraud-graph/releases/tag/v0.1.0
