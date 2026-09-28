# Contributing to gen_fraud_graph

Thank you for your interest in contributing to **gen_fraud_graph**! This document provides guidelines and information for contributors.

This repository is an independent, mobile-money-focused fork of [SantanderAI/gen-fraud-graph](https://github.com/SantanderAI/gen-fraud-graph), maintained by [@SergeWilfried](https://github.com/SergeWilfried). It is not affiliated with or governed by Santander. Changes that are not specific to mobile money may be better proposed upstream.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Reporting Bugs](#reporting-bugs)
- [Proposing Features](#proposing-features)
- [Pull Request Process](#pull-request-process)
- [Code Style](#code-style)
- [Testing](#testing)
- [Licensing of Contributions](#licensing-of-contributions)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior privately to the maintainer, [@SergeWilfried](https://github.com/SergeWilfried).

## How to Contribute

### Reporting Bugs

1. **Check existing issues** — Search the [issue tracker](../../issues) to see if the bug has already been reported.
2. **Open a new issue** — If not, open a new issue using the **Bug Report** template. Include:
   - A clear and descriptive title
   - Steps to reproduce the behavior
   - Expected behavior vs actual behavior
   - Environment details (OS, Python version, package versions)
   - Any relevant logs or screenshots

### Proposing Features

1. **Open an issue** using the **Feature Request** template.
2. Describe the problem you are trying to solve.
3. Describe your proposed solution.
4. Discuss the feature with maintainers before implementing.

## Pull Request Process

1. **Fork** the repository to your GitHub account.
2. **Create a branch** from `main` with a descriptive name:
   ```bash
   git checkout -b feature/add-new-typology
   ```
3. **Make your changes** following the [Code Style](#code-style) guidelines.
4. **Add tests** for any new functionality.
5. **Update documentation** if your changes affect the public API.
6. **Commit** with clear, descriptive commit messages following [Conventional Commits](https://www.conventionalcommits.org/):
   ```
   feat: add smurfing typology for AML detection
   fix: correct edge weight calculation in ring generator
   docs: update export format documentation
   ```
7. **Push** your branch and open a Pull Request against `main`.
8. **Wait for review** — the maintainer aims to review PRs within 2 weeks.

### PR Requirements

All pull requests must pass the following automated checks (as they appear in
the PR checks panel) before merge:

- [ ] **`Lint & format & type-check`** and **`Test (Python 3.10/3.11/3.12)`** — Ruff, Black, Mypy, unit tests with coverage
- [ ] **`Analyze (python)`** — SAST (CodeQL)
- [ ] **`pip-audit`** — Dependency vulnerability audit
- [ ] **`Dependency license allowlist`** and **`SPDX headers`** — License compliance
- [ ] **`Scan for internal patterns`** — No internal URLs, IPs, or corporate email addresses

Additionally:

- At least **1 maintainer approval** is required.
- All review conversations must be resolved.
- The branch must be up to date with `main`.

## Code Style

### Python

- Follow [PEP 8](https://peps.python.org/pep-0008/).
- Use [Black](https://black.readthedocs.io/) for formatting (line length: 100).
- Use [Ruff](https://docs.astral.sh/ruff/) for linting.
- Use [mypy](https://mypy-lang.org/) for type checking.
- All public functions and classes must have docstrings (Google style).

### File Headers

Every Python file must declare its license in the first five lines (checked by CI).
Keep the existing copyright line on files inherited from upstream; new files use:

```python
# Copyright (c) 2026 SergeWilfried
# SPDX-License-Identifier: Apache-2.0
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | Use |
|:---|:---|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `test:` | Adding or updating tests |
| `refactor:` | Code refactoring (no feature/fix) |
| `ci:` | CI/CD changes |
| `chore:` | Maintenance tasks |

## Testing

- Write tests for all new functionality.
- Use [pytest](https://docs.pytest.org/) as the test framework.
- Place tests in the `tests/` directory, mirroring the source structure.
- Run the full test suite before submitting a PR:
  ```bash
  pytest tests/ -v --cov=gen_fraud_graph
  ```
- Minimum code coverage target: **80%**.

## Licensing of Contributions

There is no separate Contributor License Agreement. As described in Section 5 of the
[Apache License 2.0](LICENSE), any contribution you intentionally submit to this project is
licensed under the Apache License 2.0, without additional terms or conditions.

## Release Process

This project follows [Semantic Versioning (SemVer)](https://semver.org/):

- **MAJOR** — Incompatible API changes
- **MINOR** — New features (backward-compatible)
- **PATCH** — Bug fixes (backward-compatible)

Releases are managed by maintainers. If you believe a release is warranted, open an issue to discuss.

---

Thank you for contributing to **gen_fraud_graph**!
