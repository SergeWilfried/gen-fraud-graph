# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Output writers for generated graph data."""

from __future__ import annotations

import csv
import io
import os
import zipfile
from collections.abc import Sequence
from typing import Literal

from gen_fraud_graph.schema import (
    ACCOUNT_HEADERS,
    TRANSACTION_HEADERS,
    neptune_account_headers,
    neptune_transaction_headers,
)


def get_headers(
    doc_type: Literal["account", "transaction"],
    fmt: Literal["csv", "neptune"],
) -> list[str]:
    """Return CSV column headers for *doc_type* in the given *fmt*.

    Headers are derived from the dataclass specs in
    :mod:`gen_fraud_graph.schema`, the single source of truth.
    """
    if fmt == "neptune":
        if doc_type == "account":
            return neptune_account_headers()
        return neptune_transaction_headers()

    if doc_type == "account":
        return list(ACCOUNT_HEADERS)
    return [*TRANSACTION_HEADERS, "embedding"]


def write_output(
    file_base_name: str,
    headers: Sequence[str],
    rows: Sequence[Sequence],
    *,
    compress: bool = False,
) -> str:
    """Write *rows* to a CSV file, optionally ZIP-compressed.

    Returns:
        The path of the written file.
    """
    csv_filename = f"{file_base_name}.csv"

    if compress:
        zip_filename = f"{file_base_name}.csv.zip"
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(headers)
        writer.writerows(rows)
        with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(os.path.basename(csv_filename), buf.getvalue())
        buf.close()
        return zip_filename

    with open(csv_filename, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(rows)
    return csv_filename


def append_csv(
    csv_path: str,
    headers: Sequence[str],
    rows: Sequence[Sequence],
    *,
    resume_from: int = 0,
) -> None:
    """Append *rows* to an existing or new CSV file.

    If the file already exists, writing resumes after its current content.
    """
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as fh:
        writer = csv.writer(fh)
        if not file_exists:
            writer.writerow(headers)
        writer.writerows(rows)
