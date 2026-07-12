# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""Tests for the canonical benchmark presets."""

from __future__ import annotations

from unittest import mock

import pytest

from gen_fraud_graph.presets import PRESETS, preset_config


class TestPresetConfig:
    def test_known_presets_build_configs(self):
        for name in PRESETS:
            cfg = preset_config(name, output_dir="/tmp/x")
            assert cfg.seed is not None
            assert cfg.embedding_provider == "fake"
            assert cfg.output_dir == "/tmp/x"

    def test_momo_100k_shape(self):
        cfg = preset_config("momo-100k")
        assert cfg.num_accounts == 100_000
        assert cfg.num_transactions == 900_000
        assert cfg.seed == 100_001
        assert cfg.workers == 4

    def test_pattern_counts_scale_linearly(self):
        """The fraud mix must stay comparable across preset sizes."""
        small = preset_config("momo-100k")
        large = preset_config("momo-10m")
        assert large.num_fraud_rings == small.num_fraud_rings * 100
        assert large.num_sim_swap_patterns == small.num_sim_swap_patterns * 100

    def test_seeds_are_distinct(self):
        seeds = {p["seed"] for p in PRESETS.values()}
        assert len(seeds) == len(PRESETS)

    def test_unknown_preset_raises(self):
        with pytest.raises(KeyError, match="momo-100k"):
            preset_config("nope")


class TestPresetCLI:
    def test_preset_rejects_pinned_flags(self):
        from gen_fraud_graph.cli import main

        with pytest.raises(SystemExit) as exc:
            main(["--preset", "momo-100k", "--scale", "0.5"])
        assert exc.value.code == 2

    def test_preset_builds_expected_config(self, tmp_dir):
        from gen_fraud_graph import cli

        with mock.patch.object(cli, "FraudGraphGenerator") as gen:
            with pytest.raises(SystemExit) as exc:
                cli.main(["--preset", "momo-100k", "--output", tmp_dir])
            assert exc.value.code == 0
        cfg = gen.call_args.args[0]
        assert cfg.seed == 100_001
        assert cfg.output_dir == tmp_dir
        gen.return_value.run.assert_called_once()
