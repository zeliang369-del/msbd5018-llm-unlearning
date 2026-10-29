"""CPU-only NPO entry/data-flow checks; no CUDA or formal model loading."""
import argparse
import ast
import json
from pathlib import Path

import pytest

import cuda_common


def test_npo_run_spec_requires_and_binds_cache(tmp_path):
    cache = tmp_path / "reference.json"
    cache.write_text(json.dumps({"version": 1, "loss_protocol": "shifted_token_nll_sum_ignore_index_-100", "metadata": {}, "values": {}}))
    args = argparse.Namespace(method="NPO", seed=0, alpha=1., gamma=1., beta=.1,
                              reference_cache=cache, run_id="npo-unit", output_dir=None, probe=True)
    _, spec = cuda_common.run_spec(args)
    assert spec["method"] == "NPO"
    assert spec["beta"] == .1
    assert spec["reference_cache_sha256"]
    assert "reference_loss_protocol" in spec


def test_ga_gd_spec_shape_is_unchanged():
    for method, alpha, gamma in (("GradAscent", 1., 1.), ("GradDiff", .5, 2.)):
        args = argparse.Namespace(method=method, seed=0, alpha=alpha, gamma=gamma,
                                  beta=.1, reference_cache=None, run_id="unit", output_dir=None, probe=True)
        _, spec = cuda_common.run_spec(args)
        assert "beta" not in spec and "reference_cache" not in spec


def test_train_entry_has_npo_only_dispatch_and_no_model_metadata_argument():
    source = Path("train_cuda.py").read_text()
    ast.parse(source)
    assert '"NPO": CudaNPO' in source
    assert "sample_id" not in source


def test_npo_collator_path_is_dependency_gated():
    pytest.importorskip("transformers")
    from npo_data import NPOForgetDataset, NPOTrainingCollator
    assert NPOForgetDataset and NPOTrainingCollator
