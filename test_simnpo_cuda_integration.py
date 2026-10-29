"""CPU tests for SimNPO CLI/spec integration; no model or CUDA initialization."""

import argparse
import ast
import math
from pathlib import Path

import pytest

import cuda_common


def args(method="SimNPO", **overrides):
    values = dict(method=method, seed=0, alpha=1.0, gamma=1.0, beta=4.5,
                  delta=0.0, run_id="simnpo-unit", output_dir=None, probe=True)
    values.update(overrides)
    return argparse.Namespace(**values)


def test_simnpo_spec_records_all_loss_parameters():
    _, spec = cuda_common.run_spec(args(alpha=0.3, gamma=0.125, beta=4.5, delta=0.2))
    assert {key: spec[key] for key in ("method", "alpha", "gamma", "beta", "delta")} == {
        "method": "SimNPO", "alpha": 0.3, "gamma": 0.125, "beta": 4.5, "delta": 0.2
    }


def test_ga_gd_specs_keep_previous_shape_and_ga_validation():
    _, ga = cuda_common.run_spec(args("GradAscent"))
    _, gd = cuda_common.run_spec(args("GradDiff", alpha=0.5, gamma=2.0))
    assert "beta" not in ga and "delta" not in ga
    assert "beta" not in gd and "delta" not in gd
    with pytest.raises(ValueError):
        cuda_common.run_spec(args("GradAscent", alpha=0.5))


@pytest.mark.parametrize("field,value", [("beta", 0.0), ("beta", math.inf), ("beta", math.nan), ("delta", math.inf), ("delta", math.nan)])
def test_simnpo_parameter_validation(field, value):
    with pytest.raises(ValueError):
        cuda_common.run_spec(args(**{field: value}))


def test_source_identity_adds_adapter_only_for_simnpo():
    normal = cuda_common.source_identity()
    simnpo = cuda_common.source_identity(include_simnpo=True)
    assert "simnpo_cuda.py" not in normal["source_sha256"]
    assert "simnpo_cuda.py" in simnpo["source_sha256"]


def test_train_cuda_has_explicit_nonfallthrough_dispatch():
    tree = ast.parse(Path("train_cuda.py").read_text())
    source = Path("train_cuda.py").read_text()
    assert '"SimNPO": CudaSimNPO' in source
    assert '"GradDiff": CudaGradDiff' in source
    assert '"GradAscent": GradAscent' in source
    ast.parse(source)
