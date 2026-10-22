"""Guard against output/evidence collisions and training-budget regressions."""
import argparse
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cuda_common as common


class RunIdentityTests(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace(method="GradDiff", seed=0, alpha=1., gamma=1.,
                                       run_id="unit-probe", output_dir=None, probe=True)

    def test_protocol_budget_and_parameter_propagation(self):
        self.args.seed, self.args.alpha, self.args.gamma = 2, .5, 2.
        directory, spec = common.run_spec(self.args)
        self.assertEqual((spec["seed"], spec["alpha"], spec["gamma"]), (2, .5, 2.))
        self.assertEqual((spec["updates"], spec["scheduler_horizon_updates"]), (3, 10))
        self.assertEqual(spec["protocol"]["microbatch"] * spec["protocol"]["gradient_accumulation_steps"], 32)
        self.assertTrue(directory.is_relative_to(common.ROOT / "study/runs"))
        self.args.probe = False
        self.assertEqual(common.run_spec(self.args)[1]["updates"], 10)

    def test_path_and_ignored_parameter_rejection(self):
        for name in ("../old", "a/b", "..", ""):
            self.args.run_id = name
            with self.assertRaises(ValueError):
                common.run_spec(self.args)
        self.args.run_id = "unit-probe"
        self.args.output_dir = common.ROOT / "reproduction_full/runs/unit-probe"
        with self.assertRaises(ValueError):
            common.run_spec(self.args)
        self.args.output_dir = None
        self.args.method, self.args.alpha = "GradAscent", .5
        with self.assertRaises(ValueError):
            common.run_spec(self.args)

    def test_cannot_reuse_run_for_different_settings_or_code(self):
        _, spec = common.run_spec(self.args)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            (root / "source").write_text("real fixture bytes")
            identity = {"commit": "test", "source_sha256": {"source": common.file_hash(root / "source")}}
            with patch.object(common, "ROOT", root), patch.object(common, "source_identity", return_value=identity):
                self.check_run_claim(root, spec, identity)

    def check_run_claim(self, root, spec, identity):
        destination = root / "study/runs/unit"
        signature = common.claim_run(destination, spec)
        self.assertEqual(signature, common.claim_run(destination, spec))
        self.assertEqual((destination / "code_snapshot/source").read_text(), "real fixture bytes")
        altered = copy.deepcopy(spec)
        altered["seed"] = 1
        with self.assertRaises(ValueError):
            common.claim_run(destination, altered)
        identity["source_sha256"]["source"] = "changed"
        with self.assertRaises(ValueError):
            common.claim_run(destination, spec)

    def test_never_adopt_an_existing_unlabelled_result(self):
        _, spec = common.run_spec(self.args)
        with tempfile.TemporaryDirectory() as temp, patch.object(common, "source_identity", return_value={"commit": "test", "source_sha256": {}}):
            destination = Path(temp)
            (destination / "existing-result.json").write_text("{}")
            with self.assertRaises(ValueError):
                common.claim_run(destination, spec)


if __name__ == "__main__":
    unittest.main()
