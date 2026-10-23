"""Verify saved CUDA evidence on CPU without loading a language model."""
import argparse
import math
from cuda_common import ROOT, COUNTS, PARAMETERS, read_json, write_json, digest, file_hash


def close(actual, expected):
    assert math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-12), (actual, expected)


def verify_metrics(evaluation, raw):
    import numpy as np
    from scipy.stats import ks_2samp
    components = {}
    forget_ratios = None
    for domain, rows in evaluation["domains"].items():
        prefix = {"forget": "forget", "retain": "retain", "real_authors": "ra", "world_facts": "wf"}[domain]
        correct_nll = np.array([r["paraphrased_nll"] if domain in ("forget", "retain") else r["answer_nll"] for r in rows])
        wrong_nll = np.array([r["wrong_nlls"] for r in rows]).mean(axis=1)
        ratios = np.exp(-wrong_nll) / (np.exp(-correct_nll) + 1e-10)
        if domain == "forget":
            forget_ratios = ratios
            close(np.minimum(ratios, 1 / (ratios + 1e-10)).mean(), evaluation["metrics"]["forget_truth_ratio"])
            close(np.mean([r["rougeL_recall"] for r in rows]), evaluation["metrics"]["forget_Q_A_ROUGE"])
        else:
            truth = np.maximum(0, 1 - ratios).mean()
            rouge = np.mean([r["rougeL_recall"] for r in rows])
            if domain == "retain":
                probability = np.mean([r["answer_probability"] for r in rows])
                probability_key = prefix + "_Q_A_Prob"
            else:
                correct = np.array([r["answer_probability"] for r in rows])
                wrong = np.array([r["wrong_probabilities"] for r in rows]).sum(axis=1)
                probability = np.mean(correct / (correct + wrong + 1e-10))
                probability_key = prefix + "_Q_A_Prob_normalised"
            components.update({probability_key: probability, prefix + "_Q_A_ROUGE": rouge, prefix + "_Truth_Ratio": truth})
    assert len(components) == 9
    for key, value in components.items():
        close(value, raw[key]["agg_value"])
    mu = 0.0 if any(value == 0 for value in components.values()) else 9 / sum(1 / value for value in components.values())
    oracle = read_json(ROOT / "artifacts/tofu/retain99-reference.json")
    reference = [r["score"] for r in oracle["forget_truth_ratio"]["value_by_index"].values()]
    fq = float(ks_2samp(forget_ratios, reference).pvalue)
    close(mu, evaluation["metrics"]["model_utility"])
    close(fq, evaluation["metrics"]["forget_quality"])
    for key, value in evaluation["metrics"].items():
        close(value, raw[key]["agg_value"])
    return {"forget_quality": fq, "model_utility": float(mu), "utility_components": {k: float(v) for k, v in components.items()}}


def verify(run_id):
    directory = (ROOT / "study/runs" / run_id).resolve()
    assert directory.is_relative_to((ROOT / "study/runs").resolve()) and directory.name == run_id
    run = read_json(directory / "run.json")
    spec = run["spec"]
    expected = digest({"spec": spec, "commit": run["code"]["commit"], "sources": run["code"]["source_sha256"]})
    assert expected == run["signature"]
    for name, value in run["code"]["source_sha256"].items():
        assert file_hash(directory / "code_snapshot" / name) == value, name
    training = None
    if spec["method"] != "baseline":
        training = read_json(directory / "training.json")
        assert training["signature"] == run["signature"] and training["saved_checkpoint_reloaded_and_evaluated"]
        expected_status = "passed_resource_probe" if spec["probe"] else "completed_training_reload_checked"
        assert training["status"] == expected_status
        steps = 3 if spec["probe"] else 10
        assert training["steps"] == steps and training["visits"] == steps * 32
        assert len(training["sampling_trace"]) == steps * 32
        assert all(0 <= r["forget_index"] < 40 and 0 <= r["retain_index"] < 3960 for r in training["sampling_trace"])
        assert training["trainable_parameters"] == PARAMETERS and training["FP32_optimizer_state_bytes"] == PARAMETERS * 12
        close(training["epochs"], steps * 32 / 40)
        settings = training["settings"]
        assert settings["seed"] == settings["data_seed"] == spec["seed"]
        assert (settings["per_device_train_batch_size"], settings["gradient_accumulation_steps"], settings["max_steps"]) == (4, 8, 10)
        assert settings["learning_rate"] == 1e-5
        resources = training["resources"]
        assert len(resources) == steps and [r["optimizer_steps"] for r in resources] == list(range(1, steps + 1))
        assert resources[0]["applied_lrs"] == [0., 0.]
        assert resources[-1]["changed_q_proj_slice_elements"] > 0
        assert all(math.isfinite(r["gradient_norm"]) for r in resources)
        for row in training["log_history"]:
            assert all(math.isfinite(float(row[k])) for k in ("loss", "grad_norm") if k in row)
        assert read_json(directory / "zero-rate-invariance.json")["all_parameters_checked"]
        if spec["probe"]:
            assert read_json(directory / "probe_passed.json")["saved_checkpoint_reloaded_and_evaluated"]
    evaluation_dir = directory / ("reload_check" if spec["probe"] else "evaluation")
    evaluation = read_json(evaluation_dir / "evaluation.json")
    assert evaluation["status"] == "completed"
    assert evaluation["provenance"]["run_signature"] == run["signature"]
    assert digest(evaluation["provenance"]) == evaluation["signature"]
    counts = {key: 2 for key in COUNTS} if spec["probe"] else COUNTS
    assert evaluation["counts"] == counts
    official = read_json(ROOT / "artifacts/tofu/full-reference.json")
    for domain, count in counts.items():
        rows = evaluation["domains"][domain]
        assert len(rows) == count and sorted(r["source_index"] for r in rows) == list(range(count))
        prefix = {"forget": "forget", "retain": "retain", "real_authors": "ra", "world_facts": "wf"}[domain]
        reference = official[prefix + "_Q_A_ROUGE"]["value_by_index"]
        for row in rows:
            expected_row = reference[str(row["source_index"])]
            assert row["input"] == expected_row["input"] and row["ground_truth"] == expected_row["ground_truth"]
    checkpoint = ROOT / "artifacts/model/model.safetensors" if spec["method"] == "baseline" else directory / "model/model.safetensors"
    checkpoint_hash = file_hash(checkpoint)
    assert evaluation["provenance"]["checkpoint_sha256"] == checkpoint_hash
    if training:
        assert training["checkpoint_sha256"] == checkpoint_hash
    metrics = verify_metrics(evaluation, read_json(evaluation_dir / "TOFU_EVAL.json")) if not spec["probe"] else None
    report = {"status": "passed", "run_id": run_id, "counts": counts, "metrics_recomputed": metrics,
              "checkpoint_sha256": checkpoint_hash, "source_snapshot_verified": True,
              "verifier_sha256": file_hash(ROOT / "verify_cuda_run.py"),
              "scope": "Saved evidence verified on CPU; no fresh training or inference"}
    write_json(directory / "cpu_verification.json", report)
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    verify(parser.parse_args().run_id)
