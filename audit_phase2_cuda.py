"""Audit all six extensions and CUDA controls; write descriptive seed/weight summaries."""
import csv
from datetime import datetime, timezone
import io
import statistics

from cuda_common import ROOT, COUNTS, read_json, write_json, file_hash, digest
from run_phase2_cuda import MATRIX, DIRECTORY
from verify_cuda_run import verify


def csv_write(path, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    with path.open("w", encoding="utf-8", newline="") as output:
        output.write(stream.getvalue())


def audit():
    execution = read_json(DIRECTORY / "execution.json")
    assert execution["matrix"] == MATRIX
    assert execution["status"] in ("six_runs_verified_pending_batch_audit", "completed_verified")
    assert all(execution["stages"][r["run_id"]]["verification"] == "passed" for r in MATRIX)
    ids = ["baseline-s0-cuda-v1", "ga-s0-cuda-v1", "gd-a1-s0-cuda-v1"] + [r["run_id"] for r in MATRIX]
    records, trainings, rows = {}, {}, []
    canonical = read_json(ROOT / "study/runs/ga-s0-cuda-v1/run.json")
    metadata_exceptions = {"cuda_common.py", "capture_cuda_source.py"}
    setting_keys = ["per_device_train_batch_size", "gradient_accumulation_steps", "num_train_epochs", "max_steps",
                    "learning_rate", "weight_decay", "warmup_steps", "lr_scheduler_type", "adam_beta1", "adam_beta2",
                    "adam_epsilon", "max_grad_norm", "bf16", "fp16", "tf32", "gradient_checkpointing",
                    "gradient_checkpointing_kwargs", "accelerator_config"]
    canonical_settings = read_json(ROOT / "study/runs/ga-s0-cuda-v1/training.json")["settings"]
    for run_id in ids:
        directory = ROOT / "study/runs" / run_id
        verify(run_id)
        run = read_json(directory / "run.json")
        spec = run["spec"]
        evaluation = read_json(directory / "evaluation/evaluation.json")
        assert spec["protocol"] == canonical["spec"]["protocol"]
        assert spec["evaluation_profile"] == canonical["spec"]["evaluation_profile"]
        assert spec["device"] == "cuda:0" and spec["parameterization"] == "full_parameter_no_LoRA"
        assert evaluation["counts"] == COUNTS
        sources = run["code"]["source_sha256"]
        expected = canonical["code"]["source_sha256"]
        changed = [name for name in sources.keys() & expected.keys() if sources[name] != expected[name]]
        assert set(changed).issubset(metadata_exceptions), (run_id, changed)
        if run_id in {r["run_id"] for r in MATRIX}:
            row = next(r for r in MATRIX if r["run_id"] == run_id)
            assert all(spec[key] == value for key, value in row.items())
            assert sources == expected, "Extensions must match the final GA-control source guard"
        training = None
        if spec["method"] != "baseline":
            training = read_json(directory / "training.json")
            trainings[run_id] = training
            settings = training["settings"]
            assert all(settings[k] == canonical_settings[k] for k in setting_keys)
            assert settings["seed"] == settings["data_seed"] == spec["seed"]
            assert training["steps"] == 10 and training["visits"] == 320 and training["epochs"] == 8
            reload = read_json(directory / "reload_check/evaluation.json")
            assert reload["status"] == "completed" and reload["counts"] == {k: 2 for k in COUNTS}
            assert reload["metrics"] is None
            assert reload["provenance"]["checkpoint_sha256"] == training["checkpoint_sha256"]
            assert reload["provenance"]["run_signature"] == run["signature"]
            assert read_json(directory / "reload_check/forward-parity.json")["status"] == "passed"
        assert read_json(directory / "evaluation/forward-parity.json")["status"] == "passed"
        metrics = evaluation["metrics"]
        assert all(0 <= value <= 1 for value in metrics.values())
        records[run_id] = {"method": spec["method"], "seed": spec["seed"], "alpha": spec["alpha"],
                           "gamma": spec["gamma"], "metrics": metrics, "counts": evaluation["counts"],
                           "checkpoint_sha256": evaluation["provenance"]["checkpoint_sha256"],
                           "evaluation_seconds": evaluation["elapsed_seconds"],
                           "training_seconds": training["trainer_metrics"]["train_runtime"] if training else None,
                           "sampling_trace_sha256": digest(training["sampling_trace"]) if training else None,
                           "source_guard_differences_from_GA_control": changed,
                           "cpu_verification": "passed"}
        rows.append(dict(run_id=run_id, method=spec["method"], seed=spec["seed"], alpha=spec["alpha"], gamma=spec["gamma"],
                         **metrics, training_seconds=records[run_id]["training_seconds"],
                         evaluation_seconds=evaluation["elapsed_seconds"], checkpoint_sha256=records[run_id]["checkpoint_sha256"]))
    for seed in (0, 1, 2):
        ga, gd = f"ga-s{seed}-cuda-v1", f"gd-a1-s{seed}-cuda-v1"
        assert trainings[ga]["sampling_trace"] == trainings[gd]["sampling_trace"], (ga, gd)
    for run_id in ("gd-a0p5-s0-cuda-v1", "gd-a2-s0-cuda-v1"):
        assert trainings[run_id]["sampling_trace"] == trainings["gd-a1-s0-cuda-v1"]["sampling_trace"]
    trace_hashes = [records[f"ga-s{s}-cuda-v1"]["sampling_trace_sha256"] for s in (0, 1, 2)]
    assert len(set(trace_hashes)) == 3
    checkpoints = [records[run_id]["checkpoint_sha256"] for run_id in ids]
    assert len(set(checkpoints)) == len(checkpoints), "Every treatment must have its own distinct checkpoint"
    summary_rows = []
    for method, prefix in (("GradAscent", "ga"), ("GradDiff", "gd-a1")):
        for metric in ("forget_quality", "model_utility", "forget_truth_ratio", "forget_Q_A_ROUGE"):
            values = [records[f"{prefix}-s{s}-cuda-v1"]["metrics"][metric] for s in (0, 1, 2)]
            summary_rows.append(dict(method=method, metric=metric, n=3, seed_0=values[0], seed_1=values[1], seed_2=values[2],
                                     mean=statistics.mean(values), sample_sd=statistics.stdev(values),
                                     minimum=min(values), maximum=max(values)))
    paired = [dict(seed=s, GA_MU_minus_GD_MU=records[f"ga-s{s}-cuda-v1"]["metrics"]["model_utility"] -
                   records[f"gd-a1-s{s}-cuda-v1"]["metrics"]["model_utility"],
                   GD_FQ_minus_GA_FQ=records[f"gd-a1-s{s}-cuda-v1"]["metrics"]["forget_quality"] -
                   records[f"ga-s{s}-cuda-v1"]["metrics"]["forget_quality"]) for s in (0, 1, 2)]
    report = dict(status="passed", recorded_at=datetime.now(timezone.utc).isoformat(),
                  extensions_verified=6, controls_verified=3, complete_evaluations=9, questions_per_evaluation=657,
                  runs=records, seed_summary=summary_rows, paired_seed_differences=paired,
                  checks=dict(individual_CPU_verification=True, exact_source_snapshots=True,
                              same_training_and_evaluation_math=True, same_protocol_and_budget=True,
                              paired_sampling_across_methods=True, matched_sampling_across_weights=True,
                              different_sampling_across_seeds=True, distinct_checkpoints=True,
                              all_saved_checkpoint_reload_checks=True),
                  limitations=["Three seeds only; all individual results retained; no general method-ranking claim",
                               "FQ is a KS-test p-value, not accuracy or a percentage forgotten; its mean is descriptive only",
                               "Three weight points at seed 0 are not a continuous trade-off curve",
                               "Mac runs are excluded from all seed and weight summaries",
                               "No prompt-robustness or new-method result is claimed",
                               "Baseline/GD seed-0 controls predate metadata-guard strengthening; execution math matches"],
                  auditor_sha256=file_hash(ROOT / "audit_phase2_cuda.py"),
                  verifier_sha256=file_hash(ROOT / "verify_cuda_run.py"))
    csv_write(DIRECTORY / "results.csv", rows)
    csv_write(DIRECTORY / "seed-summary.csv", summary_rows)
    csv_write(DIRECTORY / "paired-seed-differences.csv", paired)
    write_json(DIRECTORY / "audit.json", report)
    execution.update(status="completed_verified", final_audit="study/phase2/audit.json", updated_at=report["recorded_at"])
    write_json(DIRECTORY / "execution.json", execution)
    print({"phase2_audit": "passed", "extensions": 6, "controls": 3, "seed_summary": summary_rows})


if __name__ == "__main__":
    audit()
