"""Run the six agreed CUDA extensions serially; stop on any failed stage."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

from cuda_common import ROOT, read_json, write_json, source_identity


MATRIX = [
    dict(run_id="gd-a0p5-s0-cuda-v1", method="GradDiff", seed=0, alpha=.5, gamma=1.),
    dict(run_id="gd-a2-s0-cuda-v1", method="GradDiff", seed=0, alpha=2., gamma=1.),
    dict(run_id="ga-s1-cuda-v1", method="GradAscent", seed=1, alpha=1., gamma=1.),
    dict(run_id="gd-a1-s1-cuda-v1", method="GradDiff", seed=1, alpha=1., gamma=1.),
    dict(run_id="ga-s2-cuda-v1", method="GradAscent", seed=2, alpha=1., gamma=1.),
    dict(run_id="gd-a1-s2-cuda-v1", method="GradDiff", seed=2, alpha=1., gamma=1.),
]
DIRECTORY = ROOT / "study/phase2"


def now():
    return datetime.now(timezone.utc).isoformat()


def execute(script, arguments, log_path):
    command = [sys.executable, "-X", "utf8", str(ROOT / script), *arguments]
    print({"event": "starting", "script": script, "arguments": arguments, "at": now()}, flush=True)
    with log_path.open("ab") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                time.sleep(5)
            if process.returncode:
                raise RuntimeError(f"{script} failed (exit {process.returncode}); inspect {log_path}")
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            raise
    print({"event": "passed", "script": script, "at": now()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Actually train and evaluate all six runs")
    args = parser.parse_args()
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    identity = source_identity()
    path = DIRECTORY / "execution.json"
    if path.exists():
        state = read_json(path)
        if state["matrix"] != MATRIX or state["source_sha256"] != identity["source_sha256"]:
            raise RuntimeError("Phase-two plan/source changed; preserve existing execution evidence")
    else:
        state = dict(status="planned", recorded_at=now(), matrix=MATRIX,
                     source_sha256=identity["source_sha256"], code_commit=identity["commit"],
                     controls=["baseline-s0-cuda-v1", "ga-s0-cuda-v1", "gd-a1-s0-cuda-v1"],
                     protocol="Same CUDA device; full parameters; 10 updates/320 visits; 657 questions per run",
                     stages={})
        write_json(path, state)
    if not args.execute:
        print({"status": state["status"], "matrix": MATRIX})
        return
    try:
        execute("verify_results.py", [], DIRECTORY / "frozen-verification.log")
        for row in MATRIX:
            if source_identity()["source_sha256"] != state["source_sha256"]:
                raise RuntimeError("Execution sources changed during the batch")
            run_id = row["run_id"]
            run = ROOT / "study/runs" / run_id
            arguments = ["--method", row["method"], "--seed", str(row["seed"]),
                         "--alpha", str(row["alpha"]), "--gamma", str(row["gamma"]), "--run-id", run_id]
            state.update(status="running", current_run=run_id, updated_at=now())
            state["stages"].setdefault(run_id, {})
            write_json(path, state)
            training = run / "training.json"
            if training.exists():
                if read_json(training)["status"] != "completed_training_reload_checked":
                    raise RuntimeError(f"Unfinished training preserved: {run_id}; inspect before continuing")
            elif (run / "training_started.json").exists():
                raise RuntimeError(f"Failed/interrupted training preserved: {run_id}; no automatic retraining")
            else:
                state["current_stage"] = "training"
                write_json(path, state)
                execute("train_cuda.py", arguments, DIRECTORY / (run_id + "-training.log"))
            state["stages"][run_id]["training"] = "passed"
            state["current_stage"] = "evaluation"
            write_json(path, state)
            execute("evaluate_cuda.py", arguments, DIRECTORY / (run_id + "-evaluation.log"))
            state["stages"][run_id]["evaluation"] = "passed"
            state["current_stage"] = "verification"
            write_json(path, state)
            execute("verify_cuda_run.py", [run_id], DIRECTORY / (run_id + "-verification.log"))
            state["stages"][run_id]["verification"] = "passed"
            write_json(path, state)
            print({"event": "run_completed_verified", "run_id": run_id, "at": now()}, flush=True)
        state.update(status="six_runs_verified_pending_batch_audit", current_run=None,
                     current_stage=None, updated_at=now())
        write_json(path, state)
    except BaseException as exc:
        state.update(status="stopped_on_error", error=str(exc), updated_at=now())
        write_json(path, state)
        raise


if __name__ == "__main__":
    main()
