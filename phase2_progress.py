"""Read the phase-two ledger and partial evidence without importing a model."""
from cuda_common import ROOT, read_json


def progress():
    state = read_json(ROOT / "study/phase2/execution.json")
    result = {"status": state["status"], "verified_extensions": sum(
        row.get("verification") == "passed" for row in state["stages"].values()),
        "total_extensions": len(state["matrix"]), "current_run": state.get("current_run"),
        "stage": state.get("current_stage")}
    if state.get("current_run"):
        directory = ROOT / "study/runs" / state["current_run"]
        resources = directory / "resources_partial.json"
        if resources.exists():
            rows = read_json(resources)
            if rows:
                result["last_update"] = rows[-1]
        partial = directory / "evaluation/partial.json"
        if partial.exists():
            result["evaluated_questions"] = sum(len(rows) for rows in read_json(partial)["domains"].values())
        for name in ("training_failure.json", "evaluation_failure.json", "reload_check_failure.json"):
            path = directory / name
            if path.exists():
                result[name] = read_json(path)["error"]
    if state.get("error"):
        result["error"] = state["error"]
    print(result)


if __name__ == "__main__":
    progress()
