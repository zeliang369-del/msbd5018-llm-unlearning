"""Run identity and resource checks; usable without installing torch."""
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
PARAMETERS = 1_235_814_400
COUNTS = {"forget": 40, "retain": 400, "real_authors": 100, "world_facts": 117}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            h.update(block)
    return h.hexdigest()


def command(*args):
    try:
        p = subprocess.run(args, cwd=ROOT, capture_output=True, timeout=30)
        return {"exit_code": p.returncode, "stdout": p.stdout.decode("utf-8", errors="replace").strip(),
                "stderr": p.stderr.decode("utf-8", errors="replace").strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def source_identity():
    names = ["cuda_common.py", "cuda_runtime.py", "train_cuda.py", "evaluate_cuda.py", "streamed_adam.py",
             "full_core.py", "memory_gd.py", "reproduction_full/config.json", "reproduction_full/evaluation_profile.json"]
    names += [str(p.relative_to(ROOT)).replace("\\", "/") for p in sorted((ROOT / "open-unlearning-original/src").rglob("*.py"))]
    commit = command("git", "rev-parse", "HEAD").get("stdout")
    if not commit or len(commit) != 40:
        raise RuntimeError("Cannot record the Git commit; ensure Git is available")
    return {"commit": commit,
            "working_tree": command("git", "status", "--porcelain").get("stdout"),
            "source_sha256": {name: file_hash(ROOT / name) for name in names}}


def inventory():
    report = {"recorded_at": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
              "python": sys.version, "python_executable": sys.executable,
              "disk": dict(zip(("total", "used", "free"), shutil.disk_usage(ROOT))),
              "gpu_driver": command("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"),
              "packages": {},
              "all_distributions": sorted([{"name": dist.metadata["Name"], "version": dist.version}
                                           for dist in importlib.metadata.distributions()], key=lambda row: row["name"].lower())}
    for name in ("torch", "transformers", "accelerate", "numpy", "datasets", "peft", "safetensors"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    try:
        import psutil
        memory = psutil.virtual_memory()
        report["memory"] = {"total": memory.total, "available": memory.available}
    except ImportError:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            class Memory(ctypes.Structure):
                _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
                    (n, ctypes.c_ulonglong) for n in ("total", "available", "page", "freepage", "virtual", "freevirtual", "extended")]
            memory = Memory()
            memory.length = ctypes.sizeof(memory)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
                report["memory"] = {"total": memory.total, "available": memory.available}
    try:
        import torch
        available = torch.cuda.is_available()
        report["torch"] = {"version": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": available}
        if available:
            report["torch"].update(device=torch.cuda.get_device_name(0),
                                   vram_bytes=torch.cuda.get_device_properties(0).total_memory,
                                   bf16_supported=torch.cuda.is_bf16_supported())
    except (ImportError, OSError, RuntimeError) as exc:
        report["torch_error"] = str(exc)
    try:
        import psutil
        memory = psutil.virtual_memory()
        report["memory_after_torch_import"] = {"total": memory.total, "available": memory.available}
    except ImportError:
        pass
    return report


def add_run_arguments(parser):
    parser.add_argument("--method", choices=["baseline", "GradAscent", "GradDiff"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, help="Must end in run-id beneath study/runs")
    parser.add_argument("--probe", action="store_true")


def run_spec(args):
    import math
    import re
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.run_id):
        raise ValueError("run-id must be a simple name of at most 80 characters")
    if args.seed < 0 or args.seed >= 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    if not all(math.isfinite(x) and x >= 0 for x in (args.alpha, args.gamma)):
        raise ValueError("alpha and gamma must be finite and nonnegative")
    if args.method != "GradDiff" and (args.alpha, args.gamma) != (1, 1):
        raise ValueError("alpha/gamma are GD settings; GA uses the original unweighted loss")
    base = (ROOT / "study/runs").resolve()
    destination = (args.output_dir or base / args.run_id).resolve()
    if not destination.is_relative_to(base) or destination.name != args.run_id or destination == base:
        raise ValueError("output-dir must be beneath study/runs and end with run-id")
    spec = {"run_id": args.run_id, "method": args.method, "seed": args.seed,
            "alpha": args.alpha, "gamma": args.gamma, "probe": args.probe,
            "output_directory": str(destination), "device": "cuda:0", "parameterization": "full_parameter_no_LoRA",
            "protocol": read_json(ROOT / "reproduction_full/config.json"),
            "updates": 3 if args.probe else (0 if args.method == "baseline" else 10),
            "scheduler_horizon_updates": 10, "evaluation_profile": read_json(ROOT / "reproduction_full/evaluation_profile.json"),
            "adaptations": ["Single CUDA/SDPA, BF16 weights, FP32 master/moments on CPU/disk",
                            "Explicit max_steps=10; stop probe after update 3 on the same schedule",
                            "Windows mmap flush uses allocation granularity; optimizer arithmetic unchanged",
                            "CUDA TF32 disabled; no autocast, loss scaling, LoRA, or frozen parameters"]}
    return destination, spec


def claim_run(destination, spec):
    path = destination / "run.json"
    identity = source_identity()
    # Dirty status is informational; exact source hashes bind the run.
    signature = digest({"spec": spec, "commit": identity["commit"], "sources": identity["source_sha256"]})
    if path.exists():
        old = read_json(path)
        if old["signature"] != signature:
            raise ValueError("Existing run has a different configuration or code; choose a new run-id")
    else:
        if destination.exists() and any(destination.iterdir()):
            raise ValueError("Refusing to adopt a nonempty output directory without run.json")
        write_json(path, {"signature": signature, "spec": spec, "code": identity})
    return signature


@contextlib.contextmanager
def gpu_lock():
    """OS advisory lock releases automatically on crashes; no stale lock problem."""
    path = ROOT / "study/cuda-gpu.lock"
    path.parent.mkdir(exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0)
        stream.write(b"0")
        stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def preflight(destination, training):
    report = inventory()
    write_json(destination / ("training_preflight.json" if training else "evaluation_preflight.json"), report)
    if not report.get("torch", {}).get("cuda_available"):
        raise RuntimeError("CUDA unavailable; see the saved preflight report")
    if report["packages"]["transformers"] != "4.45.1" or report["packages"]["accelerate"] != "0.34.2":
        raise RuntimeError("Requires Transformers 4.45.1 and Accelerate 0.34.2")
    if report["packages"]["torch"].split("+")[0] != "2.9.1":
        raise RuntimeError("This CUDA protocol pins PyTorch 2.9.1")
    if not report["torch"]["bf16_supported"]:
        raise RuntimeError("Device does not support BF16")
    memory = report.get("memory_after_torch_import", report.get("memory", {}))
    if training and memory.get("available", 0) < 3 * 2**30:
        raise RuntimeError("Less than 3 GiB available RAM for model loading/CPU staging; close other applications and retry with a fresh run-id")
    if report["disk"]["free"] < (25 if training else 4) * 2**30:
        raise RuntimeError("Insufficient free disk for dense FP32 state and saved checkpoint")
    weight = ROOT / "artifacts/model/model.safetensors"
    if not weight.exists():
        raise FileNotFoundError("Pinned base missing. Run this environment's Python with fetch_light_model.py")
    if file_hash(weight) != "35b0cd9f7db2735538ff78b4879fab61b23b7ca316df27b059585bf656700a8d":
        raise RuntimeError("Pinned base SHA256 mismatch")
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        if file_hash(ROOT / "artifacts/model" / name) != file_hash(ROOT / "reproduction_full/runs/GradAscent/model" / name):
            raise RuntimeError("Downloaded tokenizer/generation settings differ from the frozen experiment: " + name)
    # Check the fixed datasets and frozen evidence before any inference/training.
    from verify_results import verify
    verify()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This protocol requires exactly one device")
    return report


if __name__ == "__main__":
    report = inventory()
    write_json(ROOT / "study/environment/dell-inventory.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
