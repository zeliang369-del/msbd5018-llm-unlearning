"""Preserve exactly the hashed sources used by an existing isolated run."""
import argparse
from pathlib import Path
from cuda_common import ROOT, file_hash, read_json


def capture(directory, root=ROOT):
    root = Path(root).resolve()
    directory = Path(directory).resolve()
    if not directory.is_relative_to((root / "study/runs").resolve()):
        raise ValueError("Only isolated study/runs directories are supported")
    run = read_json(directory / "run.json")
    for name, expected in run["code"]["source_sha256"].items():
        source = (root / name).resolve()
        destination = (directory / "code_snapshot" / name).resolve()
        if not source.is_relative_to(root) or not destination.is_relative_to(directory / "code_snapshot"):
            raise ValueError("Invalid source path")
        if destination.exists():
            if file_hash(destination) != expected:
                raise ValueError("Existing snapshot hash mismatch: " + name)
        else:
            if file_hash(source) != expected:
                raise ValueError("Current code differs from the run; cannot invent a historical snapshot: " + name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
            if file_hash(destination) != expected:
                raise ValueError("Source changed while making snapshot: " + name)
    print("Verified exact source snapshot:", directory.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args()
    capture(ROOT / "study/runs" / args.run_id)
