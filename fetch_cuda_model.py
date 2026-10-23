"""Fetch the unchanged pinned model, optionally through a public mirror."""
import argparse
import json
import subprocess
from cuda_common import file_hash, write_json
from fetch_light_model import DEST, REPO, REV, HASH, FILES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", choices=["https://huggingface.co", "https://hf-mirror.com"], default="https://huggingface.co")
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = DEST / name
        if not target.exists():
            part = DEST / (name + ".part")
            command = ["curl.exe", "-4", "-fL", "--continue-at", "-", "--connect-timeout", "20",
                       "--max-time", "1800", "--speed-limit", "1024", "--speed-time", "60", "--retry", "2",
                       "--output", str(part), f"{args.endpoint}/{REPO}/resolve/{REV}/{name}"]
            subprocess.run(command, check=True)
            if name.endswith(".json"):
                json.loads(part.read_text(encoding="utf-8"))
            if name == "model.safetensors" and file_hash(part) != HASH:
                raise RuntimeError("Pinned model SHA256 mismatch; preserving the partial file for inspection")
            part.replace(target)
        if name == "model.safetensors" and file_hash(target) != HASH:
            raise RuntimeError("Existing model SHA256 mismatch")
    write_json(DEST / "cuda-download-provenance.json", {"endpoint": args.endpoint, "model_id": REPO,
               "model_revision": REV, "files": {name: file_hash(DEST / name) for name in FILES},
               "model_weight_matches_pinned_sha256": True})
    print("Pinned model SHA256 verified; download provenance saved", flush=True)


if __name__ == "__main__":
    main()
