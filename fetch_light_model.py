"""Download/verify the fixed public 1B checkpoint without CUDA dependencies."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "artifacts/model"
REPO = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
REV = "88e31200b97e4c0c04ae0d2f0b591f427046d192"
HASH = "35b0cd9f7db2735538ff78b4879fab61b23b7ca316df27b059585bf656700a8d"
FILES = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "model.safetensors"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", default=urllib.request.getproxies().get("https"))
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = DEST/name
        if not target.exists():
            part = DEST/(name+".part")
            cmd = ["curl", "-fL", "--show-error", "--connect-timeout", "20", "--max-time", "1200"]
            if args.proxy:
                cmd += ["--proxy", args.proxy]
            if name == "model.safetensors":
                cmd += ["--continue-at", "-"]
            subprocess.run(cmd+[f"https://huggingface.co/{REPO}/resolve/{REV}/{name}", "-o", str(part)], check=True)
            part.replace(target)
    digest = hashlib.sha256()
    with (DEST/"model.safetensors").open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""):
            digest.update(chunk)
    assert digest.hexdigest() == HASH, "Model checksum mismatch"
    for name in FILES:
        if name.endswith(".json"):
            json.loads((DEST/name).read_text())
    print("Pinned model files and weight SHA256 verified")


if __name__ == "__main__":
    main()
