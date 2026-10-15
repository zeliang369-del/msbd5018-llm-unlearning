"""Fetch authentic pinned TOFU data and retain-reference logs with integrity checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts" / "tofu"
REVISION = "324592d84ae4f482ac7249b9285c2ecdb53e3a68"
REFERENCE_REVISION = "f31717fe522c2a6725ca0024980f51169a084ca1"
FILES = ["forget01.json", "forget01_perturbed.json", "retain99.json", "retain_perturbed.json", "real_authors_perturbed.json", "world_facts_perturbed.json"]


def get(url, destination, proxy):
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = ["curl", "-fL", "--silent", "--show-error", "--connect-timeout", "15", "--max-time", "120", "--retry", "2"]
    if proxy:
        command += ["--proxy", proxy]
    subprocess.run(command + ["-o", str(destination), url], check=True)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", default=urllib.request.getproxies().get("https"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tree_path = OUT / "source-tree.json"
    get(f"https://huggingface.co/api/datasets/locuslab/TOFU/tree/{REVISION}", tree_path, args.proxy)
    entries = {x["path"]: x for x in json.loads(tree_path.read_text())}

    def download(name):
        path = OUT / name
        expected = entries[name]
        if not path.exists():
            get(f"https://huggingface.co/datasets/locuslab/TOFU/resolve/{REVISION}/{name}", path, args.proxy)
        payload = path.read_bytes()
        actual = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
        assert actual == expected["oid"], f"Git blob hash mismatch for {name}"
        rows = read_rows(path)
        print(f"Verified {name}: {len(rows)} rows", flush=True)
        return {"file": name, "rows": len(rows), "git_blob_sha1": actual, "sha256": hashlib.sha256(payload).hexdigest()}

    with ThreadPoolExecutor(max_workers=3) as pool:
        verified = list(pool.map(download, FILES))
    # Only fetch the matched 1B retain99 reference, not every model in the hub.
    metadata = OUT / "reference-metadata.json"
    get(f"https://huggingface.co/api/datasets/open-unlearning/eval/revision/{REFERENCE_REVISION}", metadata, args.proxy)
    ref_meta = json.loads(metadata.read_text())
    assert ref_meta["sha"] == REFERENCE_REVISION
    matches = [x["rfilename"] for x in ref_meta["siblings"] if "tofu_Llama-3.2-1B-Instruct_retain99/TOFU_EVAL.json" in x["rfilename"]]
    assert len(matches) == 1, matches
    ref_path = OUT / "retain99-reference.json"
    if not ref_path.exists():
        get(f"https://huggingface.co/datasets/open-unlearning/eval/resolve/{ref_meta['sha']}/{matches[0]}", ref_path, args.proxy)
    reference = json.loads(ref_path.read_text())
    assert len(reference["forget_truth_ratio"]["value_by_index"]) == 40
    assert hashlib.sha256(ref_path.read_bytes()).hexdigest() == 'aa6a504c44487fe5174c1828e4c3bfe997de89856c95cba721dc833bbb37aee3'
    full_matches = [x['rfilename'] for x in ref_meta['siblings'] if 'tofu_Llama-3.2-1B-Instruct_full/evals_forget01/TOFU_EVAL.json' in x['rfilename']]
    assert len(full_matches)==1,full_matches
    full_path=OUT/'full-reference.json'
    if not full_path.exists():
        get(f"https://huggingface.co/datasets/open-unlearning/eval/resolve/{REFERENCE_REVISION}/{full_matches[0]}",full_path,args.proxy)
    assert hashlib.sha256(full_path.read_bytes()).hexdigest()=='558fdd814fe88bfcd29b29ff5d14ba8fbc946c86fbe8b10eba867acb98719155'
    manifest = {"dataset_id": "locuslab/TOFU", "dataset_revision": REVISION, "files": verified,
        "reference_dataset": "open-unlearning/eval", "reference_revision": ref_meta["sha"],
        "reference_path": matches[0], "reference_sha256": hashlib.sha256(ref_path.read_bytes()).hexdigest(),
        "forget_split": "forget01", "retain_split": "retain99", "data_source": "Official data; no fabricated QA"}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Real TOFU and matched oracle-reference logs ready.", flush=True)


if __name__ == "__main__":
    main()
