"""Resume the official CUDA wheel in bounded ranges and verify its official hash."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
URL = "https://download-r2.pytorch.org/whl/cu128/torch-2.9.1%2Bcu128-cp312-cp312-win_amd64.whl"
SHA256 = "3a01f0b64c10a82d444d9fd06b3e8c567b1158b76b2764b8f51bfd8f535064b0"
DIRECTORY = ROOT / ".cache/wheels"
NAME = "torch-2.9.1+cu128-cp312-cp312-win_amd64.whl"


def opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, expected_hash, directory, name, workers=8):
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / name
    if final.exists():
        if hash_file(final) != expected_hash:
            raise RuntimeError("Existing wheel SHA256 mismatch")
        print("Verified existing wheel", flush=True)
        return
    prefix = directory / (name + ".part")
    size = prefix.stat().st_size if prefix.exists() else 0
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "project-pinned-download/1"})
    with opener().open(request, timeout=30) as response:
        content_range = response.headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes 0-0/(\d+)", content_range)
        if response.status != 206 or not match:
            raise RuntimeError("Official server did not provide a valid byte range")
        total = int(match[1])
    if size > total:
        raise RuntimeError("Partial prefix exceeds official file size")
    part_dir = directory / ("torch-ranges" if name == NAME else "model-ranges")
    part_dir.mkdir(exist_ok=True)
    manifest_file = part_dir / "manifest.json"
    manifest = {"url": url, "sha256": expected_hash, "total_bytes": total, "prefix_bytes": size, "chunk_bytes": 64 * 2**20}
    if manifest_file.exists() and json.loads(manifest_file.read_text()) != manifest:
        raise RuntimeError("Range manifest changed; preserve downloads and investigate")
    manifest_file.write_text(json.dumps(manifest, indent=2))
    ranges = [(start, min(total - 1, start + manifest["chunk_bytes"] - 1)) for start in range(size, total, manifest["chunk_bytes"])]

    def transfer(bounds):
        start, end = bounds
        path = part_dir / f"{start}-{end}.part"
        length = end - start + 1
        for attempt in range(4):
            offset = path.stat().st_size if path.exists() else 0
            if offset == length:
                return length
            if offset > length:
                raise RuntimeError("Range part exceeds expected size")
            req = urllib.request.Request(url, headers={"Range": f"bytes={start + offset}-{end}", "User-Agent": "project-pinned-download/1"})
            try:
                request_started = time.monotonic()
                with opener().open(req, timeout=45) as response:
                    expected = f"bytes {start + offset}-{end}/{total}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise RuntimeError("Byte-range response mismatch")
                    with path.open("ab") as output:
                        remaining = length - offset
                        while remaining:
                            block = response.read(min(64 * 1024, remaining))
                            if not block:
                                raise OSError("Unexpected end of download")
                            output.write(block)
                            remaining -= len(block)
                            if remaining and time.monotonic() - request_started > 120:
                                raise TimeoutError("Reconnecting a slow range while preserving downloaded bytes")
                return length
            except (OSError, TimeoutError):
                if attempt == 3:
                    raise
                time.sleep(2)

    print(json.dumps({"event": "resume", **manifest, "workers": workers}), flush=True)
    complete = size
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(transfer, bounds) for bounds in ranges]):
            complete += future.result()
            print(json.dumps({"event": "range_complete", "verified_length_bytes": complete, "total_bytes": total}), flush=True)
    joined = directory / (name + ".assembling")
    with joined.open("wb") as output:
        for path in ([prefix] if size else []) + [part_dir / f"{start}-{end}.part" for start, end in ranges]:
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 2**20), b""):
                    output.write(block)
    actual = hash_file(joined)
    if joined.stat().st_size != total or actual != expected_hash:
        raise RuntimeError(f"Official wheel checksum mismatch: {actual}")
    joined.replace(final)
    print(json.dumps({"event": "pinned_file_verified", "sha256": actual, "bytes": total, "file": str(final)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", choices=["torch", "model"], default="torch")
    args = parser.parse_args()
    if args.asset == "torch":
        download(URL, SHA256, DIRECTORY, NAME)
    else:
        from fetch_light_model import DEST, REPO, REV, HASH
        url = f"https://hf-mirror.com/{REPO}/resolve/{REV}/model.safetensors"
        download(url, HASH, DEST, "model.safetensors")


if __name__ == "__main__":
    main()
