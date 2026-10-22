"""Install the verified official CUDA wheel and fixed experiment dependencies."""
import os
from pathlib import Path
import subprocess
import sys
from cuda_common import file_hash, inventory, write_json
from download_pinned_wheel import DIRECTORY, NAME, SHA256


def main():
    root = Path(__file__).resolve().parent
    wheel = DIRECTORY / NAME
    if not wheel.is_file() or file_hash(wheel) != SHA256:
        raise RuntimeError("Complete the official wheel download and SHA256 verification first")
    # This setting applies only to these child processes, not the user's system proxy.
    environment = dict(os.environ)
    environment["PIP_INDEX_URL"] = "https://pypi.tuna.tsinghua.edu.cn/simple"
    environment["NO_PROXY"] = "*"
    base = [sys.executable, "-m", "pip", "install", "--cache-dir", str(root / ".cache/pip"),
            "--find-links", str(DIRECTORY), "--timeout", "120", "--retries", "2", "--disable-pip-version-check"]
    subprocess.run([*base, str(wheel)], cwd=root, env=environment, check=True)
    subprocess.run([*base, "-r", "requirements-cuda.txt"], cwd=root, env=environment, check=True)
    subprocess.run([sys.executable, "-m", "pip", "check"], cwd=root, env=environment, check=True)
    report = inventory()
    write_json(root / "study/environment/dell-inventory.json", report)
    if not report.get("torch", {}).get("cuda_available"):
        raise RuntimeError("Dependencies installed, but torch.cuda.is_available() is false; see inventory")
    print("Installed dependencies; PyTorch detects CUDA", flush=True)


if __name__ == "__main__":
    main()
