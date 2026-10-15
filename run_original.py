"""Select the original text-model framework without changing the legacy venv."""
import os
from pathlib import Path
import runpy
import sys
import fcntl

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"reproduction_full/legacy_python"))
os.environ["LIGHT_UPSTREAM"]=str(ROOT/"open-unlearning-original")
os.environ["ACCELERATE_MIXED_PRECISION"]="no"
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO","0.8")
if __name__=="__main__":
    assert len(sys.argv)>1,"Provide a local experiment script to execute"
    target=ROOT/sys.argv[1]
    assert target.parent==ROOT and target.suffix==".py"
    sys.argv=[str(target),*sys.argv[2:]]
    if target.name in ("train_full.py","evaluate_full.py","predict_full.py","test_mps_transfers.py","test_memory_gd.py"):
        with (ROOT/"reproduction_full/gpu.lock").open("w") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            runpy.run_path(str(target),run_name="__main__")
    else:runpy.run_path(str(target),run_name="__main__")
