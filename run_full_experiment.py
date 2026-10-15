"""Serial, source-anchored full-parameter reproduction with recoverable checkpoints."""
from pathlib import Path
import subprocess
import sys
import json
from archive_checkpoint import archive,restore

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"reproduction_full"


def run(script,*args):
    print("STAGE",script,*args,flush=True)
    subprocess.run([sys.executable,"-u",str(ROOT/"run_original.py"),script,*args],cwd=ROOT,check=True)


if __name__=="__main__":
    run("audit_full.py")
    run("evaluate_full.py","baseline","--limit","8")
    run("audit_full.py")
    for method in ("GradDiff","GradAscent"):
        run("train_full.py",method)
        model=OUT/"runs"/method/"model/model.safetensors"
        if method=="GradAscent":
            packed=OUT/"runs"/method/"checkpoint.delta.zip"
            if model.exists() and not packed.exists():archive(model,packed,remove=True)
    for method in ("baseline","GradAscent","GradDiff"):
        if method!="baseline":
            path=OUT/"runs"/method/"model/model.safetensors"
            if not path.exists():restore(OUT/"runs"/method/"checkpoint.delta.zip",path)
        run("evaluate_full.py",method)
    run("report_full.py")
