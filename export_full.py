"""Package completed reproduction and exact checkpoint deltas, then verify it."""
import hashlib
import json
from pathlib import Path
import zipfile
from archive_checkpoint import archive

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"reproduction_full"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()


def main():
    assert json.loads((OUT/"summary.json").read_text())["status"]=="completed"
    for method in ("GradAscent","GradDiff"):
        p=OUT/"runs"/method
        if not (p/"checkpoint.delta.zip").exists():archive(p/"model/model.safetensors",p/"checkpoint.delta.zip")
    root_files=["full_core.py","full_batching.py","memory_gd.py","test_memory_gd.py","train_full.py","evaluate_full.py","streamed_adam.py","run_original.py","run_full_experiment.py",
                "audit_full.py","report_full.py","predict_full.py","archive_checkpoint.py","export_full.py","setup_full.sh",
                "requirements-light.lock","requirements-original-overlay.txt","fetch_light_model.py","fetch_tofu.py","README.md",
                "test_streamed_adam.py","test_mps_transfers.py"]
    files=[ROOT/name for name in root_files]
    files += [p for p in (ROOT/"artifacts/tofu").iterdir() if p.is_file()]
    files += [p for p in (ROOT/"open-unlearning-original").rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
    skip_parts={'legacy_python','diagnostics','probe','scratch'}
    files += [p for p in OUT.rglob('*') if p.is_file() and not(skip_parts&set(p.relative_to(OUT).parts))
              and p.suffix not in {'.log','.safetensors','.bin'} and p.name not in {'partial.json','gpu.lock'}]
    dest=ROOT/"exports/tofu-full-reproduction.zip";dest.parent.mkdir(exist_ok=True)
    manifest=[]
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for p in sorted(set(files)):
            name=p.relative_to(ROOT).as_posix();payload=p.read_bytes()
            z.writestr('tofu-full-reproduction/'+name,payload,compress_type=zipfile.ZIP_STORED if p.name.endswith('.delta.zip') else zipfile.ZIP_DEFLATED)
            manifest.append({'path':name,'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()})
        z.writestr('tofu-full-reproduction/package-manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        for e in manifest:assert hashlib.sha256(z.read('tofu-full-reproduction/'+e['path'])).hexdigest()==e['sha256']
    digest=sha(dest);dest.with_suffix('.zip.sha256').write_text(f'{digest}  {dest.name}\n')
    print(json.dumps({'file':str(dest),'MiB':dest.stat().st_size/2**20,'files':len(manifest),'sha256':digest}))


if __name__=='__main__':main()
