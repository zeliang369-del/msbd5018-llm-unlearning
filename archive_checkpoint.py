"""Lossless safetensors XOR archives against the fixed public base checkpoint.

Original headers and byte positions are preserved, so restoration reproduces
the complete original file SHA256, not merely approximately equal tensors.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "artifacts/model/model.safetensors"
CHUNK = 8 * 2**20


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(CHUNK),b""): h.update(block)
    return h.hexdigest()


def header(path):
    with Path(path).open("rb") as f:
        first=f.read(8); n=struct.unpack("<Q",first)[0]; raw=f.read(n)
    return first+raw,json.loads(raw),8+n


def restore(archive,destination,verify=True):
    destination=Path(destination)
    assert not destination.exists(), "Do not overwrite an existing checkpoint"
    with zipfile.ZipFile(archive) as z, BASE.open("rb") as base, destination.open("wb") as output:
        metadata=json.loads(z.read("metadata.json"))
        assert sha(BASE)==metadata["base_sha256"]
        output.write(z.read("original-header.bin"))
        for chunk in metadata["chunks"]:
            delta=z.read(chunk["member"])
            base.seek(chunk["base_position"]); raw=base.read(len(delta))
            result=np.bitwise_xor(np.frombuffer(delta,dtype=np.uint8),np.frombuffer(raw,dtype=np.uint8)).tobytes()
            output.seek(chunk["original_position"]); output.write(result)
    assert destination.stat().st_size==metadata["original_bytes"]
    if verify: assert sha(destination)==metadata["original_sha256"]
    return metadata


def archive(source,destination,remove=False):
    source,destination=Path(source).resolve(),Path(destination).resolve()
    assert source.exists() and not destination.exists()
    original_header, original, original_start=header(source)
    _,base_header,base_start=header(BASE)
    metadata={"format":"exact-safetensors-XOR-v1","source":str(source.relative_to(ROOT)),
              "base_sha256":sha(BASE),"original_sha256":sha(source),"original_bytes":source.stat().st_size,"chunks":[]}
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z, source.open("rb") as f, BASE.open("rb") as base:
        z.writestr("original-header.bin",original_header)
        ordered=sorted(((name,entry) for name,entry in original.items() if name!="__metadata__"),key=lambda x:x[1]["data_offsets"][0])
        counter=0
        for name,entry in ordered:
            ref=base_header[name]
            assert entry["dtype"]==ref["dtype"] and entry["shape"]==ref["shape"]
            start,end=entry["data_offsets"]
            for offset in range(0,end-start,CHUNK):
                size=min(CHUNK,end-start-offset)
                original_position=original_start+start+offset
                base_position=base_start+ref["data_offsets"][0]+offset
                f.seek(original_position);a=f.read(size);base.seek(base_position);b=base.read(size)
                assert len(a)==len(b)==size
                delta=np.bitwise_xor(np.frombuffer(a,dtype=np.uint8),np.frombuffer(b,dtype=np.uint8)).tobytes()
                member=f"delta/{counter:06d}.bin";counter+=1
                z.writestr(member,delta)
                metadata["chunks"].append({"member":member,"original_position":original_position,"base_position":base_position})
        z.writestr("metadata.json",json.dumps(metadata,indent=2))
    with tempfile.TemporaryDirectory(dir="/private/tmp") as directory:
        restored=Path(directory)/"restored.safetensors"
        restore(destination,restored)
        assert sha(restored)==metadata["original_sha256"]
    metadata["archive_sha256"]=sha(destination)
    metadata["archive_bytes"]=destination.stat().st_size
    metadata["actual_restore_byte_hash_verified"]=True
    if remove: source.unlink()
    destination.with_suffix(destination.suffix+".verification.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps({k:metadata[k] for k in ("original_bytes","archive_bytes","original_sha256","actual_restore_byte_hash_verified")}))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=["archive","restore"]);p.add_argument("source");p.add_argument("destination");p.add_argument("--remove-after-verify",action="store_true");a=p.parse_args()
    if a.action=="archive": archive(a.source,a.destination,a.remove_after_verify)
    else: restore(a.source,a.destination)
