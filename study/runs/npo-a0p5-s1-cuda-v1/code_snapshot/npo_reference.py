"""Metadata-checked per-sample NPO reference-NLL cache."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch

from npo_cuda import compute_nll_sum


LOSS_PROTOCOL = "shifted_token_nll_sum_ignore_index_-100"


class ReferenceNLLCache:
    def __init__(self, cache):
        self.cache = cache

    def lookup(self, sample_ids, samples):
        ids = list(sample_ids.detach().cpu().reshape(-1).tolist()) if torch.is_tensor(sample_ids) else list(sample_ids)
        if len(ids) != len(set(map(str, ids))):
            raise ValueError("Duplicate Forget sample_id values")
        values = self.cache.get("values", {})
        if len(ids) != len(samples["labels"]):
            raise ValueError("Forget sample_id count does not match batch size")
        result = []
        for index, sample_id in enumerate(ids):
            key = str(sample_id)
            if key not in values:
                raise ValueError(f"Unknown Forget sample_id: {sample_id}")
            sample = {name: samples[name][index:index + 1] for name in ("input_ids", "attention_mask", "labels")}
            if values[key].get("fingerprint") != sample_fingerprint(sample):
                raise ValueError(f"Reference cache sample content mismatch: {sample_id}")
            result.append(values[key]["nll_sum"])
        return torch.tensor(result, dtype=torch.float32, device=samples["labels"].device)


def sample_fingerprint(sample):
    digest = hashlib.sha256()
    for key in ("input_ids", "attention_mask", "labels"):
        value = sample[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(repr(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def build_reference_cache(model, samples, metadata, batch_size=1):
    """Build {sample_id: NLL sum + fingerprint} using eval/no_grad reference inference."""
    required = {"reference_checkpoint_sha256", "tokenizer_hash", "data_protocol"}
    missing = required - set(metadata)
    if missing:
        raise ValueError("Missing reference cache metadata: " + ", ".join(sorted(missing)))
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Reference batch_size must be a positive integer")
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    values = {}
    try:
        with torch.no_grad():
            entries = list(samples.items())
            for start in range(0, len(entries), batch_size):
                chunk = entries[start:start + batch_size]
                model_inputs = {name: torch.cat([sample[name] for _, sample in chunk], dim=0).to(device)
                                for name in ("input_ids", "attention_mask", "labels")}
                nll, outputs = compute_nll_sum(model, model_inputs)
                if nll.numel() != len(chunk) or not torch.isfinite(nll).all():
                    raise ValueError("Invalid reference NLL batch")
                for (sample_id, sample), value in zip(chunk, nll.detach().cpu().tolist()):
                    values[str(sample_id)] = {"nll_sum": float(value), "fingerprint": sample_fingerprint(sample)}
                del nll, outputs, model_inputs
    finally:
        model.train(was_training)
    return {"version": 1, "loss_protocol": LOSS_PROTOCOL, "metadata": dict(metadata), "values": values}


def save_reference_cache(path, cache):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite reference cache: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=path.name + ".", delete=False, encoding="utf-8") as stream:
        temporary = Path(stream.name); stream.write(json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n"); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)

def cache_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def load_reference_cache(path, samples, expected_metadata):
    cache = json.loads(Path(path).read_text(encoding="utf-8"))
    if cache.get("version") != 1 or cache.get("loss_protocol") != LOSS_PROTOCOL:
        raise ValueError("Reference cache loss protocol/version mismatch")
    if cache.get("metadata") != expected_metadata:
        raise ValueError("Reference cache metadata mismatch")
    values = cache.get("values", {})
    expected_ids = {str(key) for key in samples}
    if set(values) != expected_ids:
        raise ValueError("Reference cache sample IDs mismatch")
    for sample_id, sample in samples.items():
        if values[str(sample_id)].get("fingerprint") != sample_fingerprint(sample):
            raise ValueError(f"Reference cache sample content mismatch: {sample_id}")
    return ReferenceNLLCache(cache)
