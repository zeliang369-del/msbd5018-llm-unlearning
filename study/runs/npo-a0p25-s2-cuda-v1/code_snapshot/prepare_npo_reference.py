"""Prepare a fixed-checkpoint NPO reference-NLL cache; never overwrites output."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    import torch
    from cuda_runtime import configure, load_cuda
    from full_core import BASE, QA, file_hash, read_rows
    from npo_data import reference_samples
    from npo_reference import build_reference_cache, save_reference_cache

    configure(0)
    model, tokenizer, template = load_cuda(args.checkpoint)
    rows = read_rows("forget01.json")
    full_count = len(rows)
    if full_count != 40:
        raise RuntimeError(f"Expected 40 Forget samples, found {full_count}")
    scope = "full" if not args.limit else "limit"
    if args.limit:
        rows = rows[:args.limit]
    dataset = QA(rows, tokenizer, template)
    length = 192
    samples = reference_samples(dataset, tokenizer, length)
    metadata = {
        "reference_checkpoint_sha256": file_hash(args.checkpoint / "model.safetensors"),
        "tokenizer_hash": file_hash(args.checkpoint / "tokenizer.json"),
        "data_protocol": "full_core_official_item_v1_static_padding_192",
        "scope": scope,
        "forget_count": len(rows),
        "full_forget_count": full_count,
        "reference_batch_size": 4,
    }
    cache = build_reference_cache(model, samples, metadata, batch_size=4)
    save_reference_cache(args.output, cache)
    print(f"Wrote verified NPO reference cache: {args.output} ({len(samples)} samples)")


if __name__ == "__main__":
    main()
