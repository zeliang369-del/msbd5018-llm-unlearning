"""Original 657-question evaluation on CUDA, isolated from frozen Mac results."""
import argparse
import math
import time
import traceback
from cuda_common import (ROOT, COUNTS, add_run_arguments, run_spec, claim_run, preflight,
                         gpu_lock, read_json, write_json, digest, file_hash)


def evaluate(args, destination, spec, signature):
    preflight(destination, training=False)
    import torch
    from cuda_runtime import configure, load_cuda
    from full_core import BASE, SOURCES, PROMPT_DATE, read_rows, official_item, aggregate
    from data.collators import DataCollatorForSupervisedDataset
    from evals.metrics.utils import evaluate_probability, eval_text_similarity
    from omegaconf import OmegaConf
    configure(args.seed)
    if args.method == "baseline":
        model_path = BASE
    else:
        training = read_json(destination / "training.json")
        if training["signature"] != signature:
            raise ValueError("Training signature mismatch")
        model_path = destination / "model"
        if file_hash(model_path / "model.safetensors") != training["checkpoint_sha256"]:
            raise ValueError("Saved checkpoint hash mismatch")
    output_dir = destination / ("reload_check" if args.limit else "evaluation")
    provenance = {"run_signature": signature, "checkpoint_sha256": file_hash(model_path / "model.safetensors"),
                  "profile": spec["evaluation_profile"], "limit_per_domain": args.limit, "prompt_date": PROMPT_DATE}
    eval_signature = digest(provenance)
    output = output_dir / "evaluation.json"
    if output.exists():
        if read_json(output)["signature"] != eval_signature:
            raise ValueError("Existing evaluation signature mismatch")
        print("Verified saved evaluation; no fresh inference claimed", output)
        return
    partial = output_dir / "partial.json"
    cache = read_json(partial) if partial.exists() else {"signature": eval_signature, "domains": {}}
    if cache["signature"] != eval_signature:
        raise ValueError("Partial evaluation signature mismatch")
    start = time.monotonic()
    model, tokenizer, template = load_cuda(model_path)
    model.eval()
    model.generation_config.eos_token_id = read_json(BASE / "generation_config.json")["eos_token_id"]
    class Inference:
        def __init__(self):
            self.device = model.device
        def __call__(self, **kwargs):
            kwargs.pop("labels", None)
            result = model(**kwargs)
            result.logits = result.logits.float()
            return result
        def generate(self, *values, **kwargs):
            return model.generate(*values, **kwargs)
    wrapped = Inference()
    fixture = read_rows("forget01.json")[0]
    item = official_item(tokenizer, template, fixture["question"], fixture["answer"])
    batch = {k: v.unsqueeze(0).to(model.device) for k, v in item.items()}
    with torch.inference_mode():
        logits = model(**batch).logits
        if not torch.equal(logits.float(), wrapped(**batch).logits):
            raise RuntimeError("Removing unused model.loss changed logits")
    del logits, batch
    write_json(output_dir / "forward-parity.json", {"status": "passed"})
    probability_collator = DataCollatorForSupervisedDataset(tokenizer)
    generation_collator = DataCollatorForSupervisedDataset(tokenizer, padding_side="left")
    official_reference = read_json(ROOT / "artifacts/tofu/full-reference.json")
    profile = spec["evaluation_profile"]
    torch.cuda.reset_peak_memory_stats()
    for domain, source in SOURCES.items():
        source_rows = read_rows(source)
        indices = list(range(min(args.limit, len(source_rows)) if args.limit else len(source_rows)))
        saved = cache["domains"].setdefault(domain, [])
        done = {row["source_index"] for row in saved}
        if len(done) != len(saved) or not done.issubset(indices):
            raise ValueError("Invalid/duplicate partial source indices")
        pending = [index for index in indices if index not in done]
        for offset in range(0, len(pending), profile["batch_size"]):
            chunk = pending[offset:offset + profile["batch_size"]]
            rows = [source_rows[index] for index in chunk]
            fields = [("answer", 0), ("paraphrased_answer", 0)] + [("perturbed_answer", i) for i in range(len(rows[0]["perturbed_answer"]))]
            scores = {}
            for key, answer_index in fields:
                if key == "paraphrased_answer" and all(row.get(key, row["answer"]) == row["answer"] for row in rows):
                    scores[(key, answer_index)] = scores[("answer", 0)]
                    continue
                answers = [row[key][answer_index] if key == "perturbed_answer" else row.get(key, row["answer"]) for row in rows]
                entries = [official_item(tokenizer, template, row["question"], answer) for row, answer in zip(rows, answers)]
                scores[(key, answer_index)] = evaluate_probability(wrapped, probability_collator(entries))
            prompts = [official_item(tokenizer, template, row["question"], row["answer"], True) for row in rows]
            generated = eval_text_similarity(wrapped, tokenizer, generation_collator(prompts), OmegaConf.create({
                "do_sample": False, "top_p": None, "temperature": None, "max_new_tokens": profile["max_new_tokens"], "use_cache": True}))
            for j, (index, row) in enumerate(zip(chunk, rows)):
                orig, para = scores[("answer", 0)][j], scores[("paraphrased_answer", 0)][j]
                wrong = [scores[("perturbed_answer", i)][j] for i in range(len(row["perturbed_answer"]))]
                result = {"source_index": index, "question": row["question"], "reference_answer": row["answer"],
                          "answer_nll": orig["avg_loss"], "answer_probability": orig["prob"],
                          "paraphrased_nll": para["avg_loss"], "paraphrased_probability": para["prob"],
                          "wrong_nlls": [w["avg_loss"] for w in wrong], "wrong_probabilities": [w["prob"] for w in wrong], **generated[j]}
                if not all(math.isfinite(v) for v in [result["answer_nll"], result["paraphrased_nll"], *result["wrong_nlls"], result["rougeL_recall"]]):
                    raise FloatingPointError("Non-finite evaluation value")
                from full_core import PREFIX
                expected = official_reference[f"{PREFIX[domain]}_Q_A_ROUGE"]["value_by_index"][str(index)]
                if result["input"] != expected["input"] or result["ground_truth"] != expected["ground_truth"]:
                    raise RuntimeError("Prompt/target text differs from the pinned official evaluation")
                saved.append(result)
            write_json(partial, cache)
            print({"domain": domain, "completed": len(saved), "total": len(indices)}, flush=True)
        saved.sort(key=lambda row: row["source_index"])
    counts = {domain: len(rows) for domain, rows in cache["domains"].items()}
    if not args.limit and counts != COUNTS:
        raise RuntimeError("Formal evaluation must include all 657 questions")
    # Small checks deliberately have no FQ/MU headline metrics.
    logs = aggregate(cache["domains"]) if not args.limit else None
    metrics = {key: logs[key]["agg_value"] for key in ("forget_quality", "model_utility", "forget_truth_ratio", "forget_Q_A_ROUGE")} if logs else None
    calibration = {}
    if args.method == "baseline":
        reference = official_reference
        for domain, rows in cache["domains"].items():
            # Match the same reference inputs used by the frozen Mac calibration.
            from full_core import PREFIX
            nll = reference[f"{PREFIX[domain]}_Q_A_Prob"]["value_by_index"]
            rouge = reference[f"{PREFIX[domain]}_Q_A_ROUGE"]["value_by_index"]
            calibration[domain] = {"questions": len(rows),
                "processed_text_match": all(r["input"] == rouge[str(r["source_index"])]["input"] and r["ground_truth"] == rouge[str(r["source_index"])]["ground_truth"] for r in rows),
                "NLL_MAE": sum(abs(r["answer_nll"] - nll[str(r["source_index"])]["avg_loss"]) for r in rows) / len(rows),
                "ROUGE_MAE": sum(abs(r["rougeL_recall"] - rouge[str(r["source_index"])]["rougeL_recall"]) for r in rows) / len(rows)}
    write_json(output, {"status": "completed", "scope": "reload check only" if args.limit else "CUDA control: all 657 questions",
                       "signature": eval_signature, "provenance": provenance, "counts": counts, "domains": cache["domains"],
                       "metrics": metrics, "baseline_calibration": calibration, "elapsed_seconds": time.monotonic() - start,
                       "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30})
    if logs:
        write_json(output_dir / "TOFU_EVAL.json", logs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)
    parser.add_argument("--limit", type=int, default=0, help="Per domain, 0 means formal 657-question evaluation")
    args = parser.parse_args()
    if args.limit < 0 or (args.probe and args.limit == 0):
        parser.error("Probe evaluation requires a positive --limit")
    destination, spec = run_spec(args)
    with gpu_lock():
        signature = claim_run(destination, spec)
        try:
            evaluate(args, destination, spec, signature)
        except BaseException as exc:
            write_json(destination / ("reload_check_failure.json" if args.limit else "evaluation_failure.json"),
                       {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
            raise


if __name__ == "__main__":
    main()
