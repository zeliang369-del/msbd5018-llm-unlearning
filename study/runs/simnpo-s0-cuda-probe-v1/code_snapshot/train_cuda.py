"""Isolated full-parameter CUDA GA/GD; failed runs require a new run-id."""
import argparse
import gc
import math
from pathlib import Path
import shutil
import time
import traceback
import uuid
from cuda_common import (ROOT, PARAMETERS, add_run_arguments, run_spec, claim_run,
                         preflight, gpu_lock, write_json, file_hash)


def train(args, destination, spec, signature):
    preflight(destination, training=True)
    import torch
    import psutil
    from transformers import TrainingArguments, TrainerCallback
    from safetensors import safe_open
    from cuda_runtime import configure, load_cuda, CudaGradDiff, CudaStreamedAdamW
    from simnpo_cuda import CudaSimNPO
    from full_core import BASE, QA, TracedPairs, read_rows
    from memory_gd import StaticTrainingCollator
    from trainer.unlearn.grad_ascent import GradAscent
    configure(args.seed)
    started = time.monotonic()
    model, tokenizer, template = load_cuda()
    if sum(p.numel() for p in model.parameters()) != PARAMETERS or not all(p.requires_grad for p in model.parameters()):
        raise RuntimeError("Full 1,235,814,400-parameter training required")
    dataset = TracedPairs(QA(read_rows("forget01.json"), tokenizer, template),
                          QA(read_rows("retain99.json"), tokenizer, template))
    if len(dataset) != 40 or len(dataset.retain) != 3960:
        raise RuntimeError("Unexpected training split")
    cfg = spec["protocol"]
    settings = dict(output_dir=str(destination), per_device_train_batch_size=4,
                    gradient_accumulation_steps=8, num_train_epochs=10, max_steps=10,
                    learning_rate=1e-5, weight_decay=.01, warmup_steps=1, lr_scheduler_type="linear",
                    optim="adamw_torch", adam_beta1=.9, adam_beta2=.999, adam_epsilon=1e-8,
                    max_grad_norm=1, bf16=False, fp16=False, tf32=False,
                    gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                    remove_unused_columns=False, dataloader_pin_memory=False, dataloader_num_workers=0,
                    report_to="none", logging_steps=1, logging_nan_inf_filter=False,
                    save_strategy="no", eval_strategy="no", seed=args.seed, data_seed=args.seed,
                    accelerator_config={"gradient_accumulation_kwargs": {"sync_with_dataloader": False}})
    classes = {"GradAscent": GradAscent, "GradDiff": CudaGradDiff, "SimNPO": CudaSimNPO}
    cls = classes[args.method]
    method_args = {}
    if args.method in ("GradDiff", "SimNPO"):
        method_args = dict(alpha=args.alpha, gamma=args.gamma, retain_loss_type="NLL")
    if args.method == "SimNPO":
        method_args.update(beta=args.beta, delta=args.delta)
    trainer = cls(model=model, tokenizer=tokenizer, train_dataset=dataset,
                  args=TrainingArguments(**settings), data_collator=StaticTrainingCollator(tokenizer, cfg["training_padded_length"]),
                  **method_args)
    trainer.create_optimizer()
    groups = [{k: v for k, v in group.items()} for group in trainer.optimizer.param_groups]
    scratch = destination / ("scratch-" + uuid.uuid4().hex[:8])
    optimizer = None
    rows = []
    watched = model.model.layers[0].self_attn.q_proj.weight
    before = watched[:64, :64].detach().cpu().clone()

    class Evidence(TrainerCallback):
        def on_log(self, arguments, state, control, logs=None, **kwargs):
            for key in ("loss", "grad_norm"):
                if logs and key in logs and not math.isfinite(float(logs[key])):
                    raise FloatingPointError("Non-finite training " + key)
            write_json(destination / "training_log_partial.json", state.log_history)

        def on_step_end(self, arguments, state, control, **kwargs):
            torch.cuda.synchronize()
            norm = optimizer.last_gradient_norm
            if not math.isfinite(norm):
                raise FloatingPointError("Non-finite gradient norm")
            if state.global_step == 1:
                if any(optimizer.last_applied_lrs):
                    raise RuntimeError("Expected zero-rate warmup first update")
                with safe_open(str(BASE / "model.safetensors"), framework="pt", device="cpu") as reference:
                    for name, parameter in model.named_parameters():
                        if not torch.equal(parameter.detach().cpu(), reference.get_tensor(name).to(dtype=parameter.dtype)):
                            raise RuntimeError("Zero-rate warmup changed " + name)
                write_json(destination / "zero-rate-invariance.json", {"status": "passed", "all_parameters_checked": True})
            changed = int((before != watched[:64, :64].detach().cpu()).sum())
            row = {"step": state.global_step, "epoch": state.epoch, "optimizer_steps": optimizer.steps,
                   "applied_lrs": optimizer.last_applied_lrs, "gradient_norm": norm,
                   "changed_q_proj_slice_elements": changed,
                   "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                   "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                   "rss_gib": psutil.Process().memory_info().rss / 2**30,
                   "available_ram_gib": psutil.virtual_memory().available / 2**30,
                   "free_disk_gib": shutil.disk_usage(destination).free / 2**30}
            rows.append(row)
            write_json(destination / "resources_partial.json", rows)
            print(row, flush=True)
            if row["free_disk_gib"] < 1:
                raise RuntimeError("Scratch disk reserve exhausted")
            if state.global_step >= spec["updates"]:
                control.should_training_stop = True

    try:
        optimizer = CudaStreamedAdamW(groups, list(model.named_parameters()), scratch,
                                     capture_gradients=False, reserve_bytes=3 * 2**30)
        trainer.optimizer = optimizer
        def clip(parameters, max_norm, norm_type=2, **kwargs):
            trainer.accelerator.unscale_gradients()
            return optimizer.clip_grad_norm(max_norm, norm_type)
        trainer.accelerator.clip_grad_norm_ = clip
        trainer.add_callback(Evidence())
        torch.cuda.reset_peak_memory_stats()
        result = trainer.train()
        steps = spec["updates"]
        visits = round(trainer.state.epoch * len(dataset))
        if trainer.state.global_step != steps or optimizer.steps != steps or visits != steps * 32:
            raise RuntimeError(f"Budget mismatch: Trainer={trainer.state.global_step}, optimizer={optimizer.steps}, visits={visits}")
        if len(dataset.trace) < visits:
            raise RuntimeError("Incomplete sampling trace")
        if rows[-1]["changed_q_proj_slice_elements"] == 0:
            raise RuntimeError("No observed parameter changes after nonzero-rate updates")
        for row in trainer.state.log_history:
            for key in ("loss", "grad_norm"):
                if key in row and not math.isfinite(float(row[key])):
                    raise FloatingPointError("Non-finite " + key)
        optimizer.cleanup()
        gc.collect()
        torch.cuda.empty_cache()
        model.save_pretrained(destination / "model", safe_serialization=True, max_shard_size="5GB")
        tokenizer.save_pretrained(destination / "model")
        checkpoint_hash = file_hash(destination / "model/model.safetensors")
        trainer.save_state()
        summary = {"status": "trained_pending_reload_evaluation", "signature": signature, "spec": spec,
                   "settings": settings, "steps": steps, "visits": visits, "epochs": trainer.state.epoch,
                   "trainable_parameters": PARAMETERS, "FP32_optimizer_state_bytes": PARAMETERS * 12,
                   "sampling_trace": dataset.trace[:visits], "prefetched_unused_pairs": dataset.trace[visits:],
                   "resources": rows, "log_history": trainer.state.log_history, "trainer_metrics": result.metrics,
                   "checkpoint_sha256": checkpoint_hash, "elapsed_seconds": time.monotonic() - started}
        write_json(destination / "training.json", summary)
    finally:
        if optimizer is not None:
            optimizer.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)
    args = parser.parse_args()
    if args.method == "baseline":
        parser.error("baseline has no training; use evaluate_cuda.py")
    destination, spec = run_spec(args)
    with gpu_lock():
        signature = claim_run(destination, spec)
        marker = destination / "training_started.json"
        if marker.exists():
            raise RuntimeError("Training is not resumable: preserve this evidence and choose a fresh run-id")
        write_json(marker, {"signature": signature})
        try:
            train(args, destination, spec, signature)
        except BaseException as exc:
            write_json(destination / "training_failure.json", {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
            raise
    # A fresh process reloads the exact saved file and runs the original small evaluation.
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    import subprocess
    import sys
    command = [sys.executable, "-X", "utf8", str(ROOT / "evaluate_cuda.py"), *sys.argv[1:], "--limit", "2"]
    subprocess.run(command, check=True)
    from cuda_common import read_json
    training = read_json(destination / "training.json")
    training["status"] = "passed_resource_probe" if args.probe else "completed_training_reload_checked"
    training["saved_checkpoint_reloaded_and_evaluated"] = True
    write_json(destination / "training.json", training)
    if args.probe:
        write_json(destination / "probe_passed.json", {"status": "passed_resource_probe", "signature": signature,
                   "updates": 3, "visits": 96, "saved_checkpoint_reloaded_and_evaluated": True})


if __name__ == "__main__":
    main()
