"""Small real-model checks of the new adapters, independent of frozen evidence."""
import copy
from contextlib import nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM, TrainingArguments
from cuda_common import ROOT, write_json
from cuda_runtime import CudaStreamedAdamW, CudaGradDiff
from full_core import TracedPairs
from trainer.unlearn.grad_ascent import GradAscent
from trainer.unlearn.grad_diff import GradDiff
from memory_gd import pad_batch, StaticTrainingCollator


def tiny_model(device="cpu", dtype=torch.float32):
    config = LlamaConfig(vocab_size=127, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                         attention_dropout=0., tie_word_embeddings=True)
    config._attn_implementation = "sdpa"
    config.use_cache = False
    model = LlamaForCausalLM(config).to(device=device, dtype=dtype)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def gd_gradient_check(device, dtype):
    results = []
    for alpha, gamma in ((1., 1.), (.5, 2.), (2., .5)):
        torch.manual_seed(91)
        first = tiny_model(device, dtype)
        second = copy.deepcopy(first)
        inputs = {}
        for key, length in (("forget", 23), ("retain", 31)):
            ids = torch.randint(0, 127, (4, length), device=device)
            labels = ids.clone()
            labels[:, :7] = -100
            inputs[key] = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}
        joint = SimpleNamespace(alpha=alpha, gamma=gamma, retain_loss_type="NLL")
        joint.compute_retain_loss = lambda model, retain_inputs: GradDiff.compute_retain_loss(joint, model, retain_inputs)
        loss = GradDiff.compute_loss(joint, first, inputs)
        (loss / 8).backward()
        sequential = SimpleNamespace(alpha=alpha, gamma=gamma, retain_loss_type="NLL",
                                     args=SimpleNamespace(gradient_accumulation_steps=8))
        sequential._prepare_inputs = lambda value: value
        sequential.compute_loss_context_manager = nullcontext
        sequential.accelerator = SimpleNamespace(backward=lambda value: (value / 8).backward())
        sequential.compute_retain_loss = lambda model, retain_inputs: GradDiff.compute_retain_loss(sequential, model, retain_inputs)
        adapted = CudaGradDiff.training_step(sequential, second, pad_batch(inputs, 32, 1))
        a = torch.cat([p.grad.detach().cpu().float().reshape(-1) for p in first.parameters()])
        b = torch.cat([p.grad.detach().cpu().float().reshape(-1) for p in second.parameters()])
        error = float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(a))
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        assert error < (2e-6 if dtype == torch.float32 else .02), error
        assert abs(float(loss.detach() / 8) - float(adapted)) < (1e-6 if dtype == torch.float32 else .005)
        results.append({"alpha": alpha, "gamma": gamma, "relative_gradient_error": error})
    return {"device": device, "dtype": str(dtype), "coefficients": results}


def optimizer_check(device):
    torch.manual_seed(17)
    initial = (torch.randn(257) * .01).to(torch.bfloat16)
    low = torch.nn.Parameter(initial.to(device).clone())
    reference = torch.nn.Parameter(initial.float().clone())
    ordinary = torch.optim.AdamW([reference], lr=1e-5, weight_decay=.01, foreach=False)
    errors = []
    with tempfile.TemporaryDirectory() as temp:
        streamed = CudaStreamedAdamW([{"params": [low], "weight_decay": .01}], [("test", low)],
                                    Path(temp) / "state", chunk_elements=37, reserve_bytes=0, capture_gradients=False)
        try:
            for rate in (0., 1e-5, 8.88888888888889e-6):
                streamed.param_groups[0]["lr"] = ordinary.param_groups[0]["lr"] = rate
                gradient = torch.randn(257).to(torch.bfloat16)
                (low.float() * gradient.to(device).float()).sum().backward()
                reference.grad = gradient.float().clone()
                streamed.clip_grad_norm(1.)
                torch.nn.utils.clip_grad_norm_([reference], 1.)
                streamed.step()
                ordinary.step()
                streamed.zero_grad()
                ordinary.zero_grad()
                master = torch.from_numpy(np.array(streamed.maps["master"]))
                error = float((master - reference.detach()).abs().max())
                assert error < 4e-7, error
                assert torch.equal(low.detach().cpu(), reference.detach().to(torch.bfloat16))
                if rate == 0:
                    assert torch.equal(low.detach().cpu(), initial)
                errors.append(error)
            assert torch.any(low.detach().cpu() != initial)
        finally:
            streamed.cleanup()
    return {"device": device, "updates": 3, "FP32_master_max_absolute_error": max(errors),
            "zero_rate_invariance": True, "nonzero_rate_parameter_change": True}


def windows_flush_check():
    # 72000 is page-aligned by the old code to 69632, which is not a valid
    # Windows allocation-granularity offset. Exercise that actual region.
    parameter = torch.nn.Parameter(torch.ones(20000))
    with tempfile.TemporaryDirectory() as temp:
        optimizer = CudaStreamedAdamW([{"params": [parameter]}], [("alignment", parameter)],
                                     Path(temp) / "state", reserve_bytes=0, capture_gradients=False)
        try:
            optimizer._flush_drop(optimizer.maps["master"], 18000, 100)
        finally:
            optimizer.cleanup()
    return {"status": "passed", "unaligned_region_start_bytes": 72000}


def budget_check(seed, directory):
    # Match the exact 40-example / four-per-batch / eight-accumulation geometry.
    torch.manual_seed(11)
    examples = []
    for _ in range(40):
        ids = torch.randint(2, 127, (16,))
        labels = ids.clone()
        labels[:4] = -100
        examples.append({"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels})
    data = TracedPairs(examples, examples * 99)
    model = tiny_model()
    args = TrainingArguments(output_dir=str(directory), use_cpu=True, per_device_train_batch_size=4,
        gradient_accumulation_steps=8, num_train_epochs=10, max_steps=10, learning_rate=1e-5,
        warmup_steps=1, lr_scheduler_type="linear", weight_decay=.01, max_grad_norm=1,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=seed, data_seed=seed, dataloader_num_workers=0, dataloader_pin_memory=False,
        remove_unused_columns=False, report_to="none", save_strategy="no", eval_strategy="no",
        disable_tqdm=True, logging_steps=1, logging_nan_inf_filter=False,
        accelerator_config={"gradient_accumulation_kwargs": {"sync_with_dataloader": False}})
    tokenizer = SimpleNamespace(pad_token_id=1, padding_side="right")
    trainer = GradAscent(model=model, train_dataset=data, args=args,
                         data_collator=StaticTrainingCollator(tokenizer, 192))
    trainer.train()
    visits = round(trainer.state.epoch * 40)
    assert trainer.state.global_step == 10 and visits == 320, (trainer.state.global_step, visits)
    return {"seed": seed, "updates": trainer.state.global_step, "visits": visits,
            "data_passes": trainer.state.epoch, "trace": data.trace[:visits]}


def main():
    torch.set_num_threads(2)
    results = {"scope": "tiny-model adapter checks, not formal 1B training", "status": "passed",
               "gradients": [gd_gradient_check("cpu", torch.float32)], "optimizer": [optimizer_check("cpu")]}
    results["mmap_flush_alignment"] = windows_flush_check()
    if torch.cuda.is_available():
        results["gradients"].append(gd_gradient_check("cuda", torch.bfloat16))
        results["optimizer"].append(optimizer_check("cuda"))
    with tempfile.TemporaryDirectory() as temp:
        first = budget_check(0, Path(temp) / "first")
        repeated = budget_check(0, Path(temp) / "repeated")
        changed = budget_check(1, Path(temp) / "changed")
        assert first["trace"] == repeated["trace"]
        assert first["trace"] != changed["trace"]
        results["budget"] = [{k: v for k, v in entry.items() if k != "trace"} for entry in (first, repeated, changed)]
        results["same_seed_sampling_matches"] = True
        results["changed_seed_changes_sampling"] = True
    write_json(ROOT / "study/environment/cuda-adapter-numerics.json", results)
    print(results, flush=True)


if __name__ == "__main__":
    main()
