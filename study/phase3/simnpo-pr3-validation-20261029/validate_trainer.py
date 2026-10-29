"""Exercise the unchanged PR adapter through real Trainer/Accelerate."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent / "pr3-gpu-509be4a-v2"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "open-unlearning-original/src"))
import torch
from transformers import LlamaConfig, LlamaForCausalLM, TrainingArguments, TrainerCallback
from simnpo_cuda import CudaSimNPO
from trainer.unlearn.simnpo import SimNPO
from full_core import TracedPairs
from memory_gd import StaticTrainingCollator
from cuda_runtime import configure


class StopAfterThree(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= 3:
            control.should_training_stop = True


class RecordingSGD(torch.optim.SGD):
    def __init__(self, parameters):
        super().__init__(parameters, lr=1e-5)
        self.records = []

    def step(self, closure=None):
        self.records.append(torch.cat([
            p.grad.detach().cpu().float().reshape(-1)
            for group in self.param_groups for p in group["params"]
        ]))
        return super().step(closure)


def check(device, dtype, output):
    configure(0)
    config = LlamaConfig(vocab_size=127, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        attention_dropout=0., tie_word_embeddings=True, use_cache=False)
    config._attn_implementation = "sdpa"
    initial = LlamaForCausalLM(config).to(device=device, dtype=dtype)
    examples = []
    for index in range(40):
        ids = torch.randint(2, 127, (16,))
        labels = ids.clone()
        labels[:4] = -100
        examples.append(dict(input_ids=ids, labels=labels, attention_mask=torch.ones_like(ids)))
    records, traces, returned = [], [], []
    for name, cls in (("upstream", SimNPO), ("sequential", CudaSimNPO)):
        model = copy.deepcopy(initial)
        dataset = TracedPairs(examples, examples * 99)
        optimizer = RecordingSGD(model.parameters())
        arguments = TrainingArguments(output_dir=str(output / name), use_cpu=device == "cpu",
            per_device_train_batch_size=4, gradient_accumulation_steps=8,
            max_steps=10, learning_rate=1e-5, warmup_steps=1, lr_scheduler_type="linear",
            max_grad_norm=1, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=False, fp16=False, tf32=False, seed=0, data_seed=0,
            dataloader_pin_memory=False, dataloader_num_workers=0,
            remove_unused_columns=False, report_to="none", save_strategy="no",
            eval_strategy="no", disable_tqdm=True, logging_steps=1,
            logging_nan_inf_filter=False,
            accelerator_config={"gradient_accumulation_kwargs": {"sync_with_dataloader": False}})
        trainer = cls(model=model, args=arguments, train_dataset=dataset,
            data_collator=StaticTrainingCollator(SimpleNamespace(pad_token_id=1, padding_side="right"), 192),
            optimizers=(optimizer, None), callbacks=[StopAfterThree()],
            alpha=1., gamma=.125, beta=4.5, delta=0., retain_loss_type="NLL")
        result = trainer.train()
        assert trainer.state.global_step == 3 and len(optimizer.records) == 3
        visits = round(trainer.state.epoch * 40)
        assert visits == 96
        records.append(optimizer.records)
        traces.append(dataset.trace[:visits])
        returned.append(result.training_loss)
        del trainer, optimizer, model
    assert traces[0] == traces[1]
    errors = []
    for reference, adapted in zip(*records):
        assert torch.isfinite(reference).all() and torch.isfinite(adapted).all()
        error = float((reference - adapted).norm() / reference.norm())
        assert error < (2e-6 if dtype == torch.float32 else .02), error
        errors.append(error)
    return dict(device=device, dtype=str(dtype), optimizer_updates=3, sample_visits=96,
        same_seed_sampling_matches=True, per_update_gradient_relative_errors=errors,
        training_loss_upstream=returned[0], training_loss_sequential=returned[1])


if __name__ == "__main__":
    torch.set_num_threads(2)
    report = dict(status="passed", commit="509be4a9e1dd229a8397d178c784f5d1b463fae6",
        scope="Tiny-model real Trainer/Accelerate gradient and accumulation parity; not a formal 1B result",
        torch=torch.__version__, checks=[])
    temporary = Path(__file__).resolve().parent / "pr3-test-temp"
    temporary.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary) as directory:
        report["checks"].append(check("cpu", torch.float32, Path(directory) / "cpu"))
        report["checks"].append(check("cuda", torch.bfloat16, Path(directory) / "cuda"))
    (Path(__file__).resolve().parent / "pr3-trainer-parity-509be4a.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
