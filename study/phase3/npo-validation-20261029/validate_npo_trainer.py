"""Compare online upstream NPO with cached sequential NPO through real Trainer."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

CACHE = Path(__file__).resolve().parent
ROOT = CACHE / 'pr4-npo-5f792a'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'open-unlearning-original/src'))
import torch
from transformers import LlamaConfig, LlamaForCausalLM, TrainingArguments, TrainerCallback
from npo_cuda import CudaNPO
from npo_data import NPOTracedPairs, NPOTrainingCollator, reference_samples
from npo_reference import ReferenceNLLCache, build_reference_cache
from trainer.unlearn.npo import NPO
from full_core import TracedPairs
from memory_gd import StaticTrainingCollator
from cuda_runtime import configure


class StopThree(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= 3:
            control.should_training_stop = True


class RecordSGD(torch.optim.SGD):
    def __init__(self, parameters):
        super().__init__(parameters, lr=1e-5)
        self.records = []
    def step(self, closure=None):
        self.records.append(torch.cat([p.grad.detach().float().cpu().reshape(-1)
            for group in self.param_groups for p in group['params']]))
        return super().step(closure)


def check(device, dtype, directory):
    configure(0)
    config = LlamaConfig(vocab_size=127, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        attention_dropout=0., tie_word_embeddings=True, use_cache=False)
    config._attn_implementation = 'sdpa'
    initial = LlamaForCausalLM(config).to(device=device, dtype=dtype)
    examples = []
    for index in range(40):
        ids = torch.randint(2, 127, (16,))
        labels = ids.clone()
        labels[:4] = -100
        examples.append(dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels))
    tokenizer = SimpleNamespace(pad_token_id=1, padding_side='right')
    samples = reference_samples(examples, tokenizer, 192)
    cache = ReferenceNLLCache(build_reference_cache(initial, samples,
        dict(reference_checkpoint_sha256='tiny-fixed-initial', tokenizer_hash='tiny', data_protocol='tiny-static-192',reference_batch_size=4),batch_size=4))
    records, traces, losses = [], [], []
    for name, cls in (('online', NPO), ('cached', CudaNPO)):
        model = copy.deepcopy(initial)
        paired = NPOTracedPairs if name == 'cached' else TracedPairs
        collator = NPOTrainingCollator if name == 'cached' else StaticTrainingCollator
        dataset = paired(examples, examples * 99)
        optimizer = RecordSGD(model.parameters())
        args = TrainingArguments(output_dir=str(directory/name), use_cpu=device == 'cpu',
            per_device_train_batch_size=4, gradient_accumulation_steps=8, max_steps=10,
            learning_rate=1e-5, warmup_steps=1, lr_scheduler_type='linear',
            max_grad_norm=0., gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant':False},
            bf16=False, fp16=False, tf32=False, seed=0, data_seed=0,
            dataloader_pin_memory=False, dataloader_num_workers=0, remove_unused_columns=False,
            report_to='none', save_strategy='no', eval_strategy='no', disable_tqdm=True,
            logging_steps=1, logging_nan_inf_filter=False,
            accelerator_config={'gradient_accumulation_kwargs':{'sync_with_dataloader':False}})
        kwargs = dict(reference_cache=cache) if name == 'cached' else {}
        trainer = cls(model=model, args=args, train_dataset=dataset,
            data_collator=collator(tokenizer,192), optimizers=(optimizer,None),
            callbacks=[StopThree()], alpha=1., gamma=1., beta=.1, retain_loss_type='NLL', **kwargs)
        result = trainer.train()
        assert trainer.state.global_step == 3 and round(trainer.state.epoch*40) == 96
        if trainer.ref_model is not None:
            assert all(p.grad is None for p in trainer.ref_model.parameters())
        records.append(optimizer.records)
        traces.append(dataset.trace[:96])
        losses.append(result.training_loss)
        del trainer, optimizer, model
    assert traces[0] == traces[1]
    errors = []
    for online, cached in zip(*records):
        assert torch.isfinite(online).all() and torch.isfinite(cached).all()
        error = float((online-cached).norm()/online.norm())
        assert error < (2e-6 if dtype == torch.float32 else .02), error
        errors.append(error)
    return dict(device=device,dtype=str(dtype),updates=3,visits=96,
        gradient_clipping_disabled_to_expose_scaling=True, same_sampling=True,
        relative_gradient_errors=errors, online_training_loss=losses[0], cached_training_loss=losses[1])


torch.set_num_threads(2)
with tempfile.TemporaryDirectory(dir=CACHE/'pr3-test-temp') as directory:
    report = dict(status='passed', scope='Tiny-model real Trainer/Accelerate NPO integration; not formal 1B training',
        checks=[check('cpu',torch.float32,Path(directory)/'cpu'),
                check('cuda',torch.bfloat16,Path(directory)/'cuda')])
(CACHE/'npo-trainer-parity-batch4.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
