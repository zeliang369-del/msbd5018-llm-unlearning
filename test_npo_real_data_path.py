"""Regression checks for the real unbatched-QA -> cache -> Trainer path."""
import copy
import sys
from types import SimpleNamespace

import torch
import pytest
from transformers import LlamaConfig, LlamaForCausalLM

from npo_cuda import CudaNPO
from npo_data import NPOForgetDataset, NPOTrainingCollator, reference_samples
from npo_reference import build_reference_cache, load_reference_cache, save_reference_cache, sample_fingerprint
from trainer.unlearn.npo import NPO


def examples(count=4):
    rows = []
    for index in range(count):
        ids = torch.tensor([1, 2 + index % 4, 3, 4, 5, 6])
        labels = ids.clone()
        labels[:2] = -100
        rows.append(dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels))
    return rows


def model():
    config = LlamaConfig(vocab_size=31, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        attention_dropout=0., use_cache=False)
    config._attn_implementation = 'sdpa'
    return LlamaForCausalLM(config)


def test_reference_samples_match_actual_collator_slices_and_preserve_dataset():
    rows = examples()
    tokenizer = SimpleNamespace(pad_token_id=0)
    samples = reference_samples(rows, tokenizer, 8)
    dataset = NPOForgetDataset(rows)
    batch = NPOTrainingCollator(tokenizer, 8)([dataset[2], dataset[0]])
    for position, index in enumerate((2, 0)):
        sample = {key: batch[key][position:position+1] for key in ('input_ids', 'attention_mask', 'labels')}
        assert sample_fingerprint(samples[str(index)]) == sample_fingerprint(sample)
        assert samples[str(index)]['input_ids'].shape == (1, 8)
        assert samples[str(index)]['attention_mask'].dtype == torch.bool
    assert all(row['input_ids'].ndim == 1 and 'sample_id' not in row for row in rows)


def test_reference_preparation_cli_accepts_unbatched_qa_outputs(monkeypatch, tmp_path):
    import cuda_runtime
    import full_core
    import prepare_npo_reference
    tiny = model()
    tokenizer = SimpleNamespace(pad_token_id=0)
    monkeypatch.setattr(cuda_runtime, 'configure', lambda seed: None)
    monkeypatch.setattr(cuda_runtime, 'load_cuda', lambda path: (tiny, tokenizer, None))
    monkeypatch.setattr(full_core, 'read_rows', lambda name: [{}] * 40)
    monkeypatch.setattr(full_core, 'QA', lambda rows, tokenizer, template: examples(len(rows)))
    monkeypatch.setattr(full_core, 'file_hash', lambda path: 'toy-checkpoint-or-tokenizer')
    destination = tmp_path / 'reference.json'
    monkeypatch.setattr(sys, 'argv', ['prepare_npo_reference.py', '--checkpoint', str(tmp_path), '--output', str(destination), '--limit', '2'])
    prepare_npo_reference.main()
    samples = reference_samples(examples(2), tokenizer, 192)
    metadata = dict(reference_checkpoint_sha256='toy-checkpoint-or-tokenizer', tokenizer_hash='toy-checkpoint-or-tokenizer',
        data_protocol='full_core_official_item_v1_static_padding_192', scope='limit', forget_count=2,
        full_forget_count=40, reference_batch_size=4)
    cache = load_reference_cache(destination, samples, metadata)
    batch = NPOTrainingCollator(tokenizer, 192)([NPOForgetDataset(examples(2))[i] for i in (1, 0)])
    assert torch.isfinite(cache.lookup(batch['sample_id'], batch)).all()


def test_actual_upstream_npo_loss_and_gradients_match_cached_adapter(tmp_path):
    torch.manual_seed(41)
    reference = model().eval()
    first = copy.deepcopy(reference)
    with torch.no_grad():
        first.model.layers[0].mlp.up_proj.weight.add_(.01)
    second = copy.deepcopy(first)
    tokenizer = SimpleNamespace(pad_token_id=0)
    rows = examples()
    samples = reference_samples(rows, tokenizer, 8)
    metadata = dict(reference_checkpoint_sha256='toy', tokenizer_hash='toy', data_protocol='toy')
    path = tmp_path / 'cache.json'
    save_reference_cache(path, build_reference_cache(reference, samples, metadata))
    cache = load_reference_cache(path, samples, metadata)
    forget = NPOTrainingCollator(tokenizer, 8)([NPOForgetDataset(rows)[i] for i in (3, 1)])
    retain = NPOTrainingCollator(tokenizer, 8)([rows[i] for i in (0, 2)])
    online = NPO.__new__(NPO)
    online.beta, online.alpha, online.gamma = .1, .7, 1.3
    online.retain_loss_type, online.ref_model = 'NLL', reference
    upstream_input = dict(forget={key:value for key,value in forget.items() if key != 'sample_id'}, retain=retain)
    expected = online.compute_loss(first, upstream_input)
    expected.backward()
    adapted = CudaNPO.__new__(CudaNPO)
    adapted.beta, adapted.alpha, adapted.gamma = .1, .7, 1.3
    adapted.retain_loss_type, adapted.ref_model = 'NLL', None
    adapted.reference_cache, adapted.reference_nll = cache, None
    adapted.allow_direct_reference_nll = False
    actual = adapted.compute_loss(second, dict(forget=forget, retain=retain))
    actual.backward()
    assert torch.allclose(actual, expected.detach(), rtol=2e-6, atol=2e-6)
    a = torch.cat([p.grad.reshape(-1) for p in first.parameters()])
    b = torch.cat([p.grad.reshape(-1) for p in second.parameters()])
    assert torch.linalg.vector_norm(a-b) / torch.linalg.vector_norm(a) < 2e-6
    assert all(p.grad is None for p in reference.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_cpu_prepared_cache_samples_are_moved_to_reference_cuda_device():
    tiny = model().to('cuda')
    tokenizer = SimpleNamespace(pad_token_id=0)
    samples = reference_samples(examples(2), tokenizer, 8)
    metadata = dict(reference_checkpoint_sha256='toy', tokenizer_hash='toy', data_protocol='toy')
    cache = build_reference_cache(tiny, samples, metadata)
    assert len(cache['values']) == 2
    assert all(torch.isfinite(torch.tensor(row['nll_sum'])) for row in cache['values'].values())
    assert all(sample['input_ids'].device.type == 'cpu' for sample in samples.values())


def test_reference_batching_preserves_ids_values_and_partial_final_batch():
    torch.manual_seed(59)
    tiny = model().eval()
    samples = reference_samples(examples(5), SimpleNamespace(pad_token_id=0), 8)
    metadata = dict(reference_checkpoint_sha256='toy',tokenizer_hash='toy',data_protocol='toy')
    single = build_reference_cache(tiny,samples,metadata,batch_size=1)
    grouped = build_reference_cache(tiny,samples,metadata,batch_size=4)
    assert set(single['values']) == set(grouped['values']) == set(samples)
    for key in samples:
        assert single['values'][key]['fingerprint'] == grouped['values'][key]['fingerprint']
        assert abs(single['values'][key]['nll_sum']-grouped['values'][key]['nll_sum']) < 1e-4
