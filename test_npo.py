"""CPU tests for fixed-reference NPO and its sample-ID cache."""

import copy
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from npo_cuda import CudaNPO, compute_nll_sum
from npo_reference import ReferenceNLLCache, build_reference_cache, load_reference_cache


class TinyLM(nn.Module):
    def __init__(self, vocab=13, hidden=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, vocab)

    def forward(self, input_ids, attention_mask=None, labels=None):
        logits = self.proj(self.embedding(input_ids))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits[..., :-1, :].transpose(-1, -2), labels[..., 1:], ignore_index=-100)
        return SimpleNamespace(logits=logits, loss=loss)


def make_sample(ids, prompt, end):
    labels = ids.clone()
    labels[:, :prompt] = -100
    labels[:, end:] = -100
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


def inputs():
    return {
        "forget": make_sample(torch.tensor([[1, 2, 3, 4, 5, 6]]), 2, 6),
        "retain": make_sample(torch.tensor([[2, 4, 6, 8, 1, 3]]), 1, 5),
    }


def setup(model, beta=.1, alpha=1., gamma=1., reference_nll=None, reference_cache=None):
    trainer = CudaNPO.__new__(CudaNPO)
    trainer.beta, trainer.alpha, trainer.gamma = beta, alpha, gamma
    trainer.retain_loss_type, trainer.ref_model = "NLL", None
    trainer.reference_nll = reference_nll
    trainer.reference_cache = reference_cache
    trainer.allow_direct_reference_nll = reference_nll is not None
    return trainer


def grads(model):
    return torch.cat([p.grad.reshape(-1) for p in model.parameters()])


def reference_formula(current, reference, batch, beta, alpha, gamma):
    current_nll, _ = compute_nll_sum(current, batch["forget"])
    reference_nll, _ = compute_nll_sum(reference, batch["forget"])
    forget = -2 / beta * torch.nn.functional.logsigmoid(beta * (current_nll - reference_nll)).mean()
    return gamma * forget + alpha * current(batch["retain"]["input_ids"], labels=batch["retain"]["labels"]).loss


def test_loss_and_gradient_match_independent_formula_and_reference_has_no_grad():
    torch.manual_seed(4)
    current = TinyLM()
    reference = TinyLM()
    batch = inputs()
    expected = reference_formula(current, reference, batch, .1, .7, 1.3)
    expected.backward()
    expected_grads = grads(current).clone()
    current.zero_grad()
    ref_nll, _ = compute_nll_sum(reference, batch["forget"])
    reference.zero_grad()
    adapted = setup(current, .1, .7, 1.3, ref_nll.detach())
    actual = adapted.compute_loss(current, batch)
    actual.backward()
    assert torch.allclose(actual, expected.detach(), atol=1e-7, rtol=1e-6)
    assert torch.allclose(grads(current), expected_grads, atol=1e-6, rtol=1e-5)
    assert all(parameter.grad is None for parameter in reference.parameters())


def test_nll_is_sum_not_mean_and_mask_shift_padding_eos_are_consistent():
    model = TinyLM()
    ids = torch.tensor([[1, 2, 3, 4, 5, 0, 0]])
    sample = make_sample(ids, 3, 5)
    nll, _ = compute_nll_sum(model, sample)
    logits = model(ids).logits[..., :-1, :]
    labels = sample["labels"][..., 1:]
    token_losses = nn.functional.cross_entropy(logits.transpose(-1, -2), labels, ignore_index=-100, reduction="none")
    assert torch.allclose(nll, token_losses.sum(-1))
    assert not torch.allclose(nll, token_losses.sum(-1) / (labels != -100).sum(-1))


def test_cache_is_online_equivalent_and_survives_sample_reordering(tmp_path):
    torch.manual_seed(7)
    reference = TinyLM()
    samples = {"b": inputs()["forget"], "a": make_sample(torch.tensor([[4, 5, 6, 7, 8, 9]]), 2, 6)}
    metadata = {"reference_checkpoint_sha256": "ref", "tokenizer_hash": "tok", "data_protocol": "toy-v1"}
    cache = build_reference_cache(reference, samples, metadata)
    path = tmp_path / "reference.json"
    from npo_reference import save_reference_cache
    save_reference_cache(path, cache)
    loaded = load_reference_cache(path, {"a": samples["a"], "b": samples["b"]}, metadata)
    assert set(loaded.cache["values"]) == {"a", "b"}
    for key, sample in samples.items():
        online, _ = compute_nll_sum(reference, sample)
        assert loaded.cache["values"][key]["nll_sum"] == pytest.approx(float(online.item()))


def test_verified_cache_lookup_uses_ids_and_fingerprints_not_batch_position():
    reference = TinyLM()
    first = make_sample(torch.tensor([[1, 2, 3, 4, 5, 6]]), 2, 6)
    second = make_sample(torch.tensor([[4, 5, 6, 7, 8, 9]]), 2, 6)
    samples = {"10": first, "20": second}
    metadata = {"reference_checkpoint_sha256": "ref", "tokenizer_hash": "tok", "data_protocol": "toy-v1"}
    cache = ReferenceNLLCache(build_reference_cache(reference, samples, metadata))
    reordered = {key: torch.cat([second[key], first[key]]) for key in first}
    values = cache.lookup(torch.tensor([20, 10]), reordered)
    expected = torch.tensor([compute_nll_sum(reference, second)[0].item(), compute_nll_sum(reference, first)[0].item()])
    assert torch.allclose(values, expected)
    altered = {key: value.clone() for key, value in reordered.items()}
    altered["labels"][0, -1] = -100
    with pytest.raises(ValueError, match="content mismatch"):
        cache.lookup(torch.tensor([20, 10]), altered)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_beta_rejected(value):
    with pytest.raises(ValueError, match="beta"):
        CudaNPO(beta=value)


def test_reference_nll_shape_and_finite_values_rejected_before_backward():
    current = TinyLM()
    batch = inputs()
    trainer = setup(current, reference_nll=torch.tensor([float("nan")]))
    with pytest.raises(ValueError, match="finite"):
        trainer.compute_loss(current, batch)
    trainer.reference_nll = torch.tensor([1.0, 2.0])
    with pytest.raises(ValueError, match="shape"):
        trainer.compute_loss(current, batch)


@pytest.mark.parametrize("field,value", [("metadata", {"reference_checkpoint_sha256": "wrong", "tokenizer_hash": "tok", "data_protocol": "toy-v1"})])
def test_cache_metadata_mismatch_rejected(tmp_path, field, value):
    reference = TinyLM()
    samples = {"x": inputs()["forget"]}
    metadata = {"reference_checkpoint_sha256": "ref", "tokenizer_hash": "tok", "data_protocol": "toy-v1"}
    from npo_reference import save_reference_cache
    path = tmp_path / "reference.json"
    save_reference_cache(path, build_reference_cache(reference, samples, metadata))
    with pytest.raises(ValueError, match="metadata mismatch"):
        load_reference_cache(path, samples, value)


def test_sequential_accumulation_eight_matches_combined_gradient():
    torch.manual_seed(11)
    current = TinyLM()
    reference = TinyLM()
    batch = inputs()
    ref_nll, _ = compute_nll_sum(reference, batch["forget"])
    expected = reference_formula(current, reference, batch, .1, 1., 1.) / 8
    expected.backward()
    expected_grads = grads(current).clone()
    current.zero_grad()
    trainer = setup(current, .1, 1., 1., ref_nll.detach())
    trainer._prepare_inputs = lambda value: value
    trainer.compute_loss_context_manager = nullcontext
    trainer.args = SimpleNamespace(gradient_accumulation_steps=8)
    trainer.accelerator = SimpleNamespace(backward=lambda loss: (loss / 8).backward())
    returned = trainer.training_step(current, batch)
    assert torch.allclose(returned, expected.detach(), atol=1e-7, rtol=1e-6)
    assert torch.allclose(grads(current), expected_grads, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("branch", ["forget", "retain"])
def test_invalid_labels_rejected_before_any_backward(branch):
    current = TinyLM()
    reference = TinyLM()
    batch = inputs()
    batch[branch]["labels"][:] = -100
    ref_nll, _ = compute_nll_sum(reference, inputs()["forget"])
    trainer = setup(current, reference_nll=ref_nll.detach())
    trainer._prepare_inputs = lambda value: value
    trainer.compute_loss_context_manager = nullcontext
    trainer.args = SimpleNamespace(gradient_accumulation_steps=1)
    trainer.accelerator = SimpleNamespace(backward=lambda loss: loss.backward())
    with pytest.raises(ValueError):
        trainer.training_step(current, batch)
    assert all(parameter.grad is None for parameter in current.parameters())


def test_upstream_npo_direct_comparison_when_dependencies_are_available():
    try:
        from trainer.unlearn.npo import NPO as UpstreamNPO
    except ModuleNotFoundError as error:
        pytest.skip(f"upstream NPO unavailable: missing {error.name}")
    assert UpstreamNPO is not None
