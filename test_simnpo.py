"""CPU checks for the SimNPO CUDA adapter without initializing a Trainer."""

import copy
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from simnpo_cuda import CudaSimNPO


class ReferenceSimNPO:
    def __init__(self, delta=0.0, beta=1.0, alpha=1.0, gamma=1.0):
        self.delta, self.beta = delta, beta
        self.alpha, self.gamma = alpha, gamma
        self.retain_loss_type = "NLL"

    def compute_retain_loss(self, model, retain_inputs):
        return model(**retain_inputs).loss

    def compute_loss(self, model, inputs):
        from simnpo_cuda import compute_batch_nll
        forget = inputs["forget"]
        loss, _ = compute_batch_nll(model, forget)
        loss = loss / (forget["labels"] != -100).sum(-1) - self.delta
        loss = -torch.nn.functional.logsigmoid(self.beta * loss).mean() * 2 / self.beta
        retain = {k: inputs["retain"][k] for k in ("input_ids", "attention_mask", "labels")}
        return self.gamma * loss + self.alpha * self.compute_retain_loss(model, retain)


class TinyLM(nn.Module):
    def __init__(self, vocab=11, hidden=7):
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, vocab)

    def forward(self, input_ids, attention_mask=None, labels=None):
        logits = self.proj(self.embedding(input_ids))
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_labels = labels[..., 1:].contiguous()
        loss = nn.functional.cross_entropy(
            shifted_logits.transpose(-1, -2), shifted_labels, ignore_index=-100
        )
        return SimpleNamespace(logits=logits, loss=loss)


def batch(ids, valid_from, valid_to=None):
    labels = ids.clone()
    labels[:, :valid_from] = -100
    if valid_to is not None:
        labels[:, valid_to:] = -100
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "labels": labels,
    }


def make_inputs():
    forget_ids = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 4, 6, 8, 1, 3]])
    retain_ids = torch.tensor([[3, 5, 7, 9, 1, 2], [4, 6, 8, 10, 2, 4]])
    return {
        "forget": batch(forget_ids, 2, 6),
        "retain": batch(retain_ids, 1, 5),
    }


def setup(cls, model, beta=1.0, delta=0.0, alpha=1.0, gamma=1.0):
    trainer = cls.__new__(cls)
    trainer.beta = beta
    trainer.delta = delta
    trainer.alpha = alpha
    trainer.gamma = gamma
    trainer.retain_loss_type = "NLL"
    trainer.ref_model = None
    return trainer


def flat_grads(model):
    return torch.cat([p.grad.reshape(-1) for p in model.parameters()])


@pytest.mark.parametrize("beta,delta,alpha,gamma", [(1.0, 0.0, 1.0, 1.0), (4.5, 0.7, 0.3, 0.125)])
def test_loss_and_parameter_gradients_match_upstream(beta, delta, alpha, gamma):
    torch.manual_seed(3)
    reference_model = TinyLM()
    adapted_model = copy.deepcopy(reference_model)
    inputs = make_inputs()
    reference = ReferenceSimNPO(delta, beta, alpha, gamma)
    adapted = setup(CudaSimNPO, adapted_model, beta, delta, alpha, gamma)

    reference_loss = reference.compute_loss(reference_model, inputs)
    reference_loss.backward()
    adapted_loss = adapted.compute_loss(adapted_model, inputs)
    adapted_loss.backward()

    assert torch.allclose(adapted_loss, reference_loss.detach(), atol=1e-7, rtol=1e-6)
    assert torch.allclose(flat_grads(adapted_model), flat_grads(reference_model), atol=1e-6, rtol=1e-5)


def test_sequential_training_step_matches_combined_gradient_and_update():
    torch.manual_seed(9)
    reference_model = TinyLM()
    adapted_model = copy.deepcopy(reference_model)
    inputs = make_inputs()
    reference = ReferenceSimNPO(0.2, 4.5, 0.3, 0.125)
    adapted = setup(CudaSimNPO, adapted_model, 4.5, 0.2, 0.3, 0.125)
    reference.compute_loss(reference_model, inputs).backward()

    adapted._prepare_inputs = lambda value: value
    adapted.compute_loss_context_manager = nullcontext
    adapted.args = SimpleNamespace(gradient_accumulation_steps=1)
    adapted.accelerator = SimpleNamespace(backward=lambda loss: loss.backward())
    returned = adapted.training_step(adapted_model, inputs)

    assert torch.allclose(returned, reference.compute_loss(reference_model, inputs).detach(), atol=1e-7, rtol=1e-6)
    assert torch.allclose(flat_grads(adapted_model), flat_grads(reference_model), atol=1e-6, rtol=1e-5)

    ref_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.01)
    adapted_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=0.01)
    ref_optimizer.step()
    adapted_optimizer.step()
    for adapted_parameter, reference_parameter in zip(adapted_model.parameters(), reference_model.parameters()):
        assert torch.allclose(adapted_parameter, reference_parameter, atol=1e-7, rtol=1e-6)


def test_sequential_training_step_matches_accumulated_gradient_at_eight():
    torch.manual_seed(19)
    reference_model = TinyLM()
    adapted_model = copy.deepcopy(reference_model)
    inputs = make_inputs()
    reference = ReferenceSimNPO(0.2, 4.5, 0.3, 0.125)
    adapted = setup(CudaSimNPO, adapted_model, 4.5, 0.2, 0.3, 0.125)
    reference.compute_loss(reference_model, inputs).div(8).backward()

    adapted._prepare_inputs = lambda value: value
    adapted.compute_loss_context_manager = nullcontext
    adapted.args = SimpleNamespace(gradient_accumulation_steps=8)
    adapted.accelerator = SimpleNamespace(backward=lambda loss: loss.div(8).backward())
    returned = adapted.training_step(adapted_model, inputs)

    assert torch.allclose(returned, reference.compute_loss(reference_model, inputs).detach().div(8), atol=1e-7, rtol=1e-6)
    assert torch.allclose(flat_grads(adapted_model), flat_grads(reference_model), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("alpha,gamma,active", [(0.0, 1.0, "forget"), (1.0, 0.0, "retain")])
def test_zero_branch_weight_removes_that_branch_gradient(alpha, gamma, active):
    torch.manual_seed(23)
    model = TinyLM()
    trainer = setup(CudaSimNPO, model, alpha=alpha, gamma=gamma)
    inputs = make_inputs()
    trainer.compute_loss(model, inputs).backward()
    model.zero_grad()
    if active == "forget":
        trainer._compute_forget_loss(model, inputs["forget"])[0].backward()
    else:
        trainer.compute_retain_loss(model, inputs["retain"]).backward()
    expected = flat_grads(model).clone()
    model.zero_grad()
    trainer.compute_loss(model, inputs).backward()
    assert torch.allclose(flat_grads(model), expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("branch", ["forget", "retain"])
def test_all_ignored_labels_raise_clear_error(branch):
    model = TinyLM()
    inputs = make_inputs()
    inputs[branch]["labels"][:] = -100
    trainer = setup(CudaSimNPO, model)
    with pytest.raises(ValueError, match="at least one non-ignored label"):
        trainer.compute_loss(model, inputs)


def test_nonignored_first_label_but_no_shifted_label_raises():
    model = TinyLM()
    inputs = make_inputs()
    inputs["forget"]["labels"][:] = -100
    inputs["forget"]["labels"][:, 0] = 1
    trainer = setup(CudaSimNPO, model)
    with pytest.raises(ValueError, match="non-ignored shifted label"):
        trainer.compute_loss(model, inputs)


def test_upstream_simnpo_direct_comparison_when_dependencies_are_available():
    try:
        from trainer.unlearn.simnpo import SimNPO as UpstreamSimNPO
    except ModuleNotFoundError as error:
        pytest.skip(f"upstream SimNPO unavailable: missing {error.name}")

    torch.manual_seed(29)
    upstream_model = TinyLM()
    adapted_model = copy.deepcopy(upstream_model)
    inputs = make_inputs()
    upstream = UpstreamSimNPO.__new__(UpstreamSimNPO)
    upstream.beta, upstream.delta = 4.5, 0.2
    upstream.alpha, upstream.gamma = 0.3, 0.125
    upstream.retain_loss_type, upstream.ref_model = "NLL", None
    adapted = setup(CudaSimNPO, adapted_model, 4.5, 0.2, 0.3, 0.125)

    upstream_loss = upstream.compute_loss(upstream_model, inputs)
    upstream_loss.backward()
    adapted_loss = adapted.compute_loss(adapted_model, inputs)
    adapted_loss.backward()
    assert torch.allclose(adapted_loss, upstream_loss.detach(), atol=1e-7, rtol=1e-6)
    assert torch.allclose(flat_grads(adapted_model), flat_grads(upstream_model), atol=1e-6, rtol=1e-5)


def test_padding_and_prompt_mask_are_ignored_but_eos_is_scored():
    torch.manual_seed(17)
    model = TinyLM()
    trainer = setup(CudaSimNPO, model, beta=2.0, delta=0.4)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    padded_ids = torch.tensor([[1, 2, 3, 4, 5, 0, 0]])
    inputs = {"forget": batch(ids, 3, 5), "retain": batch(ids, 2, 5)}
    padded = {"forget": batch(padded_ids, 3, 5), "retain": batch(padded_ids, 2, 5)}
    base_loss = trainer.compute_loss(model, inputs)
    padded_loss = trainer.compute_loss(model, padded)
    assert torch.isfinite(base_loss)
    assert torch.isfinite(padded_loss)
    assert not torch.equal(inputs["forget"]["labels"], padded["forget"]["labels"])
    assert torch.allclose(base_loss, padded_loss, atol=1e-7, rtol=1e-6)
