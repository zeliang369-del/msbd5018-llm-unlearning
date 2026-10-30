"""CUDA-friendly NPO loss with a fixed, precomputed reference NLL."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_UPSTREAM_SRC = Path(__file__).resolve().parent / "open-unlearning-original" / "src"
if str(_UPSTREAM_SRC) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM_SRC))

try:
    from trainer.unlearn.grad_diff import GradDiff as _GradDiffBase
except ModuleNotFoundError as error:
    if error.name != "omegaconf":
        raise

    class _GradDiffBase:
        def compute_retain_loss(self, model, retain_inputs):
            return model(**retain_inputs).loss


def compute_nll_sum(model, inputs):
    """Return per-sample shifted token NLL sums, matching upstream utility."""
    outputs = model(**inputs)
    logits = outputs.logits[..., :-1, :].contiguous()
    labels = inputs["labels"][..., 1:].contiguous()
    losses = F.cross_entropy(
        logits.transpose(-1, -2), labels, ignore_index=-100, reduction="none"
    )
    return losses.sum(dim=-1), outputs


class CudaNPO(_GradDiffBase):
    """NPO adapter using fixed reference NLL sums and sequential backward."""

    def __init__(self, beta=0.1, *args, reference_cache=None, reference_nll=None,
                 allow_direct_reference_nll=False, **kwargs):
        if not torch.isfinite(torch.as_tensor(beta)) or float(beta) <= 0:
            raise ValueError("NPO beta must be finite and strictly positive")
        if kwargs.get("retain_loss_type", "NLL") != "NLL":
            raise ValueError("cached NPO requires retain_loss_type='NLL'")
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.reference_cache = reference_cache
        self.reference_nll = reference_nll
        self.allow_direct_reference_nll = allow_direct_reference_nll

    @staticmethod
    def _validate(labels):
        if torch.any((labels[..., 1:] != -100).sum(dim=-1) == 0):
            raise ValueError("NPO requires at least one non-ignored shifted label per sample")

    def _resolve_reference_nll(self, inputs, forget_inputs):
        sample_ids = forget_inputs.get("sample_id")
        if self.reference_cache is not None:
            if sample_ids is None:
                raise ValueError("NPO cache lookup requires Forget sample_id")
            return self.reference_cache.lookup(sample_ids, forget_inputs)
        if self.allow_direct_reference_nll:
            value = inputs.get("reference_nll", self.reference_nll)
            if value is not None:
                return value
        raise ValueError("NPO requires a verified reference cache")

    def _forget_loss(self, model, forget_inputs, reference_nll):
        self._validate(forget_inputs["labels"])
        model_inputs = {k: forget_inputs[k] for k in ("input_ids", "attention_mask", "labels")}
        current_nll, outputs = compute_nll_sum(model, model_inputs)
        reference_nll = torch.as_tensor(reference_nll, device=current_nll.device, dtype=current_nll.dtype)
        if reference_nll.shape != current_nll.shape:
            raise ValueError("Reference NLL shape must match the Forget batch")
        if not torch.isfinite(reference_nll).all():
            raise ValueError("Reference NLL must contain only finite values")
        loss = -2 / self.beta * F.logsigmoid(self.beta * (current_nll - reference_nll)).mean()
        return loss, outputs

    def compute_loss(self, model, inputs, return_outputs=False):
        forget = {k: inputs["forget"][k] for k in ("input_ids", "attention_mask", "labels")}
        if "sample_id" in inputs["forget"]:
            forget["sample_id"] = inputs["forget"]["sample_id"]
        retain = {k: inputs["retain"][k] for k in ("input_ids", "attention_mask", "labels")}
        self._validate(forget["labels"])
        self._validate(retain["labels"])
        reference_nll = self._resolve_reference_nll(inputs, forget)
        forget_loss, outputs = self._forget_loss(model, forget, reference_nll)
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain)
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)
        forget = {k: inputs["forget"][k] for k in ("input_ids", "attention_mask", "labels")}
        if "sample_id" in inputs["forget"]:
            forget["sample_id"] = inputs["forget"]["sample_id"]
        retain = {k: inputs["retain"][k] for k in ("input_ids", "attention_mask", "labels")}
        self._validate(forget["labels"])
        self._validate(retain["labels"])
        reference_nll = self._resolve_reference_nll(inputs, forget)
        with self.compute_loss_context_manager():
            forget_loss, _ = self._forget_loss(model, forget, reference_nll)
            forget_loss = self.gamma * forget_loss
        total = forget_loss.detach()
        self.accelerator.backward(forget_loss)
        del forget_loss, forget
        with self.compute_loss_context_manager():
            retain_loss = self.alpha * self.compute_retain_loss(model, retain)
        total = total + retain_loss.detach()
        self.accelerator.backward(retain_loss)
        del retain_loss, retain, inputs
        return total / self.args.gradient_accumulation_steps
