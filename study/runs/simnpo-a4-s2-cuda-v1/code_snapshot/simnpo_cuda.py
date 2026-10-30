"""CUDA-friendly sequential training adapter for the upstream SimNPO loss."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_UPSTREAM_SRC = Path(__file__).resolve().parent / "open-unlearning-original" / "src"
if str(_UPSTREAM_SRC) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM_SRC))

try:
    from trainer.utils import compute_batch_nll
except ModuleNotFoundError as error:
    if error.name != "omegaconf":
        raise

    def compute_batch_nll(model, inputs):
        outputs = model(**inputs)
        logits = outputs.logits[..., :-1, :].contiguous()
        labels = inputs["labels"][..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            logits.transpose(-1, -2), labels, ignore_index=-100, reduction="none"
        ).sum(dim=-1)
        return loss, outputs

try:
    from trainer.unlearn.simnpo import SimNPO as _SimNPOBase
except ModuleNotFoundError as error:
    if error.name != "omegaconf":
        raise

    class _SimNPOBase:
        """Minimal CPU-test fallback when the upstream Trainer dependencies are absent."""

        def compute_retain_loss(self, model, retain_inputs):
            return model(**retain_inputs).loss


class CudaSimNPO(_SimNPOBase):
    """Compute SimNPO's two branches with only one autograd graph at a time.

    The loss formula is intentionally the one in the upstream SimNPO class:
    per-example shifted token NLL, divided by the unshifted label mask count,
    followed by the delta/beta transform and the weighted retain NLL.
    """

    def _validate_token_counts(self, labels):
        counts = (labels != -100).sum(dim=-1)
        if torch.any(counts == 0):
            raise ValueError("SimNPO requires at least one non-ignored label per sample")
        shifted_counts = (labels[..., 1:] != -100).sum(dim=-1)
        if torch.any(shifted_counts == 0):
            raise ValueError("SimNPO requires at least one non-ignored shifted label per sample")
        return counts

    def _compute_forget_loss(self, model, forget_inputs):
        counts = self._validate_token_counts(forget_inputs["labels"])
        forget_loss, forget_outputs = compute_batch_nll(model, forget_inputs)
        forget_loss = forget_loss / counts - self.delta
        forget_loss = -F.logsigmoid(self.beta * forget_loss).mean() * 2 / self.beta
        return forget_loss, forget_outputs

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = {
            key: inputs["forget"][key]
            for key in ("input_ids", "attention_mask", "labels")
        }
        retain_inputs = {
            key: inputs["retain"][key]
            for key in ("input_ids", "attention_mask", "labels")
        }
        self._validate_token_counts(retain_inputs["labels"])
        forget_loss, forget_outputs = self._compute_forget_loss(model, forget_inputs)
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Accumulate the two mathematically additive gradients sequentially."""
        model.train()
        inputs = self._prepare_inputs(inputs)

        forget_inputs = {
            key: inputs["forget"][key]
            for key in ("input_ids", "attention_mask", "labels")
        }
        retain_inputs = {
            key: inputs["retain"][key]
            for key in ("input_ids", "attention_mask", "labels")
        }
        # Validate both branches before the first backward so invalid retain
        # input cannot leave a partially accumulated Forget gradient.
        self._validate_token_counts(forget_inputs["labels"])
        self._validate_token_counts(retain_inputs["labels"])
        with self.compute_loss_context_manager():
            forget_loss, _ = self._compute_forget_loss(model, forget_inputs)
            forget_loss = self.gamma * forget_loss
        total = forget_loss.detach()
        self.accelerator.backward(forget_loss)
        del forget_loss, forget_inputs

        with self.compute_loss_context_manager():
            retain_loss = self.alpha * self.compute_retain_loss(model, retain_inputs)
        total = total + retain_loss.detach()
        self.accelerator.backward(retain_loss)
        del retain_loss, retain_inputs, inputs

        return total / self.args.gradient_accumulation_steps
