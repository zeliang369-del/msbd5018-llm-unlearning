"""Evaluate the original GD branches sequentially; make one combined update.

Linearity of differentiation allows backward(-gamma*NLL_forget) followed by
backward(alpha*NLL_retain) without retaining two vocabulary-logit graphs.
Neither parameters nor optimizer state changes between the two backwards.
"""
import torch
import torch.nn.functional as F
from trainer.unlearn.grad_ascent import GradAscent
from trainer.unlearn.grad_diff import GradDiff
from data.collators import DataCollatorForSupervisedDataset


def pad_batch(batch, length, pad_token):
    if 'input_ids' not in batch:
        return {key:pad_batch(value,length,pad_token) for key,value in batch.items()}
    n=length-batch['input_ids'].shape[1]
    assert n>=0,'Static padding must never truncate training text'
    return {key:F.pad(value,(0,n),value={'input_ids':pad_token,'attention_mask':0,'labels':-100}[key]) for key,value in batch.items()}


class StaticTrainingCollator(DataCollatorForSupervisedDataset):
    def __init__(self,tokenizer,length):
        super().__init__(tokenizer);self.length=length
    def __call__(self,instances):
        # Base implementation recursively calls self for nested forget/retain.
        batch=super().__call__(instances)
        return pad_batch(batch,self.length,self.tokenizer.pad_token_id)


def release_device():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()


class SequentialGradDiff(GradDiff):
    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.gamma * GradAscent.compute_loss(self, model, inputs)
        total = loss.detach()
        self.accelerator.backward(loss)
        del loss
        release_device()
        retain = {k: inputs['retain'][k] for k in ('input_ids','attention_mask','labels')}
        with self.compute_loss_context_manager():
            loss = self.alpha * self.compute_retain_loss(model, retain)
        total = total + loss.detach()
        self.accelerator.backward(loss)
        del loss, retain, inputs
        release_device()
        return total / self.args.gradient_accumulation_steps
