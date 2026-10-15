"""Check losses AND parameter gradients against an unsplit official microbatch."""
import json
from types import SimpleNamespace
from pathlib import Path
import torch
from full_batching import FullGradAscent,FullGradDiff
from trainer.unlearn.grad_ascent import GradAscent
from trainer.unlearn.grad_diff import GradDiff


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.table=torch.nn.Parameter(torch.randn(23,23))
    def forward(self,input_ids,attention_mask,labels):
        logits=self.table[input_ids]
        loss=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,23),labels[:,1:].reshape(-1),ignore_index=-100)
        return SimpleNamespace(loss=loss,logits=logits)


def make(length):
    ids=torch.randint(0,23,(1,12));labels=ids.clone();labels[:,:12-length]=-100
    return dict(input_ids=ids,labels=labels,attention_mask=torch.ones_like(ids))


def setup(cls):
    obj=cls.__new__(cls)
    if issubclass(cls,GradDiff): obj.alpha=obj.gamma=1.;obj.retain_loss_type="NLL";obj.ref_model=None
    return obj


def main():
    torch.manual_seed(9)
    pairs=[{"forget":make(f),"retain":make(r)} for f,r in zip([2,4,7,9],[8,3,6,4])]
    combined={k:{field:torch.cat([p[k][field] for p in pairs]) for field in pairs[0][k]} for k in ("forget","retain")}
    counts={k:[int((p[k]["labels"][:,1:]!=-100).sum()) for p in pairs] for k in ("forget","retain")}
    findings={}
    for original,split in [(GradAscent,FullGradAscent),(GradDiff,FullGradDiff)]:
        model=Model();loss=setup(original).compute_loss(model,combined);loss.backward();expected=model.table.grad.clone()
        model.zero_grad();actual=0.
        trainer=setup(split)
        for i,pair in enumerate(pairs):
            one={**pair,"micro_loss_weights":{k:4*counts[k][i]/sum(counts[k]) for k in counts}}
            value=trainer.compute_loss(model,one)/4;value.backward();actual+=float(value.detach())
        error=float((model.table.grad-expected).abs().max())
        assert error<1e-7 and abs(actual-float(loss.detach()))<1e-6
        findings[original.__name__]={"loss_error":abs(actual-float(loss.detach())),"gradient_max_absolute_error":error}
    report={"status":"passed","purpose":"Preserve official four-example token-mean objective when splitting into size-one microbatches","methods":findings}
    Path('reproduction_v2/full-batch-verification.json').write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report))


if __name__=="__main__":main()
