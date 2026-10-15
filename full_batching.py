"""Split an original four-example token-mean loss into memory-safe microbatches."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent / "open-unlearning/src"))
from trainer.unlearn.grad_ascent import GradAscent
from trainer.unlearn.grad_diff import GradDiff


class FourExampleLoss:
    def get_batch_samples(self,epoch_iterator,num_batches,device):
        samples,num_items=super().get_batch_samples(epoch_iterator,num_batches,device)
        for start in range(0,len(samples),4):
            group=samples[start:start+4]
            counts={key:[int((batch[key]["labels"][...,1:]!=-100).sum()) for batch in group] for key in ("forget","retain")}
            for index,batch in enumerate(group):
                batch["micro_loss_weights"]={key:len(group)*counts[key][index]/sum(counts[key]) for key in counts}
        return samples,num_items

    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        weights=inputs.pop("micro_loss_weights")
        if isinstance(self,GradDiff):
            old_alpha,old_gamma=self.alpha,self.gamma
            self.alpha=old_alpha*weights["retain"];self.gamma=old_gamma*weights["forget"]
            try: return super().compute_loss(model,inputs,return_outputs,num_items_in_batch)
            finally: self.alpha,self.gamma=old_alpha,old_gamma
        result=super().compute_loss(model,inputs,return_outputs,num_items_in_batch)
        if return_outputs: return result[0]*weights["forget"],result[1]
        return result*weights["forget"]


class FullGradAscent(FourExampleLoss,GradAscent): pass
class FullGradDiff(FourExampleLoss,GradDiff): pass
