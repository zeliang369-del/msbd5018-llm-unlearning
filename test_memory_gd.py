"""Compare actual original GD autograd with memory-adapted training_step."""
import copy
import json
from contextlib import nullcontext
from types import SimpleNamespace
import torch
import full_core
from transformers import LlamaConfig,LlamaForCausalLM
from trainer.unlearn.grad_diff import GradDiff
from memory_gd import SequentialGradDiff,pad_batch


def main():
    reports=[]
    for device,dtype in [('cpu',torch.float32),('mps',torch.bfloat16)]:
        torch.manual_seed(91)
        cfg=LlamaConfig(vocab_size=127,hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,attention_dropout=0.0,tie_word_embeddings=True)
        cfg._attn_implementation='sdpa';cfg.use_cache=False
        first=LlamaForCausalLM(cfg).to(device=device,dtype=dtype)
        second=copy.deepcopy(first)
        for model in (first,second):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        inputs={}
        for key,n in [('forget',43),('retain',59)]:
            ids=torch.randint(0,127,(4,n),device=device)
            labels=ids.clone();labels[:,:7]=-100
            inputs[key]={'input_ids':ids,'attention_mask':torch.ones_like(ids),'labels':labels}
        joint=SimpleNamespace(gamma=1.,alpha=1.,retain_loss_type='NLL')
        joint.compute_retain_loss=lambda model,retain_inputs:GradDiff.compute_retain_loss(joint,model,retain_inputs)
        original=GradDiff.compute_loss(joint,first,inputs)
        (original/8).backward()
        sequential=SimpleNamespace(gamma=1.,alpha=1.,retain_loss_type='NLL',args=SimpleNamespace(gradient_accumulation_steps=8))
        sequential._prepare_inputs=lambda x:x
        sequential.compute_loss_context_manager=nullcontext
        sequential.accelerator=SimpleNamespace(backward=lambda loss:(loss/8).backward())
        sequential.compute_retain_loss=lambda model,retain_inputs:GradDiff.compute_retain_loss(sequential,model,retain_inputs)
        adapted=SequentialGradDiff.training_step(sequential,second,pad_batch(inputs,64,1))
        a=torch.cat([p.grad.detach().cpu().float().reshape(-1) for p in first.parameters()])
        b=torch.cat([p.grad.detach().cpu().float().reshape(-1) for p in second.parameters()])
        error=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a))
        assert error < (2e-6 if device=='cpu' else .02),error
        assert abs(float(original.detach()/8)-float(adapted))<(.002 if device=='mps' else 1e-7)
        reports.append({'device':device,'dtype':str(dtype),'relative_gradient_error':error,'original_loss':float(original.detach()),'adapted_loss':float(adapted*8)})
        del first,second,inputs
        if device=='mps':torch.mps.empty_cache()
    report={'status':'passed','reference':'unmodified original GradDiff.compute_loss, real tiny Llama architecture with actual non-reentrant gradient checkpointing; adapted path also statically right-padded with ignored labels','tests':reports}
    full_core.save_json(full_core.OUT/'sequential-gd-verification.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
