"""Exercise each real entrypoint branch through checkpoint/metadata saving.

Tiny local tensors and a controlled three-update trainer replace expensive GPU
work; Python branch scope, serialization and saved evidence remain real.
"""
from types import SimpleNamespace
import json
from pathlib import Path
import pytest
import torch
from safetensors.torch import save_file,load_file
import train_cuda
import cuda_runtime
import full_core
import memory_gd
import simnpo_cuda
import npo_cuda
import npo_data
import npo_reference
import transformers
import trainer.unlearn.grad_ascent as ga_module


@pytest.mark.parametrize('method',['GradAscent','GradDiff','SimNPO','NPO'])
def test_every_method_saves_checkpoint_and_training_metadata(method,tmp_path,monkeypatch):
    weight=torch.nn.Parameter(torch.ones(8,8))
    name='model.layers.0.self_attn.q_proj.weight'

    class Model:
        model=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(q_proj=SimpleNamespace(weight=weight)))])
        def parameters(self):return iter([weight])
        def named_parameters(self):return iter([(name,weight)])
        def save_pretrained(self,path,**kwargs):
            path.mkdir(parents=True,exist_ok=True);save_file({name:weight.detach().contiguous()},str(path/'model.safetensors'))

    class Tokenizer:
        def save_pretrained(self,path):(path/'tokenizer.json').write_text('{}')

    class QA:
        def __init__(self,rows,*args):self.rows=rows
        def __len__(self):return len(self.rows)

    class Pairs:
        def __init__(self,forget,retain):self.forget=forget;self.retain=retain;self.trace=[]
        def __len__(self):return len(self.forget)

    class Optimizer:
        def __init__(self,*args,**kwargs):
            self.steps=0;self.last_gradient_norm=1.;self.last_applied_lrs=[0.,0.]
        def cleanup(self):pass
        def clip_grad_norm(self,*args):return self.last_gradient_norm

    dispatched=[]
    class Trainer:
        def __init__(self,model,tokenizer,train_dataset,args,**kwargs):
            dispatched.append(kwargs);self.dataset=train_dataset;self.args=args;self.callbacks=[]
            self.state=SimpleNamespace(epoch=0.,global_step=0,log_history=[])
            self.accelerator=SimpleNamespace(unscale_gradients=lambda:None)
        def create_optimizer(self):self.optimizer=SimpleNamespace(param_groups=[dict(params=[weight],lr=0.)])
        def add_callback(self,callback):self.callbacks.append(callback)
        def train(self):
            for step in (1,2,3):
                self.optimizer.steps=step;self.optimizer.last_applied_lrs=[0.,0.] if step==1 else [1e-5,1e-5]
                if step>1:
                    with torch.no_grad():weight.add_(.01)
                self.state.global_step=step;self.state.epoch=step*32/40
                self.state.log_history.append(dict(loss=1.,grad_norm=1.,step=step))
                for callback in self.callbacks:callback.on_step_end(self.args,self.state,SimpleNamespace(should_training_stop=False))
            self.dataset.trace=[dict(forget_index=i%40,retain_index=i%3960) for i in range(96)]
            return SimpleNamespace(metrics=dict(train_runtime=.01,train_loss=1.))
        def save_state(self):
            (destination/'trainer_state.json').write_text(json.dumps(dict(global_step=self.state.global_step)))

    base=tmp_path/'base';base.mkdir();save_file({name:weight.detach().clone()},str(base/'model.safetensors'))
    (base/'tokenizer.json').write_text('{}')
    destination=tmp_path/method;destination.mkdir()
    monkeypatch.setattr(train_cuda,'preflight',lambda *args,**kwargs:None)
    monkeypatch.setattr(train_cuda,'PARAMETERS',weight.numel())
    monkeypatch.setattr(cuda_runtime,'configure',lambda seed:None)
    monkeypatch.setattr(cuda_runtime,'load_cuda',lambda:(Model(),Tokenizer(),None))
    monkeypatch.setattr(cuda_runtime,'CudaGradDiff',Trainer)
    monkeypatch.setattr(cuda_runtime,'CudaStreamedAdamW',Optimizer)
    monkeypatch.setattr(simnpo_cuda,'CudaSimNPO',Trainer)
    monkeypatch.setattr(npo_cuda,'CudaNPO',Trainer)
    monkeypatch.setattr(ga_module,'GradAscent',Trainer)
    monkeypatch.setattr(full_core,'BASE',base)
    monkeypatch.setattr(full_core,'QA',QA)
    monkeypatch.setattr(full_core,'TracedPairs',Pairs)
    monkeypatch.setattr(full_core,'read_rows',lambda name:[None]*(40 if name=='forget01.json' else 3960))
    monkeypatch.setattr(npo_data,'NPOTracedPairs',Pairs)
    monkeypatch.setattr(npo_data,'reference_samples',lambda *args:{'0':{}})
    monkeypatch.setattr(npo_reference,'load_reference_cache',lambda *args:object())
    monkeypatch.setattr(memory_gd,'StaticTrainingCollator',lambda *args:object())
    monkeypatch.setattr(npo_data,'NPOTrainingCollator',lambda *args:object())
    monkeypatch.setattr(transformers,'TrainingArguments',lambda **kwargs:SimpleNamespace(**kwargs))
    for attr in ('synchronize','reset_peak_memory_stats','empty_cache'):
        monkeypatch.setattr(torch.cuda,attr,lambda:None)
    for attr in ('max_memory_allocated','max_memory_reserved'):
        monkeypatch.setattr(torch.cuda,attr,lambda:0)
    args=SimpleNamespace(method=method,seed=1,alpha=1.,gamma=.125 if method=='SimNPO' else 1.,
                         beta=4.5 if method=='SimNPO' else .1,delta=0.,reference_cache=tmp_path/'cache.json')
    spec=dict(protocol=dict(training_padded_length=192),updates=3)
    train_cuda.train(args,destination,spec,'tiny-checkpoint-path')
    summary=json.loads((destination/'training.json').read_text())
    assert summary['status']=='trained_pending_reload_evaluation'
    assert (summary['steps'],summary['visits'])==(3,96)
    assert (destination/'trainer_state.json').is_file()
    assert train_cuda.file_hash(destination/'model/model.safetensors')==summary['checkpoint_sha256']
    assert torch.equal(load_file(str(destination/'model/model.safetensors'))[name],weight.detach())
    assert (destination/'zero-rate-invariance.json').is_file()
    assert len(dispatched)==1
    if method=='SimNPO':assert dispatched[0]['beta']==4.5 and dispatched[0]['delta']==0.
    if method=='NPO':assert dispatched[0]['beta']==.1 and 'reference_cache' in dispatched[0]
