"""Train the actual official 1B parameter space with memory-safe AdamW storage."""
import argparse
import gc
import json
import math
import inspect
import functools
from pathlib import Path
import shutil
import signal
import time
import uuid
import zipfile
import psutil
import torch
import transformers
from omegaconf import OmegaConf
from transformers import TrainingArguments,TrainerCallback
from full_core import ROOT,BASE,UPSTREAM,PROMPT_DATE,QA,TracedPairs,read_rows,save_json,file_hash
from model import get_model
from trainer.utils import seed_everything
from data.collators import DataCollatorForSupervisedDataset
from streamed_adam import StreamedAdamW
from streamed_adam import host_tensor
from safetensors import safe_open
from trainer.unlearn.grad_ascent import GradAscent
from trainer.unlearn.grad_diff import GradDiff
from memory_gd import SequentialGradDiff,StaticTrainingCollator

OUT=ROOT/"reproduction_full"
CONFIG=json.loads((OUT/"config.json").read_text())


def load_full(path=BASE):
    assert torch.backends.mps.is_available()
    cfg=OmegaConf.load(UPSTREAM/"configs/model/Llama-3.2-1B-Instruct.yaml")
    cfg.template_args.date_string=PROMPT_DATE
    cfg.model_args.pretrained_model_name_or_path=str(path)
    cfg.model_args.attn_implementation="sdpa";cfg.model_args.torch_dtype="bfloat16";cfg.model_args.local_files_only=True
    cfg.tokenizer_args.pretrained_model_name_or_path=str(BASE);cfg.tokenizer_args.local_files_only=True
    model,tokenizer=get_model(cfg);model.config.use_cache=False;model.to("mps")
    tokenizer.apply_chat_template=functools.partial(tokenizer.apply_chat_template,date_string=PROMPT_DATE)
    return model,tokenizer,cfg.template_args


class Resource(TrainerCallback):
    def __init__(self,optimizer,destination):self.optimizer=optimizer;self.destination=destination;self.rows=[]
    def on_substep_end(self,args,state,control,**kwargs):torch.mps.empty_cache()
    def on_step_end(self,args,state,control,**kwargs):
        torch.mps.synchronize()
        if state.global_step==1:
            # The first update has lr=0. It must not perturb any parameter.
            torch.mps.empty_cache()
            with safe_open(str(BASE/"model.safetensors"),framework="pt",device="cpu") as reference:
                for name,parameter in kwargs["model"].named_parameters():
                    current=host_tensor(parameter)
                    expected=reference.get_tensor(name).to(dtype=current.dtype)
                    assert torch.equal(current,expected),f"Zero-rate warmup changed {name}"
            save_json(self.destination/"zero-rate-invariance.json",{"status":"passed","all_parameters_equal_public_base":True})
        row={"step":state.global_step,"epoch":state.epoch,"optimizer_steps":self.optimizer.steps,
             "gradient_norm":getattr(self.optimizer,"last_gradient_norm",None),
             "driver_gib":torch.mps.driver_allocated_memory()/2**30,"allocated_gib":torch.mps.current_allocated_memory()/2**30,
             "CPU_gradient_buffer_gib":sum(x.numel()*4 for x in self.optimizer.shadows.values())/2**30,
             "process_rss_gib":psutil.Process().memory_info().rss/2**30,
             "available_RAM_gib":psutil.virtual_memory().available/2**30,
             "free_disk_gib":shutil.disk_usage(ROOT).free/2**30}
        self.rows.append(row);save_json(self.destination/"resources_partial.json",self.rows)
        print(json.dumps({"event":"full_step",**row}),flush=True);torch.mps.empty_cache()
        assert row["free_disk_gib"]>.5,"Scratch storage nearly exhausted"


def main():
    p=argparse.ArgumentParser();p.add_argument("method",choices=list(CONFIG["methods"]));p.add_argument("--probe",action="store_true");args=p.parse_args()
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
    destination=OUT/("probe" if args.probe else "runs")/args.method
    if (destination/"training.json").exists():
        x=json.loads((destination/"training.json").read_text());assert x["config"]==CONFIG
        assert x["status"] in ("passed_resource_probe","completed"),"Rejected diagnostic must not be reused as a successful run"
        if not args.probe:
            raw=destination/"model/model.safetensors"
            if raw.exists():assert x["checkpoint_sha256"]==file_hash(raw)
            else:
                packed=destination/"checkpoint.delta.zip"
                with zipfile.ZipFile(packed) as z:assert json.loads(z.read("metadata.json"))["original_sha256"]==x["checkpoint_sha256"]
                checked=json.loads(packed.with_suffix(".zip.verification.json").read_text())
                assert checked["actual_restore_byte_hash_verified"] and checked["archive_sha256"]==file_hash(packed)
        print("Verified full-parameter training",args.method,args.probe);return
    start=time.monotonic();seed_everything(0)
    assert transformers.__version__=="4.45.1","Use run_original.py to select the source table's framework"
    model,tokenizer,template=load_full()
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==1_235_814_400
    dataset=TracedPairs(QA(read_rows("forget01.json"),tokenizer,template),QA(read_rows("retain99.json"),tokenizer,template))
    settings=dict(output_dir=str(destination),per_device_train_batch_size=CONFIG["microbatch"],gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
                  num_train_epochs=1 if args.probe else 10,learning_rate=1e-5,weight_decay=.01,warmup_steps=1,
                  lr_scheduler_type="linear",optim="adamw_torch",adam_epsilon=1e-8,max_grad_norm=1,
                  bf16=False,fp16=False,gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False},
                  remove_unused_columns=False,dataloader_pin_memory=False,dataloader_num_workers=0,report_to="none",
                  logging_steps=1,logging_nan_inf_filter=False,save_strategy="no",eval_strategy="no",seed=0)
    settings["accelerator_config"]={"gradient_accumulation_kwargs":{"sync_with_dataloader":False}}
    if args.probe:settings["max_steps"]=3
    assert CONFIG["microbatch"]==4,"Use the original four-example loss directly"
    cls=GradAscent if args.method=="GradAscent" else SequentialGradDiff
    processing={"processing_class" if "processing_class" in inspect.signature(transformers.Trainer.__init__).parameters else "tokenizer":tokenizer}
    trainer=cls(model=model,train_dataset=dataset,**processing,args=TrainingArguments(**settings),
                data_collator=StaticTrainingCollator(tokenizer,CONFIG['training_padded_length']),**CONFIG["methods"][args.method])
    trainer.model_accepts_loss_kwargs=False
    trainer.create_optimizer()
    groups=[{k:v for k,v in g.items() if k!="params"}|{"params":g["params"]} for g in trainer.optimizer.param_groups]
    scratch=Path("/private/tmp")/f"nlp-full-state-{args.method}-{uuid.uuid4().hex[:8]}"
    optimizer=None
    try:
        print(json.dumps({"event":"prepare_FP32_states","bytes":1_235_814_400*12,"method":args.method,"probe":args.probe}),flush=True)
        # This checkpoint ties the input embedding and output head: gradients
        # are dense, so no sparse-row approximation or omission is permitted.
        optimizer=StreamedAdamW(groups,list(model.named_parameters()),scratch,capture_gradients=False,
                               reserve_bytes=1*2**30)
        trainer.optimizer=optimizer
        def clip(parameters,max_norm,norm_type=2,**kw):
            trainer.accelerator.unscale_gradients()
            return optimizer.clip_grad_norm(max_norm,norm_type)
        trainer.accelerator.clip_grad_norm_=clip
        print(json.dumps({"event":"precision","mixed_precision":trainer.accelerator.mixed_precision,
                          "scaler":None if trainer.accelerator.scaler is None else trainer.accelerator.scaler.get_scale()}),flush=True)
        resources=Resource(optimizer,destination);trainer.add_callback(resources)
        before=model.model.layers[0].self_attn.q_proj.weight[:64,:64].detach().cpu().clone()
        result=trainer.train()
        expected_steps=3 if args.probe else 10;expected_visits=96 if args.probe else 320
        consumed_visits=round(trainer.state.epoch*len(dataset))
        assert trainer.state.global_step==expected_steps and consumed_visits==expected_visits
        assert len(dataset.trace)>=consumed_visits
        assert optimizer.steps==expected_steps,"Optimizer and Trainer update counts differ"
        for entry in trainer.state.log_history:
            for key in ("loss","grad_norm"):
                if key in entry:assert math.isfinite(float(entry[key]))
        changed=int((before!=model.model.layers[0].self_attn.q_proj.weight[:64,:64].detach().cpu()).sum())
        # Dispose of optimizer storage before creating a normal HF checkpoint.
        optimizer.cleanup();gc.collect();torch.mps.empty_cache()
        summary={"status":"passed_resource_probe" if args.probe else "completed","method":args.method,"config":CONFIG,"settings":settings,
                 "steps":trainer.state.global_step,"epochs":trainer.state.epoch,"visits":consumed_visits,
                 "runtime_versions":{"torch":torch.__version__,"transformers":transformers.__version__},
                 "sampling_trace":dataset.trace[:consumed_visits],"prefetched_unused_pairs":dataset.trace[consumed_visits:],"resources":resources.rows,"log_history":trainer.state.log_history,
                 "trainable_parameters":1_235_814_400,"FP32_optimizer_state_bytes":1_235_814_400*12,
                 "scratch_disposed":not scratch.exists(),"changed_64x64_weight_slice_elements":changed,"trainer_metrics":result.metrics,
                 "optimizer_source_sha256":file_hash(ROOT/"streamed_adam.py"),"batching_source_sha256":file_hash(ROOT/"full_batching.py"),
                 "execution_adapter":{"name":"static_padding_192"+('_sequential_GD' if args.method=='GradDiff' else ''),"sha256":file_hash(ROOT/"memory_gd.py")},
                 "elapsed_seconds":time.monotonic()-start}
        if not args.probe:
            model.save_pretrained(destination/"model",safe_serialization=True);tokenizer.save_pretrained(destination/"model")
            summary["checkpoint_sha256"]=file_hash(destination/"model/model.safetensors")
        trainer.save_state();save_json(destination/"training.json",summary)
        print(json.dumps({"event":"full_train_saved","method":args.method,"probe":args.probe,"seconds":summary["elapsed_seconds"],"changed_weight_elements":changed}),flush=True)
    finally:
        if optimizer is not None:optimizer.cleanup()


if __name__=="__main__":main()
