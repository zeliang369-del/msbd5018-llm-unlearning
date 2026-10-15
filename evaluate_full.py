"""Complete original TOFU core evaluation for saved native 1B checkpoints."""
import argparse
import gc
import json
import math
from pathlib import Path
import time
import torch
import transformers
from omegaconf import OmegaConf
from full_core import ROOT,OUT,BASE,DATA,PROMPT_DATE,SOURCES,PREFIX,UTILITY_KEYS,read_rows,official_item,aggregate,save_json,file_hash,digest
from train_full import load_full,CONFIG
from data.collators import DataCollatorForSupervisedDataset
from evals.metrics.utils import evaluate_probability,eval_text_similarity


class Inference:
    def __init__(self,model):self.model=model;self.device=model.device
    def __call__(self,**kwargs):
        # The original evaluator independently computes CE; model.loss is unused.
        kwargs.pop("labels",None)
        output=self.model(**kwargs);output.logits=output.logits.float();return output
    def generate(self,*args,**kwargs):return self.model.generate(*args,**kwargs)


def evaluate(stage,limit=0):
    assert transformers.__version__=="4.45.1"
    profile=json.loads((OUT/"evaluation_profile.json").read_text())
    model_path=BASE if stage=="baseline" else OUT/"runs"/stage/"model"
    checkpoint=model_path/"model.safetensors"
    provenance={"stage":stage,"limit":limit,"framework":transformers.__version__,"torch":torch.__version__,
                "upstream_commit":CONFIG["upstream_commit"],"checkpoint_sha256":file_hash(checkpoint),
                "evaluator_sha256":file_hash(Path(__file__)),"core_sha256":file_hash(ROOT/"full_core.py"),
                "profile":profile,"prompt_date":PROMPT_DATE,"manifest":json.loads((DATA/"manifest.json").read_text())}
    signature=digest(provenance)
    destination=OUT/("calibration" if limit else "runs")/stage
    output=destination/"evaluation.json"
    if output.exists():
        assert json.loads(output.read_text())["signature"]==signature
        print("Verified evaluation",stage,limit,flush=True);return
    partial=destination/"partial.json";cache={"signature":signature,"domains":{}}
    if partial.exists():cache=json.loads(partial.read_text());assert cache["signature"]==signature
    start=time.monotonic()
    model,tokenizer,template=load_full(model_path);model.eval()
    model.generation_config.eos_token_id=json.loads((BASE/"generation_config.json").read_text())["eos_token_id"]
    wrapped=Inference(model)
    probability_collator=DataCollatorForSupervisedDataset(tokenizer)
    generation_collator=DataCollatorForSupervisedDataset(tokenizer,padding_side="left")
    # Check the optimization against the original forward before using it.
    fixture=read_rows("forget01.json")[0]
    item=official_item(tokenizer,template,fixture["question"],fixture["answer"])
    batch={k:v.unsqueeze(0).to("mps") for k,v in item.items()}
    with torch.inference_mode():
        reference=model(**batch).logits
        optimized=wrapped(**batch).logits
        assert torch.equal(reference.float(),optimized),"Removing unused model.loss changed logits"
    save_json(destination/"forward-parity.json",{"status":"passed","logits_equal_with_and_without_unused_loss":True})
    del reference,optimized,batch;torch.mps.empty_cache()
    for domain,source in SOURCES.items():
        source_rows=read_rows(source);indices=list(range(min(limit,len(source_rows)) if limit else len(source_rows)))
        done={r["source_index"] for r in cache["domains"].setdefault(domain,[])}
        pending=[i for i in indices if i not in done]
        for offset in range(0,len(pending),profile["batch_size"]):
            chunk=pending[offset:offset+profile["batch_size"]];rows=[source_rows[i] for i in chunk]
            fields=[("answer",0),("paraphrased_answer",0)]+[("perturbed_answer",i) for i in range(len(rows[0]["perturbed_answer"]))]
            scores={}
            for key,answer_index in fields:
                if key=="paraphrased_answer" and all(r.get(key,r["answer"])==r["answer"] for r in rows):
                    scores[(key,answer_index)]=scores[("answer",0)];continue
                answers=[r["perturbed_answer"][answer_index] if key=="perturbed_answer" else r.get(key,r["answer"]) for r in rows]
                entries=[official_item(tokenizer,template,r["question"],a) for r,a in zip(rows,answers)]
                scores[(key,answer_index)]=evaluate_probability(wrapped,probability_collator(entries))
                torch.mps.empty_cache()
            prompts=[official_item(tokenizer,template,r["question"],r["answer"],True) for r in rows]
            generated=eval_text_similarity(wrapped,tokenizer,generation_collator(prompts),OmegaConf.create({"do_sample":False,
                       "top_p":None,"temperature":None,"max_new_tokens":profile["max_new_tokens"],"use_cache":True}))
            for j,(index,row) in enumerate(zip(chunk,rows)):
                orig=scores[("answer",0)][j];para=scores[("paraphrased_answer",0)][j]
                wrong=[scores[("perturbed_answer",i)][j] for i in range(len(row["perturbed_answer"]))]
                result={"source_index":index,"question":row["question"],"reference_answer":row["answer"],
                        "answer_nll":orig["avg_loss"],"answer_probability":orig["prob"],
                        "paraphrased_nll":para["avg_loss"],"paraphrased_probability":para["prob"],
                        "wrong_nlls":[w["avg_loss"] for w in wrong],"wrong_probabilities":[w["prob"] for w in wrong],**generated[j]}
                assert all(math.isfinite(v) for v in [result["answer_nll"],result["paraphrased_nll"],*result["wrong_nlls"]])
                cache["domains"][domain].append(result)
            save_json(partial,cache)
            if len(cache["domains"][domain])%40==0 or limit:
                print(json.dumps({"event":"eval_full","stage":stage,"domain":domain,
                                  "complete":len(cache["domains"][domain]),"total":len(indices)}),flush=True)
            torch.mps.empty_cache()
    logs=aggregate(cache["domains"])
    metrics={k:logs[k]["agg_value"] for k in ("forget_quality","model_utility","forget_truth_ratio","forget_Q_A_ROUGE")}
    result={"status":"completed","stage":stage,"signature":signature,"provenance":provenance,"metrics":metrics,
            "domains":cache["domains"],"counts":{k:len(v) for k,v in cache["domains"].items()},
            "elapsed_seconds":time.monotonic()-start,"scope":"calibration only" if limit else "All 657 core evaluation questions"}
    save_json(output,result);save_json(destination/"TOFU_EVAL.json",logs)
    print(json.dumps({"event":"evaluated_full","stage":stage,"metrics":metrics,"seconds":result["elapsed_seconds"]}),flush=True)
    del wrapped,model;gc.collect();torch.mps.empty_cache()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["baseline","GradAscent","GradDiff"]);p.add_argument("--limit",type=int,default=0);a=p.parse_args();evaluate(a.stage,a.limit)
