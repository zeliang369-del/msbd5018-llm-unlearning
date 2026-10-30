"""Small shared layer for the source-anchored full-parameter experiment."""
import hashlib
import json
import os
from pathlib import Path
import sys
import torch
from torch.utils.data import Dataset

ROOT=Path(__file__).resolve().parent
UPSTREAM=ROOT/"open-unlearning-original"
BASE=ROOT/"artifacts/model"
DATA=ROOT/"artifacts/tofu"
OUT=ROOT/"reproduction_full"
PROMPT_DATE="10 Apr 2025"
os.environ.setdefault("HF_HUB_OFFLINE","1")
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
sys.path.insert(0,str(UPSTREAM/"src"))
from data.utils import preprocess_chat_instance
from data.unlearn import ForgetRetainDataset

SOURCES={"forget":"forget01_perturbed.json","retain":"retain_perturbed.json",
         "real_authors":"real_authors_perturbed.json","world_facts":"world_facts_perturbed.json"}
PREFIX={"forget":"forget","retain":"retain","real_authors":"ra","world_facts":"wf"}
UTILITY_KEYS=[f"{p}_{suffix}" for p in ("retain","ra","wf")
              for suffix in (("Q_A_Prob","Q_A_ROUGE","Truth_Ratio") if p=="retain" else ("Q_A_Prob_normalised","Q_A_ROUGE","Truth_Ratio"))]


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*2**20),b""):h.update(block)
    return h.hexdigest()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def read_rows(name):return [json.loads(line) for line in (DATA/name).read_text().splitlines() if line.strip()]


def official_item(tokenizer,template,question,answer,generation=False):
    item=preprocess_chat_instance(tokenizer,template,[question],[answer],512,predict_with_generate=generation)
    assert len(item["input_ids"])<=512 and len(item["labels"])<=512,"No silent truncation"
    assert int(item["input_ids"][0])==tokenizer.bos_token_id and int(item["input_ids"][1])!=tokenizer.bos_token_id
    return item


class QA(Dataset):
    def __init__(self,rows,tokenizer,template):self.items=[official_item(tokenizer,template,r["question"],r["answer"]) for r in rows]
    def __len__(self):return len(self.items)
    def __getitem__(self,index):return self.items[index]


class TracedPairs(ForgetRetainDataset):
    def __init__(self,forget,retain):super().__init__(forget,retain,anchor="forget");self.trace=[]
    def __getitem__(self,index):
        retain_index=torch.randint(0,len(self.retain),(1,)).item()
        self.trace.append({"forget_index":index,"retain_index":retain_index})
        return {"forget":self.forget[index],"retain":self.retain[retain_index]}


def aggregate(domains):
    import numpy as np
    from evals.metrics.memorization import truth_ratio,probability_w_options,hm_aggregate
    from evals.metrics.privacy import ks_test
    logs={}
    for domain,rows in domains.items():
        p=PREFIX[domain]
        def block(probability,nll):
            values={str(r["source_index"]):{"prob":r[probability],"avg_loss":r[nll]} for r in rows}
            return {"agg_value":float(np.mean([v["prob"] for v in values.values()])),"value_by_index":values}
        orig=block("answer_probability","answer_nll");para=block("paraphrased_probability","paraphrased_nll");wrong=block("wrong_probabilities","wrong_nlls")
        logs[f"{p}_Q_A_Prob"]=orig
        logs[f"{p}_truth_ratio" if domain=="forget" else f"{p}_Truth_Ratio"]=truth_ratio._metric_fn(None,
            pre_compute={"correct":para if domain in ("forget","retain") else orig,"wrong":wrong},
            aggregator="closer_to_1_better" if domain=="forget" else "true_better")
        logs[f"{p}_Q_A_ROUGE"]={"agg_value":float(np.mean([r["rougeL_recall"] for r in rows])),
               "value_by_index":{str(r["source_index"]):{k:r[k] for k in ("generation","ground_truth","rougeL_recall")} for r in rows}}
        if domain not in ("forget","retain"):
            logs[f"{p}_Q_A_Prob_normalised"]=probability_w_options._metric_fn(None,pre_compute={"correct":orig,"wrong":wrong})
    reference=json.loads((DATA/"retain99-reference.json").read_text())
    logs["forget_quality"]=ks_test._metric_fn(None,pre_compute={"forget":logs["forget_truth_ratio"]},
             reference_logs={"retain_model_logs":{"retain":reference["forget_truth_ratio"]}})
    logs["model_utility"]=hm_aggregate._metric_fn(None,pre_compute={k:logs[k] for k in UTILITY_KEYS})
    return logs
