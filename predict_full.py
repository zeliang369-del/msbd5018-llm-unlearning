"""Live generation from a saved native full-parameter checkpoint."""
import argparse
import json
from full_core import ROOT,OUT,BASE,official_item
from train_full import load_full
from archive_checkpoint import restore
from data.collators import DataCollatorForSupervisedDataset
from evals.metrics.utils import eval_text_similarity
from omegaconf import OmegaConf


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("question");p.add_argument("--stage",choices=["baseline","GradAscent","GradDiff"],default="GradDiff");a=p.parse_args()
    directory=BASE if a.stage=="baseline" else OUT/"runs"/a.stage/"model"
    if not (directory/"model.safetensors").exists():restore(OUT/"runs"/a.stage/"checkpoint.delta.zip",directory/"model.safetensors")
    model,tokenizer,template=load_full(directory);model.eval()
    model.generation_config.eos_token_id=json.loads((BASE/"generation_config.json").read_text())["eos_token_id"]
    item=official_item(tokenizer,template,a.question,"",True)
    batch=DataCollatorForSupervisedDataset(tokenizer,padding_side="left")([item])
    result=eval_text_similarity(model,tokenizer,batch,OmegaConf.create({"do_sample":False,"top_p":None,"temperature":None,"max_new_tokens":200,"use_cache":True}))[0]
    print(json.dumps({"stage":a.stage,"question":a.question,"answer":result["generation"],"source":"live saved-checkpoint inference"},ensure_ascii=False,indent=2))
