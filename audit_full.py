"""Verify the frozen source, data, metric algebra, runtime, and prompt calibration."""
import json
from pathlib import Path
import tarfile
import numpy as np
import torch
import transformers
import accelerate
from full_core import ROOT,OUT,BASE,DATA,UPSTREAM,UTILITY_KEYS,PREFIX,file_hash,save_json
from evals.metrics.memorization import hm_aggregate,truth_ratio


def main():
    assert transformers.__version__=="4.45.1" and accelerate.__version__=="0.34.2"
    import evals.metrics.utils,trainer.unlearn.grad_diff,data.utils
    for module in (evals.metrics.utils,trainer.unlearn.grad_diff,data.utils):
        assert Path(module.__file__).resolve().is_relative_to(UPSTREAM),module.__file__
    protected=["src/trainer/unlearn/grad_ascent.py","src/trainer/unlearn/grad_diff.py","src/data/utils.py",
               "src/data/unlearn.py","src/evals/metrics/utils.py","src/evals/metrics/memorization.py","src/evals/metrics/privacy.py"]
    hashes={}
    with tarfile.open(OUT/"upstream-original.tar.gz") as archive:
        prefix=archive.getmembers()[0].name.split('/')[0]
        for name in protected:
            expected=archive.extractfile(prefix+'/'+name).read()
            assert (UPSTREAM/name).read_bytes()==expected,name
            hashes[name]=file_hash(UPSTREAM/name)
    assert file_hash(BASE/"model.safetensors")=="35b0cd9f7db2735538ff78b4879fab61b23b7ca316df27b059585bf656700a8d"
    manifest=json.loads((DATA/"manifest.json").read_text())
    for entry in manifest["files"]:assert file_hash(DATA/entry["file"])==entry["sha256"]
    refs={}
    for role,filename in [("baseline","full-reference.json"),("oracle","retain99-reference.json")]:
        assert file_hash(DATA/filename)=={'baseline':'558fdd814fe88bfcd29b29ff5d14ba8fbc946c86fbe8b10eba867acb98719155','oracle':'aa6a504c44487fe5174c1828e4c3bfe997de89856c95cba721dc833bbb37aee3'}[role]
        source=json.loads((DATA/filename).read_text())
        mu=hm_aggregate._metric_fn(None,pre_compute={k:source[k] for k in UTILITY_KEYS})["agg_value"]
        assert abs(mu-source["model_utility"]["agg_value"])<1e-12
        for domain,p in PREFIX.items():
            correct=f"{p}_Q_A_PARA_Prob" if domain in ("forget","retain") else f"{p}_Q_A_Prob"
            key=f"{p}_truth_ratio" if domain=="forget" else f"{p}_Truth_Ratio"
            value=truth_ratio._metric_fn(None,pre_compute={"correct":source[correct],"wrong":source[f"{p}_Q_A_PERT_Prob"]},
                       aggregator="closer_to_1_better" if domain=="forget" else "true_better")["agg_value"]
            assert abs(value-source[key]["agg_value"])<1e-12
        refs[role]={"MU_recomputed":float(mu),"file_sha256":file_hash(DATA/filename)}
    calibration={}
    path=OUT/"calibration/baseline/evaluation.json"
    if path.exists():
        local=json.loads(path.read_text());reference=json.loads((DATA/"full-reference.json").read_text())
        for domain,rows in local["domains"].items():
            p=PREFIX[domain];nll=[];rouge=[]
            for r in rows:
                i=str(r["source_index"]);expected=reference[f"{p}_Q_A_ROUGE"]["value_by_index"][i]
                assert r["input"]==expected["input"] and r["ground_truth"]==expected["ground_truth"]
                nll.append(abs(r["answer_nll"]-reference[f"{p}_Q_A_Prob"]["value_by_index"][i]["avg_loss"]))
                rouge.append(abs(r["rougeL_recall"]-expected["rougeL_recall"]))
            calibration[domain]={"questions":len(rows),"input_target_match":True,"NLL_MAE":float(np.mean(nll)),"ROUGE_MAE":float(np.mean(rouge))}
    report={"status":"passed","original_commit":json.loads((OUT/"config.json").read_text())["upstream_commit"],
            "unmodified_core_files":hashes,"runtime":{"torch":torch.__version__,"transformers":transformers.__version__,"accelerate":accelerate.__version__},
            "metric_recalculation":refs,"calibration":calibration}
    save_json(OUT/"audit.json",report);print(json.dumps(report,indent=2))


if __name__=="__main__":main()
