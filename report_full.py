"""Compare actual full-parameter runs to independent official references."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import ks_2samp
from full_core import ROOT,OUT,DATA,PREFIX,save_json,file_hash,digest

COUNTS={"forget":40,"retain":400,"real_authors":100,"world_facts":117}


def main():
    config=json.loads((OUT/"config.json").read_text())
    results={};training={};traces=[]
    for method in ("baseline","GradAscent","GradDiff"):
        directory=OUT/"runs"/method
        result=json.loads((directory/"evaluation.json").read_text())
        assert result["counts"]==COUNTS and result["status"]=="completed"
        assert result["signature"]==digest(result["provenance"])
        assert result["provenance"]["framework"]=="4.45.1"
        assert result["provenance"]["evaluator_sha256"]==file_hash(ROOT/"evaluate_full.py")
        assert result["provenance"]["core_sha256"]==file_hash(ROOT/"full_core.py")
        for domain,rows in result["domains"].items():assert sorted(r["source_index"] for r in rows)==list(range(COUNTS[domain]))
        results[method]=result
        if method!="baseline":
            record=json.loads((directory/"training.json").read_text())
            assert record["status"]=="completed" and record["config"]==config
            assert record["steps"]==10 and record["epochs"]==8 and record["visits"]==320
            assert record["scratch_disposed"] and record["trainable_parameters"]==1_235_814_400
            assert record["checkpoint_sha256"]==result["provenance"]["checkpoint_sha256"]==file_hash(directory/"model/model.safetensors")
            assert record["optimizer_source_sha256"]==file_hash(ROOT/"streamed_adam.py")
            assert record['execution_adapter']['sha256']==file_hash(ROOT/'memory_gd.py')
            assert json.loads((OUT/'sequential-gd-verification.json').read_text())['status']=='passed'
            assert all(r["optimizer_steps"]==r["step"] for r in record["resources"])
            assert json.loads((directory/"zero-rate-invariance.json").read_text())["all_parameters_equal_public_base"]
            traces.append(digest(record["sampling_trace"]));training[method]=record
    assert len(set(traces))==1,"Unlearning methods used different data draws"
    utility_keys=[('retain_Q_A_Prob','保留资料·答案概率'),('retain_Q_A_ROUGE','保留资料·ROUGE'),('retain_Truth_Ratio','保留资料·truth score'),
                  ('ra_Q_A_Prob_normalised','真实作者·候选归一化概率'),('ra_Q_A_ROUGE','真实作者·ROUGE'),('ra_Truth_Ratio','真实作者·truth score'),
                  ('wf_Q_A_Prob_normalised','世界事实·候选归一化概率'),('wf_Q_A_ROUGE','世界事实·ROUGE'),('wf_Truth_Ratio','世界事实·truth score')]
    components={}
    for method in results:
        original=json.loads((OUT/'runs'/method/'TOFU_EVAL.json').read_text())
        components[method]={key:original[key]['agg_value'] for key,_ in utility_keys}
        assert abs(9/sum(1/v for v in components[method].values())-results[method]['metrics']['model_utility'])<1e-12
    for method in ('GradAscent','GradDiff'):
        for domain,rows in results[method]['domains'].items():
            start={r['source_index']:r for r in results['baseline']['domains'][domain]}
            for row in rows:
                assert row['input']==start[row['source_index']]['input']
                assert row['ground_truth']==start[row['source_index']]['ground_truth']
    raw_reference={role:json.loads((DATA/filename).read_text()) for role,filename in (("baseline","full-reference.json"),("oracle","retain99-reference.json"))}
    oracle=raw_reference["oracle"]
    gold_ratios=[v["score"] for v in oracle["forget_truth_ratio"]["value_by_index"].values()]
    references={}
    for role,record in raw_reference.items():
        ratio=[v["score"] for v in record["forget_truth_ratio"]["value_by_index"].values()]
        references[role]={"forget_quality":float(ks_2samp(ratio,gold_ratios).pvalue),
                          "model_utility":record["model_utility"]["agg_value"],
                          "forget_truth_ratio":record["forget_truth_ratio"]["agg_value"],
                          "forget_Q_A_ROUGE":record["forget_Q_A_ROUGE"]["agg_value"],"origin":"official CUDA log, imported"}
    official_table={"GradAscent":{"forget_quality":.27,"model_utility":.33,"forget_truth_ratio":.59},
                    "GradDiff":{"forget_quality":.77,"model_utility":.43,"forget_truth_ratio":.57}}
    base=results["baseline"]["metrics"];ga=results["GradAscent"]["metrics"];gd=results["GradDiff"]["metrics"]
    claims={"GD_preserves_more_utility_than_GA":{"direction_matches":gd["model_utility"]>ga["model_utility"],"difference":gd["model_utility"]-ga["model_utility"],"official_difference":.10}}
    for method in ("GradAscent","GradDiff"):
        for metric in ("forget_quality","forget_truth_ratio"):
            delta=results[method]["metrics"][metric]-base[metric]
            claims[f"{method}_improves_{metric}"]={"direction_matches":delta>0,"difference":delta,
                "official_difference":official_table[method][metric]-references["baseline"][metric]}
    calibration={}
    for domain,rows in results["baseline"]["domains"].items():
        p=PREFIX[domain];errors=[];rouge=[]
        for r in rows:
            i=str(r["source_index"]);original=raw_reference["baseline"][f"{p}_Q_A_ROUGE"]["value_by_index"][i]
            assert r["input"]==original["input"] and r["ground_truth"]==original["ground_truth"]
            errors.append(abs(r["answer_nll"]-raw_reference["baseline"][f"{p}_Q_A_Prob"]["value_by_index"][i]["avg_loss"]))
            rouge.append(abs(r["rougeL_recall"]-original["rougeL_recall"]))
        calibration[domain]={"questions":len(rows),"processed_text_match":True,"NLL_MAE":float(np.mean(errors)),"ROUGE_MAE":float(np.mean(rouge))}
    resources={m:{"train_seconds":t["elapsed_seconds"],"eval_seconds":results[m]["elapsed_seconds"],
                  "trainer_wall_seconds":t['trainer_metrics']['train_runtime'],
                  "max_step_end_driver_gib":max(r["driver_gib"] for r in t["resources"]),
                  "max_step_end_rss_gib":max(r["process_rss_gib"] for r in t["resources"]),
                  "minimum_free_disk_gib":min(r["free_disk_gib"] for r in t["resources"])} for m,t in training.items()}
    summary={"status":"completed","config":config,"local_results":{m:r["metrics"] for m,r in results.items()},
             "official_references":references,"official_full_parameter_table":official_table,"tested_claims":claims,
             "baseline_calibration":calibration,"same_sampling_trace_sha256":traces[0],"resources":resources,
             "utility_components":components,"utility_component_labels":dict(utility_keys),
             "scope":"Official 1B/forget01/GA-GD subset, native full parameter space; hardware and documented runtime adaptations",
             "audit":json.loads((OUT/"audit.json").read_text()),
             "execution_verification":json.loads((OUT/'sequential-gd-verification.json').read_text())}
    if (OUT/'timing-context.json').exists():summary['timing_context']=json.loads((OUT/'timing-context.json').read_text())
    save_json(OUT/"summary.json",summary)
    table={m:{**r["metrics"],"origin":"本机完整评价"} for m,r in results.items()}
    table["oracle"]={**references["oracle"],"origin":"官方 CUDA 记录导入"}
    domains={}
    for domain,rows in results["baseline"]["domains"].items():
        p=PREFIX[domain];gold=oracle[f"{p}_Q_A_ROUGE"]["value_by_index"]
        mapped={m:{r["source_index"]:r for r in result["domains"][domain]} for m,result in results.items()}
        domains[domain]=[{"index":r["source_index"],"question":r["question"],"reference":r["reference_answer"],
              "answers":{**{m:mp[r["source_index"]]["generation"] for m,mp in mapped.items()},"oracle":gold[str(r["source_index"])]["generation"]}} for r in rows]
    save_json(OUT/"demo.json",{"summary":summary,"table":table,"domains":domains})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    names=["baseline","GradAscent","GradDiff","oracle"]
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    for ax,key,label in zip(axes,["forget_quality","model_utility","forget_truth_ratio"],["Forget Quality (KS p-value)","Full Model Utility","Forget truth score"]):
        values=[table[n][key] for n in names]
        bars=ax.bar(names,values,color=["#88949d","#c77c55","#317d88","#c3c3bf"])
        bars[-1].set_hatch("//");ax.bar_label(bars,labels=[f"{v:.3g}" for v in values],padding=3,fontsize=8)
        ax.scatter([1.32,2.32],[official_table[n][key] for n in names[1:3]],color="black",marker="x",label="Official reference")
        ax.set_title(label);ax.tick_params(axis="x",rotation=18);ax.set_ylim(0,1.05)
        if key=="forget_quality":ax.set_yscale("log");ax.set_ylim(max(1e-25,min(values)/3),1.5)
    axes[0].legend(fontsize=8);fig.suptitle("TOFU 1B / forget01 / full-parameter: local MPS vs official CUDA")
    fig.tight_layout();fig.savefig(OUT/"comparison.png",dpi=170);plt.close(fig)
    lines=["# 官方 1B 全参数轻量复现实验结果","",
           "本次选择官方实验中的 1B / forget01 / GA-GD 子集，保留全部参数空间与原始算法。Mac 适配集中在优化器存储和设备；没有用 LoRA 的结果替代原全参数结论。","",
           "## 完整对照结果","","| 模型 | FQ（p 值） | 完整 MU | forget truth score | 遗忘题 ROUGE | 来源 |","|---|---:|---:|---:|---:|---|"]
    for name in names:
        r=table[name];lines.append(f"| {name} | {r['forget_quality']:.6g} | {r['model_utility']:.6f} | {r['forget_truth_ratio']:.6f} | {r['forget_Q_A_ROUGE']:.6f} | {r['origin']} |")
    lines += ["","本机三组各评价 657 题，处理后的输入与目标文本三组逐题一致，并与官方基础模型记录核对一致。Oracle 从未用 forget01 数据进行 TOFU 微调，其记录来自官方，未声称由本机重训。FQ 不是遗忘准确率，高 p 值也不证明完全遗忘。","",
              "## 与官方表逐项比较","","| 方法 | 指标 | 本机 | 官方表 | 差值 |","|---|---|---:|---:|---:|"]
    for m,r in official_table.items():
        for key,expected in r.items():
            actual=table[m][key];lines.append(f"| {m} | {key} | {actual:.6f} | {expected:.2f} | {actual-expected:+.6f} |")
    lines += ["","官方表为四舍五入结果，具体来源为固定提交的 docs/repro.md。单 MPS、SDPA、Torch 版本和单机累积与双 CUDA/ZeRO3 有数值差异，不把两者说成逐位相同。","",
              "## 哪些结论复现了",""]
    for name,c in claims.items():
        lines.append(f"- {name}：{'变化方向一致' if c['direction_matches'] else '本次未复现该方向'}；本机差值 {c['difference']:+.6g}，官方参照差值 {c['official_difference']:+.6g}。")
    lines += ["","方向比较来自一次固定 seed=0 的实验，不是统计显著性结论；效应大小也必须看上表。不能因流程执行成功就自动宣称所有论文结论复现成功。","",
              "## 为什么 GD 的总 MU 没有高于 GA","","| 组成指标 | 起点 | GA | GD | GD−GA |","|---|---:|---:|---:|---:|"]
    for key,label in utility_keys:
        a,b,c=(components[m][key] for m in ('baseline','GradAscent','GradDiff'))
        lines.append(f'| {label} | {a:.6f} | {b:.6f} | {c:.6f} | {c-b:+.6f} |')
    lines += ['', '本机 GD 在保留作者资料的三项指标上均略高于 GA，但真实作者的 ROUGE 从 GA 的约 0.603 降至约 0.414，其他部分通用知识指标也下降。完整 MU 是九项调和平均，因此 GD 的总分反而略低。这支持保留约束在训练领域有作用，但没有复现官方表中总体 MU 更高的结论。具体差异原因没有通过单独的控制实验定位，不能直接归咎于硬件。', '',
              "## 训练过程是否与原设置对应","","使用原源码的 GA/GD 损失、官方完整 40 题遗忘集与 3960 题保留池。相同 seed、相同采样顺序。每次原生 microbatch=4，单机累积 8 次达到有效 batch 32。","",
              "配置为原来的名义 10 epochs；原 Transformers 4.45.1 在此数据量下按旧规则计算更新上限，本机实际为 8 次完整数据遍历、10 次优化、320 次训练访问。记录实际优化器调用数与 Trainer 计数一致，未按最终遗忘效果选择检查点。官方表没有提供逐步训练日志，本机与原双设备运行的逐样本抽取及归约顺序未逐步对齐，不能声称训练轨迹完全相同。","",
              "全部 1,235,814,400 个参数参与优化。AdamW 主权重和动量为 FP32，暂存在文件映射中，更新后还原到模型参数类型；临时状态在保存模型前清理。标准 AdamW 数学对照与零学习率全参数不变检查通过。","",
              "GA、GD 均统一右填充至 192 tokens（训练题最长 170），新增位置 attention=0、labels=-100，不截断原文也不计入损失。GD 把两个原始损失分支依次反传，再做同一次优化器更新，避免同时保留两个计算图。两个反传之间不改变参数；目标仍为 -gamma*NLL_forget + alpha*NLL_retain。已与未改动的原 GD 在真实小型 Llama 结构上核对梯度，包含与正式训练一致的非重入梯度检查点：填充与依次反传的 FP32 CPU 梯度相对误差约 1.4e-7，BF16 MPS 约 0.23%，损失一致。这种舍入顺序差异也属于复现边界。","",
              "## 基础模型逐题校准","","| 领域 | 题数 | 输入与目标文本 | NLL 平均绝对差 | ROUGE 平均绝对差 |","|---|---:|---|---:|---:|"]
    for d,c in calibration.items():lines.append(f"| {d} | {c['questions']} | 与官方逐题一致 | {c['NLL_MAE']:.6f} | {c['ROUGE_MAE']:.6f} |")
    lines += ["","## 真实回答示例",""]
    for index in (0,5,6):
        r=next(r for r in domains["forget"] if r["index"]==index)
        lines += [f"### 遗忘题 {index}：{r['question']}","",f"参考答案：{r['reference']}",""]
        for m in names:lines += [f"**{m}**：{r['answers'][m]}",""]
    lines += ["## 本机资源与运行时间","","| 方法 | 脚本训练计时秒数 | Trainer 墙钟秒数 | 完整评价计时秒数 | 步骤末最高 driver GiB | 步骤末最高 RSS GiB | 最低磁盘余量 GiB |","|---|---:|---:|---:|---:|---:|---:|"]
    for m,r in resources.items():lines.append(f"| {m} | {r['train_seconds']:.1f} | {r['trainer_wall_seconds']:.1f} | {r['eval_seconds']:.1f} | {r['max_step_end_driver_gib']:.2f} | {r['max_step_end_rss_gib']:.2f} | {r['minimum_free_disk_gib']:.2f} |")
    lines += ["","这些是步骤末采样，不是严格瞬时峰值；driver 包括分配缓存，不能直接等同于物理 RAM 使用量。优化器临时容量约 14.83GB，训练结束已清理。","",
              "## 复现边界与失败记录","","本机保留 Torch 2.9.1，原依赖为 2.4.1；Transformers 4.45.1、Accelerate 0.34.2 与原表时期一致。日期固定为 10 Apr 2025 以匹配参考输入。","",
              "此前 LoRA 只观察到很弱的遗忘，列为诊断，不冒充正式全参数结果。分块跨设备转换曾产生错误梯度、资源测试曾触发空间保护，GD 双计算图运行也曾出现异常梯度并触发内存不足，均独立留档。正式路径使用同类型整体拷贝和同步、分支依次反传、串行 GPU 锁及磁盘余量保护，并保存对应核验记录。","",
              "本项目没有复现所有模型、所有方法和全部隐私攻击指标；不保证知识不可恢复。所有结果可回到 runs/ 中的逐题记录、训练日志和模型哈希核查。"]
    lines += ['', '脚本使用 time.monotonic 记录训练/评价耗时，Trainer 使用自己的墙钟 train_runtime；训练脚本计时在保存检查点前结束。GD 第 6 至 9 步之间包含较长后台间隔，两种时钟记录明显不同，因此同时保留。没有测量纯 GPU 活跃计算时间，不能将任一计时都称为纯计算时间。恢复工作期间已附加空闲睡眠防止措施；这也不保证电脑合盖后继续运行。']
    (OUT/"实验结果.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"results":summary["local_results"],"claims":claims},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
