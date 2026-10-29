"""Render phase-three results from independently verified saved records.

Use the isolated requirements-report.lock environment, after all GPU jobs finish.
This renderer never fills human labels or infers factual correctness from ROUGE.
"""
import csv
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
from statistics import mean

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'study/phase3/phase3-final-20261029'
FREEZE=ROOT/'study/phase3/robustness-20261029'
RUNS=[('Baseline','baseline-s0-cuda-v1','robust-baseline-s0-v1'),
      ('GA','ga-s0-cuda-v1','robust-ga-s0-v1'),
      ('GD','gd-a1-s0-cuda-v1','robust-gd-a1-s0-v1'),
      ('SimNPO','simnpo-s0-cuda-v1','robust-simnpo-s0-v1'),
      ('NPO','npo-s0-cuda-v1','robust-npo-s0-v1')]
FAMILIES=['direct','contextual','role_play','instruction_challenge']
COUNTS=dict(forget=40,retain=400,real_authors=100,world_facts=117)
COLORS=['#777777','#E69F00','#0072B2','#009E73','#CC79A7']


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):h.update(block)
    return h.hexdigest()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def save_csv(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True,exist_ok=True)
    inputs=[];records=[]

    def load(path):
        inputs.append(path);return read(path)

    suite=load(FREEZE/'suite.json')
    for label,run_id,prompt_id in RUNS:
        directory=ROOT/'study/runs'/run_id
        run=load(directory/'run.json');verified=load(directory/'cpu_verification.json')
        evaluation=load(directory/'evaluation/evaluation.json')
        raw=load(directory/'evaluation/TOFU_EVAL.json')
        assert verified['status']=='passed' and verified['counts']==COUNTS
        assert evaluation['status']=='completed' and evaluation['counts']==COUNTS
        assert evaluation['provenance']['limit_per_domain']==0
        assert run['spec']['seed']==0 and not run['spec']['probe']
        assert digest(dict(spec=run['spec'],commit=run['code']['commit'],sources=run['code']['source_sha256']))==run['signature']
        assert evaluation['provenance']['run_signature']==run['signature']
        assert digest(evaluation['provenance'])==evaluation['signature']
        assert verified['checkpoint_sha256']==evaluation['provenance']['checkpoint_sha256']
        computed=verified['metrics_recomputed']
        for key in ('forget_quality','model_utility'):
            assert math.isclose(computed[key],evaluation['metrics'][key],rel_tol=1e-10,abs_tol=1e-12)
        components=computed['utility_components']
        assert len(components)==9
        for key,value in components.items():
            assert math.isfinite(value) and math.isclose(value,raw[key]['agg_value'],rel_tol=1e-10,abs_tol=1e-12)
        training=None
        if label!='Baseline':
            training=load(directory/'training.json')
            assert training['status']=='completed_training_reload_checked'
            assert training['signature']==run['signature']
            assert (training['steps'],training['visits'],training['trainable_parameters'])==(10,320,1_235_814_400)
            assert training['saved_checkpoint_reloaded_and_evaluated']
        stress_dir=ROOT/'study/robustness/cuda-runs'/prompt_id
        stress=load(stress_dir/'evaluation.json');stress_check=load(stress_dir/'cpu_verification.json')
        assert stress['status']=='completed' and len(stress['rows'])==320
        assert stress_check['status']=='passed' and stress_check['total_generations']==320
        assert all(stress_check[k] for k in ('rouge_proxy_recomputed','processed_inputs_verified',
                                          'source_snapshots_verified','checkpoint_bytes_verified'))
        assert stress['provenance']['suite_sha256']==sha(FREEZE/'suite.json')
        assert stress['provenance']['model_run_signature']==run['signature']
        assert stress['provenance']['checkpoint_sha256']==verified['checkpoint_sha256']
        assert all(row['manual_fact_recovered'] is None for row in stress['rows'])
        records.append(dict(label=label,run_id=run_id,prompt_id=prompt_id,run=run,verified=verified,
            evaluation=evaluation,components=components,training=training,stress=stress,stress_check=stress_check))
    baseline=records[0]
    for r in records[1:]:
        assert r['training']['sampling_trace']==records[1]['training']['sampling_trace']
        assert r['run']['spec']['protocol']==baseline['run']['spec']['protocol']
        for key in ('profile','prompt_date'):
            assert r['evaluation']['provenance'][key]==baseline['evaluation']['provenance'][key]
        for domain in COUNTS:
            assert [(v['source_index'],v['input'],v['ground_truth']) for v in r['evaluation']['domains'][domain]]==[
                (v['source_index'],v['input'],v['ground_truth']) for v in baseline['evaluation']['domains'][domain]]
        for key in ('runtime_sources','runtime_commit','batch_size','max_new_tokens','decoding','prompt_date','seed'):
            assert r['stress']['provenance'][key]==baseline['stress']['provenance'][key]
        assert [(v['id'],v['processed_input'],v['reference_answer']) for v in r['stress']['rows']]==[
            (v['id'],v['processed_input'],v['reference_answer']) for v in baseline['stress']['rows']]
    assert records[3]['run']['spec']['beta']==4.5 and records[3]['run']['spec']['gamma']==.125
    assert records[4]['run']['spec']['beta']==.1
    assert records[4]['run']['spec']['reference_cache_sha256']=='7e31f3a4fe0282b1a5c79339169572c782b8e5eb5872b55856791ac5763c5033'
    primary=load(FREEZE/'primary-handoff.json');supplement=load(FREEZE/'supplementary-handoff.json')
    for receipt,total in ((primary,960),(supplement,640)):
        assert receipt['status']=='ready_for_human_annotation_not_annotated'
        assert receipt['total_responses']==total and receipt['labels_filled']==0
        path=FREEZE/PureWindowsPath(receipt['archive']).name
        inputs.append(path);assert sha(path)==receipt['archive_sha256']
    validation=load(ROOT/'study/phase3/npo-validation-20261029/v2/reference-verification.json')
    assert validation['status']=='passed'
    formal=load(ROOT/'study/phase3/npo-formal-20261029/execution.json')
    assert formal['status']=='completed_verified'
    for run_id in ('robust-baseline-s0-pilot-v1','robust-ga-s0-pilot-v1','robust-gd-a1-s0-pilot-v1'):
        pilot=load(ROOT/'study/robustness/cuda-runs'/run_id/'cpu_verification.json')
        assert pilot['status']=='passed' and pilot['total_generations']==40

    summary=[]
    for r in records:
        spec=r['run']['spec'];t=r['training'];metrics=r['evaluation']['metrics']
        summary.append(dict(method=r['label'],seed=0,run_id=r['run_id'],alpha=spec['alpha'] if t else '',
            gamma=spec['gamma'] if t else '',beta=spec.get('beta',''),delta=spec.get('delta',''),
            updates=t['steps'] if t else 0,visits=t['visits'] if t else 0,
            forget_quality=metrics['forget_quality'],model_utility=metrics['model_utility'],
            utility_change_from_baseline=metrics['model_utility']-baseline['evaluation']['metrics']['model_utility'],
            forget_rouge=metrics['forget_Q_A_ROUGE'],checkpoint_sha256=r['verified']['checkpoint_sha256']))
    save_csv('seed0-comparison.csv',summary)
    component_rows=[dict(component=key,**{r['label']:r['components'][key] for r in records})
                    for key in baseline['components']]
    save_csv('utility-components.csv',component_rows)
    proxies=[];paired=[]
    for r in records:
        index={(v['domain'],v['source_index'],v['family']):v for v in r['stress']['rows']}
        for domain in ('forget','retain'):
            for family in FAMILIES:
                selected=[index[domain,i,family] for i in range(40)]
                score=mean(v['rougeL_recall_proxy'] for v in selected)
                assert math.isclose(score,r['stress_check']['proxies'][domain+'/'+family],abs_tol=1e-12)
                proxies.append(dict(method=r['label'],domain=domain,family=family,n=40,
                    mean_rougeL_recall_proxy=score,mean_answer_characters=mean(len(v['generation']) for v in selected),
                    paired_mean_change_from_direct=mean(v['rougeL_recall_proxy']-index[domain,v['source_index'],'direct']['rougeL_recall_proxy'] for v in selected)))
                if family!='direct':
                    for v in selected:
                        direct=index[domain,v['source_index'],'direct']
                        paired.append(dict(method=r['label'],domain=domain,source_index=v['source_index'],family=family,
                            direct_proxy=direct['rougeL_recall_proxy'],changed_prompt_proxy=v['rougeL_recall_proxy'],
                            proxy_difference=v['rougeL_recall_proxy']-direct['rougeL_recall_proxy'],
                            direct_characters=len(direct['generation']),changed_prompt_characters=len(v['generation'])))
    save_csv('prompt-proxy-summary.csv',proxies);save_csv('paired-prompt-proxies.csv',paired)
    oracle=load(ROOT/'artifacts/tofu/retain99-reference.json')
    examples=[]
    for i in range(40):
        row=baseline['evaluation']['domains']['forget'][i]
        examples.append(dict(source_index=i,question=row['question'],reference_answer=row['reference_answer'],
            saved_generations={r['label']:r['evaluation']['domains']['forget'][i]['generation'] for r in records},
            oracle_generation=oracle['forget_Q_A_ROUGE']['value_by_index'][str(i)]['generation']))
    (OUT/'forget-answer-review.json').write_text(json.dumps(dict(scope='40 saved formal-evaluation examples; no human labels',rows=examples),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'savefig.facecolor':'white','figure.facecolor':'white'})
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.4),layout='constrained')
    for ax,key,title in zip(axes,('model_utility','forget_quality'),('Model utility (MU)','Forget quality (KS p-value)')):
        values=[r[key] for r in summary]
        bars=ax.bar([r['method'] for r in summary],values,color=COLORS,width=.63)
        ax.bar_label(bars,labels=[f'{v:.4f}' for v in values],padding=4,fontsize=9)
        ax.set_ylim(0,1);ax.set_title(title);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Five settings, same CUDA seed-0 evaluation',fontsize=13)
    fig.supxlabel('Four trained methods: 10 updates / 320 visits each; one seed, no error bars',fontsize=9)
    for suffix in ('png','pdf'):fig.savefig(OUT/('seed0-comparison.'+suffix),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.8),layout='constrained')
    for ax,prefix,title in zip(axes,('retain','ra','wf'),('Retained authors (400)','Real authors (100)','World facts (117)')):
        keys=[prefix+('_Q_A_Prob' if prefix=='retain' else '_Q_A_Prob_normalised'),prefix+'_Q_A_ROUGE',prefix+'_Truth_Ratio']
        for j,r in enumerate(records):
            ax.bar([x+(j-2)*.16 for x in range(3)],[r['components'][k] for k in keys],width=.15,color=COLORS[j],label=r['label'])
        ax.set_xticks(range(3),['Answer\nprobability','ROUGE-L\nrecall','Truth\nratio']);ax.set_title(title)
        ax.set_ylim(0,1);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Component score');axes[-1].legend(loc='upper right',fontsize=8)
    fig.suptitle('All nine components of non-forget utility',fontsize=13)
    for suffix in ('png','pdf'):fig.savefig(OUT/('utility-components.'+suffix),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.0),layout='constrained')
    for ax,domain,title in zip(axes,('forget','retain'),('Forget questions (40)','Fixed retained controls (40)')):
        values=[[next(p['mean_rougeL_recall_proxy'] for p in proxies if p['method']==r['label'] and p['domain']==domain and p['family']==f) for f in FAMILIES] for r in records]
        im=ax.imshow(values,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
        ax.set_yticks(range(5),[r['label'] for r in records]);ax.set_xticks(range(4),['Direct','Contextual','Role-play','Instruction\nchallenge'])
        ax.set_title(title)
        for i,row in enumerate(values):
            for j,v in enumerate(row):ax.text(j,i,f'{v:.3f}',ha='center',va='center',color='white' if v>.6 else 'black')
    fig.colorbar(im,ax=axes,label='Mean ROUGE-L recall proxy',shrink=.75)
    fig.suptitle('1,600 full prompt outputs: lexical overlap, not factual recovery',fontsize=13)
    for suffix in ('png','pdf'):fig.savefig(OUT/('prompt-proxies.'+suffix),dpi=180)
    plt.close(fig)
    table='\n'.join(f"| {r['method']} | {r['forget_quality']:.6f} | {r['model_utility']:.6f} | {r['utility_change_from_baseline']:+.6f} | {r['forget_rouge']:.6f} |" for r in summary)
    components_table='\n'.join('| '+row['component']+' | '+' | '.join(f"{row[r['label']]:.6f}" for r in records)+' |' for row in component_rows)
    npo=summary[-1];sim=summary[-2]
    cases='\n\n'.join(f"### Saved example {i} / 已保存案例 {i}\n\nQuestion: {examples[i]['question']}\n\nReference: {examples[i]['reference_answer']}\n\nSimNPO: {examples[i]['saved_generations']['SimNPO']}\n\nNPO: {examples[i]['saved_generations']['NPO']}" for i in (3,5,26))
    document=f'''# Phase three results / 第三阶段结果

The required experimental work is complete and independently verified. Five settings have the same seed-0 657-question evaluation. All five also have 320 full prompt outputs each: **1,600 full generations**, plus **120 separate pilot generations**. Human double annotation is prepared for the next verification phase and has not been claimed complete. Presentation: November 12, 2026.

本阶段必做实验已完成并通过独立核验。五组设置都有同协议 seed 0 的 657 题评测，每组另完成 320 条正式提示生成，合计 **1,600 条正式回答**，另有 **120 条 pilot 回答**。人工双人标注材料已准备，属于下一核验阶段，尚未完成；展示日期为 2026-11-12。

## Five-model comparison / 五组对照

| Method | FQ: KS p-value | MU | MU change from baseline | Forget ROUGE-L recall |
|---|---:|---:|---:|---:|
{table}

![Five settings](seed0-comparison.png)

SimNPO keeps MU near baseline ({sim['model_utility']:.6f} versus {summary[0]['model_utility']:.6f}), while target facts still remain in saved answers. Its FQ is {sim['forget_quality']:.6f}; forgetting-text overlap falls to {sim['forget_rouge']:.6f}. NPO has MU {npo['model_utility']:.6f}, FQ {npo['forget_quality']:.6f}, and forgetting-text overlap {npo['forget_rouge']:.6f}. These are results at the tested settings and budget, not evidence that one method generally wins or achieves the same amount of forgetting.

SimNPO 的 MU 接近起点，但保存回答仍有目标知识残留。NPO 的 MU 为 {npo['model_utility']:.6f}、FQ 为 {npo['forget_quality']:.6f}，遗忘题文本重合度为 {npo['forget_rouge']:.6f}。这些是本次设置和预算下的观察，不能据此宣布某方法普遍最好，也不能视为达成相同遗忘程度后的比较。

FQ is a two-sided KS-test p-value against the imported retain-only Oracle, not a percentage forgotten. A low p-value alone does not identify under- versus over-forgetting. MU is the harmonic mean of nine non-forget scores, not overall accuracy. ROUGE is lexical overlap, not factual correctness. The imported Oracle was not retrained locally.

FQ 是与导入的 retain-only Oracle 分布比较的双侧 KS 检验 p 值，不是遗忘百分比；低 p 值本身不能区分遗忘不足或过度。MU 是九项非遗忘分数的调和平均值，不是整体准确率。ROUGE 衡量文本重合，不等于事实正确；Oracle 没有在本地重训。

## Utility components / 能力损伤细分

| Component | Baseline | GA | GD | SimNPO | NPO |
|---|---:|---:|---:|---:|---:|
{components_table}

![Nine components](utility-components.png)

Read the three domains separately: retained fictional authors, real authors, and world facts. A small aggregate MU change may hide a larger change in one component. Compare all nine values before attributing utility preservation to a method.

分别观察保留虚构作者、真实作者和世界知识三个领域。综合 MU 变化小，不代表每项能力都不受影响；判断方法保留能力时应同时看九项分量。

## Prompt stress tests / 四类提示测试

![Prompt proxies](prompt-proxies.png)

Each model uses the identical frozen suite: 40 forget questions and 40 fixed retained controls, each asked as Direct, Contextual, Role-play, and Instruction challenge. Greedy decoding, batch size 8, a 200-token output cap, the fixed prompt date, exact processed inputs, and runtime source hashes match across all five models. The CPU audit verifies counts, source snapshots, checkpoint bytes, input texts, and recomputed proxies.

各模型使用完全相同的冻结测试集：40 遗忘题、40 固定保留对照，各用 Direct、Contextual、Role-play、Instruction challenge 四类问法。贪心解码、batch=8、最多生成 200 token、固定日期、实际输入文本和运行源码哈希一致。CPU 核验覆盖数量、源码快照、检查点文件、输入文本及代理指标重算。

`prompt-proxy-summary.csv` gives all 40 model/domain/family means and mean answer lengths. `paired-prompt-proxies.csv` gives 1,200 within-question changes from Direct. These are paired lexical-overlap comparisons; longer answers, paraphrases, partial facts, contradictions, and refusals can change ROUGE without tracking factual recovery. No automatic threshold is used to create human labels or a factual-recovery rate.

上述 CSV 分别给出 40 组均值及回答长度、1,200 个同题相对 Direct 的变化。它们只是配对文本重合对照；回答变长、改述、部分正确、矛盾及拒答都可能影响 ROUGE，不能直接解释为事实恢复。因此没有用自动阈值生成人工标签或事实恢复率。

## Matched protocol and limits / 对照协议与限制

All methods start from the same pinned TOFU-trained Llama-3.2-1B checkpoint. Training uses all 1,235,814,400 parameters, BF16 CUDA weights and dense FP32 streamed optimizer state, microbatch 4, accumulation 8, 10 optimizer updates, and 320 visits. All four methods have exactly matching seed-0 sampling traces. Complete evaluation uses 40 forget, 400 retained-author, 100 real-author, and 117 world-fact questions, with identical profiles and processed texts. Fresh-process checkpoint reloads and independent metric recomputation passed.

各方法从同一固定 TOFU 训练后的 Llama-3.2-1B 起点出发，使用全部 1,235,814,400 个参数、BF16 CUDA 权重与稠密 FP32 流式优化器状态，microbatch=4、梯度累积=8，完成十次更新、320 次样本访问，四种训练方法的 seed 0 采样配对完全一致。完整评测为 40 遗忘、400 保留作者、100 真实作者及 117 世界知识题，评价配置和输入文本一致；新进程重载和指标独立重算均通过。

Loss settings differ by method: GA uses its original unweighted objective; GD uses alpha=gamma=1; SimNPO uses alpha=1, gamma=0.125, beta=4.5, delta=0; NPO uses alpha=gamma=1 and beta=0.1 with the fixed initial-reference cache. SimNPO uses length-normalized forgetting likelihood; NPO uses summed token NLL against a reference. Matching data and compute budget does not match objective scale or achieved forgetting. No new-method multi-seed stability claim follows; existing GA/GD three-seed phase-two results remain separate. RMU and additional coefficient sweeps are optional follow-up work.

方法损失及权重不同：GA 使用原始未加权目标；GD 为 alpha=gamma=1；SimNPO 为 alpha=1、gamma=0.125、beta=4.5、delta=0；NPO 为 alpha=gamma=1、beta=0.1，并使用固定初始参照缓存。SimNPO 遗忘项按长度归一化，NPO 使用对参照的 token NLL 求和。数据和预算相同，不代表目标尺度或遗忘程度相同；不能据此断言新方法多种子稳定。第二阶段 GA/GD 三种子结果单独保留，RMU 和新增权重搜索为可选后续工作。

## NPO integration findings / NPO 接入记录

Original PR #4 commit: `5f792a1350c378f3c1f6c3bf3e5632a499be2238`. Tested local fix commit: `673ac4f70b978916113d414b67beab8a875bfe1b`. Real QA tests exposed batch-dimension/collator fingerprint mismatches and CPU inputs sent to a CUDA reference model. A singleton BF16 cache then failed the strict cached-versus-online test; recomputing references in the same batch size 4 as training resolved it. The loss formula was preserved. Twenty-seven method tests and real Trainer/Accelerate CPU/GPU loss-gradient comparisons passed, including checks without clipping to expose scaling errors.

真实 QA 路径发现并修复了批次维度、collator 缓存指纹及 CUDA 参照输入设备问题。之后单样本 BF16 缓存未通过严格的在线对照；改为与训练相同的 batch=4 生成参照后通过。NPO 损失公式保持不变。27 项方法测试及真实 Trainer/Accelerate 的 CPU/GPU 损失梯度对照通过，梯度对照禁用裁剪以便发现缩放错误。

The full cache covers all 40 forget samples; 80 ordered/shuffled online comparisons passed. Maximum NLL deviation was 1.9073486328125e-06 and initial NPO gradient-weight deviation was 1.1920928955078125e-07. The separate three-update probe passed before formal training. Original failures, the full cache, tests, logs, and a patch relative to PR #4 are preserved under `../npo-validation-20261029/`; no failed cache result was silently replaced.

完整缓存覆盖全部 40 个遗忘样本；顺序及打乱共 80 次在线对照通过，最大 NLL 偏差约 1.91e-6，初始 NPO 梯度权重偏差约 1.19e-7；正式训练前另通过三次更新试跑。原始失败记录、缓存、测试、日志及相对 PR #4 的修补文件均保留。

## Next handoffs / 接下来给组员什么

1. B / annotation team: original `annotation-primary-20261029.zip` contains 960 Baseline/GA/GD answers, including all four prompt types. Send it now for rule calibration and independent annotation. Each suggested A/B/C/D slot has 240 forget judgments.
2. B: forward the NPO integration patch and this validated result, preserving the submitted SimNPO/NPO authorship. The supplemental `annotation-supplementary-20261029.zip` adds 640 SimNPO/NPO answers, with 160 forget judgments per suggested slot. No member is claimed to have accepted a slot.
3. C: freeze the labeling rule and coordinate independent disagreement resolution. Across both packs, 800 forget answers need 1,600 independent judgments; retained spotchecks are limited controls. Match the Baseline Direct answer from this prompt protocol, report unclear/excluded counts, and use N/A if the conditional denominator is zero.
4. D: use this five-model table, all nine components, saved answer examples, and the clearly labeled proxy plots for the report/PPT. Add human factual-recovery results only after returned labels and adjudication. Confirmed contributions and current team tasks are documented in docs/TEAM_PROGRESS.md.

先把原三模型 960 条包给负责标注的组员，让她校准规则并开展独立标注；再把 NPO 修补文件及验证结果给 B，两种新方法 640 条包作为补充。C 统一标注边界并处理独立判断后的分歧；两个包共 800 条遗忘回答、1,600 次判断，保留抽检不能外推整体准确率。D 先使用指标、九项分量和明确标为代理的提示图准备报告/PPT，收到人工复核结果后再添加事实恢复结论。已确认贡献与当前任务见 docs/TEAM_PROGRESS.md。

## Saved illustrative answers / 已保存案例

These three selected examples illustrate the saved behavior, not a representative accuracy estimate or completed human annotation. All 40 questions are in `forget-answer-review.json`. The TOFU people and works are fictional benchmark content.

以下三个选例仅展示保存行为，不代表总体准确率或已完成人工标注。全部 40 题在 JSON 中，人物和作品为 TOFU 虚构基准内容。

{cases}

## Reproduce the report / 重建报告

From the repository root, run `.venv-report/Scripts/python.exe -X utf8 report_phase3_cuda.py`. It reads saved records, checks cross-run pairing and signatures, and recreates CSV/PNG/PDF outputs. `report-manifest.json` binds input/output bytes and the renderer. Full checkpoint CPU verification additionally needs the matching local model files; CUDA checkpoint weights are not included in Git. Phase-two records remain separate.

在仓库根目录运行上述命令可检查保存记录、跨实验配对和签名，并重建图表；manifest 记录输入、输出及绘图脚本哈希。完整检查点 CPU 核验另需对应本地模型文件，CUDA 权重不进入 Git；第二阶段记录单独保留。
'''
    (OUT/'RESULTS.md').write_text(document,encoding='utf-8')
    names=['RESULTS.md','seed0-comparison.csv','utility-components.csv','prompt-proxy-summary.csv',
           'paired-prompt-proxies.csv','forget-answer-review.json']
    names += [stem+'.'+suffix for stem in ('seed0-comparison','utility-components','prompt-proxies') for suffix in ('png','pdf')]
    manifest=dict(status='verified_phase3_experiments',date_local='2026-10-29',
        formal_settings=5,new_formal_training_runs=2,full_prompt_generations=1600,pilot_generations=120,
        human_factual_labels='pending_independent_human_review',sampling_trace_matched=True,
        standard_profiles_and_inputs_matched=True,prompt_runtime_and_inputs_matched=True,
        renderer_sha256=sha(Path(__file__)),requirements_report_sha256=sha(ROOT/'requirements-report.lock'),
        input_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in inputs},
        generated_sha256={name:sha(OUT/name) for name in names})
    (OUT/'report-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2));print(OUT/'RESULTS.md')


if __name__=='__main__':main()
