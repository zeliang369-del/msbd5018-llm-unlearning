"""Produce the complete three-seed/five-weight report after the grid audit passes."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean,stdev
from collections import defaultdict

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'study/phase3/grid-20261029'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):h.update(block)
    return h.hexdigest()


def save_csv(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    audit=read(OUT/'independent-audit.json');plan=read(OUT/'plan.json')
    assert audit['status']=='passed' and audit['formal_settings']==audit['prompt_settings']==49
    inputs=[OUT/'plan.json',OUT/'independent-audit.json'];results=[];proxies=[];paired=[];groups=defaultdict(list)
    suite_hash=sha(ROOT/'study/phase3/robustness-20261029/suite.json')
    runtime=None;canonical={}
    for item in plan['matrix']:
        directory=ROOT/'study/runs'/item['run_id']
        filenames=[directory/'run.json',directory/'cpu_verification.json',directory/'evaluation/evaluation.json']
        run,verified,evaluation=map(read,filenames);inputs.extend(filenames)
        assert verified['status']=='passed' and verified['counts']==dict(forget=40,retain=400,real_authors=100,world_facts=117)
        assert verified['checkpoint_sha256']==evaluation['provenance']['checkpoint_sha256']
        for key in ('forget_quality','model_utility'):
            assert math.isclose(evaluation['metrics'][key],verified['metrics_recomputed'][key],abs_tol=1e-12)
        if item['method']!='baseline':
            tpath=directory/'training.json';training=read(tpath);inputs.append(tpath)
            assert training['steps']==10 and training['visits']==320
            trace=training['sampling_trace']
            if item['seed'] in canonical:assert trace==canonical[item['seed']]
            else:canonical[item['seed']]=trace
        prompt_dir=ROOT/'study/robustness/cuda-runs'/item['prompt_run_id']
        prompt_paths=[prompt_dir/'evaluation.json',prompt_dir/'cpu_verification.json']
        prompt,pcheck=map(read,prompt_paths);inputs.extend(prompt_paths)
        assert pcheck['status']=='passed' and len(prompt['rows'])==320 and pcheck['total_generations']==320
        assert prompt['provenance']['suite_sha256']==suite_hash
        assert prompt['provenance']['checkpoint_sha256']==verified['checkpoint_sha256']
        profile={key:prompt['provenance'][key] for key in ('runtime_commit','runtime_sources','batch_size','max_new_tokens','decoding','prompt_date','seed')}
        if runtime is None:runtime=profile
        else:assert profile==runtime
        assert all(r['manual_fact_recovered'] is None for r in prompt['rows'])
        metrics=verified['metrics_recomputed']
        record=dict(method=item['method'],seed=item['seed'],alpha=item['alpha'],gamma=item['gamma'],
            beta=item.get('beta',''),delta=item.get('delta',''),run_id=item['run_id'],
            fq=metrics['forget_quality'],mu=metrics['model_utility'],forget_rouge=evaluation['metrics']['forget_Q_A_ROUGE'],
            checkpoint_sha256=verified['checkpoint_sha256'],**metrics['utility_components'])
        results.append(record);groups[item['method'],item['alpha']].append(record)
        prompt_index={(r['domain'],r['source_index'],r['family']):r for r in prompt['rows']}
        assert len(prompt_index)==320
        for domain in ('forget','retain'):
            for family in ('direct','contextual','role_play','instruction_challenge'):
                selected=[r for r in prompt['rows'] if r['domain']==domain and r['family']==family]
                assert len(selected)==40
                proxies.append(dict(method=item['method'],seed=item['seed'],alpha=item['alpha'],domain=domain,family=family,
                    mean_rougeL_recall_proxy=mean(r['rougeL_recall_proxy'] for r in selected),
                    mean_answer_characters=mean(len(r['generation']) for r in selected),
                    paired_mean_change_from_direct=mean(r['rougeL_recall_proxy']-prompt_index[domain,r['source_index'],'direct']['rougeL_recall_proxy'] for r in selected)))
                if family!='direct':
                    for r in selected:
                        direct=prompt_index[domain,r['source_index'],'direct']
                        paired.append(dict(method=item['method'],seed=item['seed'],alpha=item['alpha'],domain=domain,
                            source_index=r['source_index'],family=family,direct_proxy=direct['rougeL_recall_proxy'],
                            changed_prompt_proxy=r['rougeL_recall_proxy'],
                            difference=r['rougeL_recall_proxy']-direct['rougeL_recall_proxy'],
                            direct_characters=len(direct['generation']),changed_prompt_characters=len(r['generation'])))
    assert len(results)==49 and len(proxies)==49*8 and len(paired)==11760 and set(canonical)=={0,1,2}
    summary=[]
    for (method,alpha),values in sorted(groups.items()):
        assert len(values)==(1 if method=='baseline' else 3)
        row=dict(method=method,alpha=alpha,n=len(values))
        for key in [k for k in results[0] if k in ('fq','mu','forget_rouge') or k.endswith(('Prob','normalised','ROUGE','Ratio'))]:
            scores=[r[key] for r in values]
            row[key+'_mean']=mean(scores);row[key+'_sample_sd']=stdev(scores) if len(scores)>1 else ''
        summary.append(row)
    proxy_groups=defaultdict(list)
    for r in proxies:proxy_groups[r['method'],r['alpha'],r['domain'],r['family']].append(r)
    proxy_summary=[]
    for (method,alpha,domain,family),values in sorted(proxy_groups.items()):
        assert len(values)==(1 if method=='baseline' else 3)
        scores=[r['mean_rougeL_recall_proxy'] for r in values]
        proxy_summary.append(dict(method=method,alpha=alpha,domain=domain,family=family,n=len(values),
            rouge_proxy_mean=mean(scores),rouge_proxy_sample_sd=stdev(scores) if len(scores)>1 else '',
            mean_answer_characters=mean(r['mean_answer_characters'] for r in values),
            paired_mean_change_from_direct=mean(r['paired_mean_change_from_direct'] for r in values)))
    assert len(proxy_summary)==136
    save_csv('all-49-settings.csv',results);save_csv('three-seed-summary.csv',summary);save_csv('prompt-proxy-summary.csv',proxies)
    save_csv('paired-prompt-proxies.csv',paired);save_csv('prompt-three-seed-summary.csv',proxy_summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    methods=['GradDiff','SimNPO','NPO'];colors=['#0072B2','#009E73','#CC79A7']
    fig,axes=plt.subplots(2,3,figsize=(13,7.5),layout='constrained')
    for col,(method,color) in enumerate(zip(methods,colors)):
        values=sorted([r for r in summary if r['method']==method],key=lambda r:r['alpha'])
        assert len(values)==5
        for ax,key,label in ((axes[0,col],'mu','MU'),(axes[1,col],'fq','FQ (KS p-value)')):
            ax.errorbar([r['alpha'] for r in values],[r[key+'_mean'] for r in values],
                yerr=[r[key+'_sample_sd'] for r in values],fmt='o',capsize=4,color=color)
            ax.set_xscale('log',base=2);ax.set_xticks(plan['alphas'],[str(v).rstrip('0').rstrip('.') for v in plan['alphas']])
            ax.set_xlabel('Retain weight alpha');ax.set_ylabel(label);ax.set_title(method)
            lower=min(r[key+'_mean']-r[key+'_sample_sd'] for r in values)
            upper=max(r[key+'_mean']+r[key+'_sample_sd'] for r in values)
            ax.grid(alpha=.2);ax.set_ylim(min(-.05,lower-.05),max(1.05,upper+.05))
    fig.suptitle('Five retain weights, three seeds each: mean ± sample SD (n=3)',fontsize=13)
    for ext in ('png','pdf'):fig.savefig(OUT/('weight-seed-summary.'+ext),dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
    palette={'baseline':'#777777','GradAscent':'#E69F00','GradDiff':'#0072B2','SimNPO':'#009E73','NPO':'#CC79A7'}
    for method,color in palette.items():
        selected=[r for r in results if r['method']==method]
        ax.scatter([r['mu'] for r in selected],[r['fq'] for r in selected],label=method,color=color,alpha=.75)
    ax.set(xlabel='Model utility (MU)',ylabel='Forget quality (KS p-value)',xlim=(0,1),ylim=(0,1.05))
    ax.legend();ax.grid(alpha=.2);ax.set_title('49 observed settings; no fitted trade-off frontier')
    for ext in ('png','pdf'):fig.savefig(OUT/('all-settings.'+ext),dpi=180)
    plt.close(fig)
    ordered=sorted(summary,key=lambda r:(['baseline','GradAscent','GradDiff','SimNPO','NPO'].index(r['method']),r['alpha']))
    labels=[r['method']+(f" / a={r['alpha']:g}" if r['method'] in methods else '') for r in ordered]
    component_keys=['retain_Q_A_Prob','retain_Q_A_ROUGE','retain_Truth_Ratio','ra_Q_A_Prob_normalised',
                    'ra_Q_A_ROUGE','ra_Truth_Ratio','wf_Q_A_Prob_normalised','wf_Q_A_ROUGE','wf_Truth_Ratio']
    values=[[r[k+'_mean'] for k in component_keys] for r in ordered]
    fig,ax=plt.subplots(figsize=(12,9),layout='constrained')
    im=ax.imshow(values,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
    ax.set_yticks(range(17),labels)
    ax.set_xticks(range(9),['Retain P','Retain R','Retain T','Real P','Real R','Real T','World P','World R','World T'],rotation=35,ha='right')
    for i,row in enumerate(values):
        for j,v in enumerate(row):ax.text(j,i,f'{v:.3f}',ha='center',va='center',fontsize=8,color='white' if v>.65 else 'black')
    fig.colorbar(im,ax=ax,label='Mean component score',shrink=.7)
    ax.set_title('Nine utility components: trained settings n=3; fixed baseline n=1')
    for ext in ('png','pdf'):fig.savefig(OUT/('utility-component-grid.'+ext),dpi=180)
    plt.close(fig)
    families=['direct','contextual','role_play','instruction_challenge']
    fig,axes=plt.subplots(1,2,figsize=(13,9),layout='constrained')
    lookup={(r['method'],r['alpha'],r['domain'],r['family']):r['rouge_proxy_mean'] for r in proxy_summary}
    for ax,domain in zip(axes,('forget','retain')):
        values=[[lookup[r['method'],r['alpha'],domain,f] for f in families] for r in ordered]
        im=ax.imshow(values,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
        ax.set_yticks(range(17),labels);ax.set_xticks(range(4),['Direct','Context','Role-play','Challenge'])
        ax.set_title('Forget questions' if domain=='forget' else 'Fixed retained controls')
        for i,row in enumerate(values):
            for j,v in enumerate(row):ax.text(j,i,f'{v:.3f}',ha='center',va='center',fontsize=8,color='white' if v>.65 else 'black')
    fig.colorbar(im,ax=axes,label='Mean text-overlap proxy',shrink=.7)
    fig.suptitle('Prompt-family proxies across weights/seeds; not factual recovery',fontsize=13)
    for ext in ('png','pdf'):fig.savefig(OUT/('prompt-proxy-grid.'+ext),dpi=180)
    plt.close(fig)
    table='\n'.join(f"| {r['method']} | {r['alpha']:g} | {r['n']} | {r['fq_mean']:.6f} | {r['fq_sample_sd'] if r['fq_sample_sd']=='' else format(r['fq_sample_sd'],'.6f')} | {r['mu_mean']:.6f} | {r['mu_sample_sd'] if r['mu_sample_sd']=='' else format(r['mu_sample_sd'],'.6f')} |" for r in summary)
    document=f'''# Full weight/seed grid / 完整权重与种子矩阵

All 49 settings passed complete 657-question evaluation and checkpoint/source verification. This includes 48 formal training runs, ten updates / 320 visits each. Ten earlier formal runs were reused and 38 were newly trained. GD, SimNPO, and NPO each cover five retain weights × three seeds; GA covers three seeds; the fixed baseline is evaluated once.

全部 49 组配置完成 657 题评测及检查点、来源核验，含 48 次正式训练：复用原有 10 次，新增 38 次。GD、SimNPO、NPO 各完成五档保留权重×三个种子；GA 三个种子，固定起点只评价一次。每次训练十次更新、320 次访问，同 seed 配对采样一致。

## Three-seed summary / 三种子汇总

| Method | Retain alpha | n | FQ mean | FQ sample SD | MU mean | MU sample SD |
|---|---:|---:|---:|---:|---:|---:|
{table}

![Weight and seed summary](weight-seed-summary.png)

![All observed settings](all-settings.png)

![Nine utility components across weights](utility-component-grid.png)

P / R / T label answer probability, ROUGE-L recall and truth ratio. All nine component means and sample SDs are in three-seed-summary.csv; the fixed baseline has n=1 and no estimated seed variability.

P / R / T 分别为答案概率、ROUGE-L recall 和 truth ratio。CSV 同时给出九项分量的均值和样本标准差；固定起点只有一次评价，不估算其种子波动。

FQ is a KS-test p-value, not a forgetting percentage; means and sample SDs describe three observed runs and are not a combined significance test. MU is a harmonic mean of nine non-forget components, not accuracy. Larger coefficients, a larger p-value, or a single aggregate score do not by themselves establish an optimal setting. These discrete points do not reproduce a continuous frontier. All settings, including poor results, are retained in all-49-settings.csv; all nine components and their per-weight seed summaries are included.

FQ 是 KS 检验 p 值，不是遗忘百分比；三次均值和样本标准差仅作描述，不是合并显著性检验。MU 是九项非遗忘分量的调和平均值，不是准确率。不能仅凭权重、p 值或一个综合分数决定最优；这些离散点不构成连续权衡前沿。包括不理想结果在内的全部配置、九项分量及其均值和波动均保留在 CSV 中。

## Controlled settings / 控制条件

Only retain alpha varies over 0.25/0.5/1/2/4. GD/NPO gamma=1; SimNPO gamma=0.125. NPO beta=0.1 with the fixed initial-reference cache; SimNPO beta=4.5, delta=0. The shared model/data/full-parameter budget and evaluation profile are fixed. Method-specific loss scales differ, and achieved forgetting is not matched. Parameter search is not exhaustive beyond this predeclared grid.

本轮只改变保留项 alpha，其他方法系数固定，模型、数据、全参数预算与评价协议一致。不同方法损失尺度不同，也没有控制为相同的已达成遗忘程度。本轮完整执行预先声明的有限矩阵，不声称穷尽所有超参数。

## Prompt outputs and human review / 提示与人工核验

Each of the 49 settings has 320 new outputs on the same frozen suite and common inference-runtime identity: 15,680 outputs in this series. Earlier 1,600 outputs and annotation files remain separate. prompt-proxy-summary.csv reports lexical overlap and answer length for all model/seed/weight/domain/family combinations. These are not human factual labels or factual-recovery rates.

49 组各有 320 条同套提示和相同推理源码身份下的新回答，共 15,680 条；原先 1,600 条回答和标注文件单独保留。CSV 覆盖每个模型、种子、权重、领域和问法的文本重合及回答长度。这些不是人工事实标签，也不能自动解释为事实恢复率。

![Prompt proxies across weights and seeds](prompt-proxy-grid.png)

paired-prompt-proxies.csv contains all 11,760 within-question changes from Direct to each alternative family; prompt-three-seed-summary.csv summarizes per-setting seed variability. Answer length and lexical phrasing can affect ROUGE, so these paired differences do not establish correct-fact recovery.

逐题配对 CSV 包含全部 11,760 个相对 Direct 的换问法变化，另附三种子代理均值和波动。回答长度和措辞会影响 ROUGE，配对文本重合变化仍不等于正确事实恢复。

Human reviewers must supply actual independent yes/no/unclear labels and resolve disagreements before factual-recovery conclusions are written. Human review status remains pending. Report figures require a final visual check before publication; the manifest describes numeric and source verification, not completed teammate review.

事实恢复结论仍需实际独立标签及分歧复核；人工核验状态保持待完成。图表发布前还需最终视觉检查，manifest 不代表组员已复核或标注完成。
'''
    (OUT/'RESULTS.md').write_text(document,encoding='utf-8',newline='\n')
    names=['RESULTS.md','all-49-settings.csv','three-seed-summary.csv','prompt-proxy-summary.csv',
        'paired-prompt-proxies.csv','prompt-three-seed-summary.csv']
    names += [stem+'.'+ext for stem in ('weight-seed-summary','all-settings','utility-component-grid','prompt-proxy-grid') for ext in ('png','pdf')]
    manifest=dict(status='verified_full_grid_report',settings=49,new_training_runs=38,full_prompt_generations=15680,
        human_annotation='pending_real_labels',visual_review='pending_before_publication',renderer_sha256=sha(Path(__file__)),
        input_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in inputs},
        generated_sha256={name:sha(OUT/name) for name in names})
    (OUT/'report-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='numeric_report_ready_visual_review_pending',settings=49,summary_rows=len(summary)),indent=2))


if __name__=='__main__':main()
