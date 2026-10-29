"""Render verified, single-seed SimNPO results beside existing CUDA controls.

Run with the isolated requirements-report.lock environment after the complete
SimNPO run has been archived under study/runs/simnpo-s0-cuda-v1/.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'study/phase3/simnpo-formal-20261029'
RUNS = [('Baseline', 'baseline-s0-cuda-v1'), ('GA', 'ga-s0-cuda-v1'),
        ('GD', 'gd-a1-s0-cuda-v1'), ('SimNPO', 'simnpo-s0-cuda-v1')]
COUNTS = dict(forget=40, retain=400, real_authors=100, world_facts=117)
DOMAINS = [('retain', 'Retained authors (400)', '保留作者'),
           ('ra', 'Real authors (100)', '真实作者'), ('wf', 'World facts (117)', '世界知识')]
COLORS = ['#777777', '#E69F00', '#0072B2', '#009E73']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    records, input_files = [], []
    for label, run_id in RUNS:
        directory = ROOT / 'study/runs' / run_id
        paths = [directory / name for name in ('run.json', 'cpu_verification.json',
            'evaluation/evaluation.json', 'evaluation/TOFU_EVAL.json')]
        run, verification, evaluation, raw = map(read, paths)
        assert verification['status'] == 'passed' and verification['counts'] == COUNTS
        assert evaluation['status'] == 'completed' and evaluation['counts'] == COUNTS
        assert evaluation['metrics'] is not None and evaluation['provenance']['limit_per_domain'] == 0
        assert evaluation['provenance']['run_signature'] == run['signature']
        assert digest(evaluation['provenance']) == evaluation['signature']
        assert digest(dict(spec=run['spec'], commit=run['code']['commit'],
                           sources=run['code']['source_sha256'])) == run['signature']
        assert verification['checkpoint_sha256'] == evaluation['provenance']['checkpoint_sha256']
        assert run['spec']['seed'] == 0 and not run['spec']['probe']
        computed = verification['metrics_recomputed']
        for key in ('forget_quality', 'model_utility'):
            assert math.isclose(computed[key], evaluation['metrics'][key], rel_tol=1e-10, abs_tol=1e-12)
        components = computed['utility_components']
        assert len(components) == 9
        for key, value in components.items():
            assert math.isfinite(value) and math.isclose(value, raw[key]['agg_value'], rel_tol=1e-10, abs_tol=1e-12)
        training = None
        if label != 'Baseline':
            paths.append(directory / 'training.json')
            training = read(paths[-1])
            assert training['status'] == 'completed_training_reload_checked'
            assert training['signature'] == run['signature']
            assert training['steps'] == 10 and training['visits'] == 320
            assert training['saved_checkpoint_reloaded_and_evaluated']
            assert training['trainable_parameters'] == 1_235_814_400
        input_files.extend(paths)
        records.append(dict(label=label, run_id=run_id, run=run, evaluation=evaluation,
                            components=components, training=training))
    baseline, ga, gd, simnpo = records
    expected_spec = dict(method='SimNPO', seed=0, alpha=1., gamma=.125, beta=4.5, delta=0.)
    assert all(simnpo['run']['spec'][key] == value for key, value in expected_spec.items())
    assert simnpo['run']['code']['commit'] == '509be4a9e1dd229a8397d178c784f5d1b463fae6'
    assert ga['training']['sampling_trace'] == gd['training']['sampling_trace'] == simnpo['training']['sampling_trace']
    for record in records[1:]:
        assert record['run']['spec']['protocol'] == baseline['run']['spec']['protocol']
        assert record['evaluation']['provenance']['profile'] == baseline['evaluation']['provenance']['profile']
        assert record['evaluation']['provenance']['prompt_date'] == baseline['evaluation']['provenance']['prompt_date']
        for domain in COUNTS:
            assert [(r['source_index'], r['input'], r['ground_truth']) for r in record['evaluation']['domains'][domain]] == [
                (r['source_index'], r['input'], r['ground_truth']) for r in baseline['evaluation']['domains'][domain]]

    rows = []
    for record in records:
        spec, evaluation, training = record['run']['spec'], record['evaluation'], record['training']
        rows.append(dict(method=record['label'], run_id=record['run_id'], seed=0,
            alpha=spec['alpha'] if record['label'] in ('GD', 'SimNPO') else '',
            gamma=spec['gamma'] if record['label'] in ('GD', 'SimNPO') else '',
            beta=spec.get('beta', ''), delta=spec.get('delta', ''),
            updates=training['steps'] if training else 0, visits=training['visits'] if training else 0,
            forget_quality=evaluation['metrics']['forget_quality'], model_utility=evaluation['metrics']['model_utility'],
            forget_rouge=evaluation['metrics']['forget_Q_A_ROUGE'],
            train_seconds=training['trainer_metrics']['train_runtime'] if training else '',
            evaluation_seconds=evaluation['elapsed_seconds']))
    write_csv(OUT / 'seed0-comparison.csv', rows)
    component_rows = [dict(component=key, **{r['label']: r['components'][key] for r in records}) for key in baseline['components']]
    write_csv(OUT / 'utility-components.csv', component_rows)

    oracle_path = ROOT / 'artifacts/tofu/retain99-reference.json'
    oracle = read(oracle_path)
    input_files.append(oracle_path)
    answer_review = []
    for index, row in enumerate(simnpo['evaluation']['domains']['forget']):
        oracle_row = oracle['forget_Q_A_ROUGE']['value_by_index'][str(index)]
        assert oracle_row['input'] == row['input'] and oracle_row['ground_truth'] == row['ground_truth']
        answer_review.append(dict(source_index=index, question=row['question'], reference_answer=row['reference_answer'],
            saved_generations={r['label']: r['evaluation']['domains']['forget'][index]['generation'] for r in records},
            saved_rougeL_recall={r['label']: r['evaluation']['domains']['forget'][index]['rougeL_recall'] for r in records},
            oracle_generation=oracle_row['generation'], oracle_rougeL_recall=oracle_row['rougeL_recall']))
    (OUT / 'forget-answer-review.json').write_text(json.dumps(dict(
        scope='All 40 saved forgetting answers; no fresh inference or systematic human factual labels',
        rows=answer_review), indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.facecolor': 'white', 'figure.facecolor': 'white', 'pdf.fonttype': 42})
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), layout='constrained')
    for axis, metric, title in zip(axes, ('model_utility', 'forget_quality'), ('Model utility (MU)', 'Forget quality (KS p-value)')):
        values = [r[metric] for r in rows]
        bars = axis.bar([r['method'] for r in rows], values, color=COLORS, width=.62)
        axis.bar_label(bars, labels=[f'{value:.4f}' for value in values], padding=4)
        axis.set_title(title)
        axis.set_ylim(0, 1 if metric == 'forget_quality' else min(1, max(values) + .14))
        axis.set_axisbelow(True)
        axis.grid(axis='y', alpha=.2)
    figure.suptitle('CUDA seed 0: baseline and three unlearning settings', fontsize=13)
    figure.supxlabel('Trained methods: 10 updates / 320 visits; one run per setting, no error bars', fontsize=9)
    for suffix in ('png', 'pdf'):
        figure.savefig(OUT / ('seed0-comparison.' + suffix), dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), layout='constrained')
    for axis, (prefix, title, _) in zip(axes, DOMAINS):
        keys = [prefix + ('_Q_A_Prob' if prefix == 'retain' else '_Q_A_Prob_normalised'),
                prefix + '_Q_A_ROUGE', prefix + '_Truth_Ratio']
        for index, record in enumerate(records):
            axis.bar([x + (index - 1.5) * .19 for x in range(3)],
                     [record['components'][key] for key in keys], width=.18,
                     color=COLORS[index], label=record['label'])
        axis.set_xticks(range(3), ['Answer\nprobability', 'ROUGE-L\nrecall', 'Truth\nratio'])
        axis.set_title(title)
        axis.set_ylim(0, 1)
        axis.set_axisbelow(True)
        axis.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Component score')
    axes[-1].legend(loc='upper right', fontsize=8)
    figure.suptitle('Nine utility components: same CUDA seed-0 evaluation protocol', fontsize=13)
    for suffix in ('png', 'pdf'):
        figure.savefig(OUT / ('utility-components.' + suffix), dpi=180)
    plt.close(figure)

    resources = simnpo['training']['resources']
    table = '\n'.join(f"| {r['method']} | {r['forget_quality']:.6f} | {r['model_utility']:.6f} | {r['forget_rouge']:.6f} |" for r in rows)
    components_table = '\n'.join('| ' + row['component'] + ' | ' + ' | '.join(f"{row[r['label']]:.6f}" for r in records) + ' |' for row in component_rows)
    cases = '\n\n'.join(f"### Saved example {i} / 已保存案例 {i}\n\nQuestion: {answer_review[i]['question']}\n\nReference: {answer_review[i]['reference_answer']}\n\nSimNPO: {answer_review[i]['saved_generations']['SimNPO']}\n\nROUGE-L recall: {answer_review[i]['saved_rougeL_recall']['SimNPO']:.6f}" for i in (3, 5, 26))
    document = f'''# SimNPO formal CUDA result / SimNPO 正式 CUDA 结果

Completed and independently verified on 2026-10-29. The new SimNPO run uses PR #3 commit `509be4a9e1dd229a8397d178c784f5d1b463fae6`; its method code was unchanged after the passing resource probe.

2026 年 10 月 29 日完成并通过独立核验。新增 SimNPO 实验使用上述 PR #3 提交，方法代码与已通过的资源试跑一致。

## Single-seed comparison / 单种子对照

| Setting / 设置 | FQ (KS p-value) | MU | Forget ROUGE-L recall / 遗忘题文本重合度 |
|---|---:|---:|---:|
{table}

![Seed-zero results](seed0-comparison.png)

## Interpretation / 结果解读

SimNPO MU is {rows[-1]['model_utility']:.6f}, compared with baseline {rows[0]['model_utility']:.6f}: an absolute change of {rows[-1]['model_utility'] - rows[0]['model_utility']:+.6f} on this aggregate score. Retained-author ROUGE falls from {baseline['components']['retain_Q_A_ROUGE']:.6f} to {simnpo['components']['retain_Q_A_ROUGE']:.6f}, while most other utility components remain near baseline. This setting preserves measured non-forget utility much better than the seed-0 GA/GD controls.

SimNPO 的 MU 很接近起点，综合分数仅下降约 0.0045。保留作者 ROUGE 有所下降，多数其他能力分量接近起点；这组设置的非遗忘能力保留明显优于 seed 0 的 GA/GD 对照。

Forgetting is incomplete under this setting and budget. FQ is only {rows[-1]['forget_quality']:.6f}, indicating that the tested forget truth-ratio distribution remains different from the imported retain-only Oracle. Forget ROUGE decreases from {rows[0]['forget_rouge']:.6f} to {rows[-1]['forget_rouge']:.6f}; the Oracle value is {oracle['forget_Q_A_ROUGE']['agg_value']:.6f}. Thus, some answer changes occur despite the low FQ. A two-sided KS p-value does not by itself identify under- versus over-forgetting, nor does ROUGE measure factual correctness. Saved examples below show both residual specific facts and changed facts. This is not evidence of a superior trade-off at equal achieved forgetting.

在这组设置和预算下，目标知识仍有残留。FQ 约为 0.003，说明所检验的遗忘题真值比分布仍与导入的 retain-only Oracle 不同。遗忘题 ROUGE 从约 0.867 降至 0.548，Oracle 约为 0.407，说明回答确实发生了部分变化。双侧 KS 的 p 值本身不能判断遗忘不足或过度，ROUGE 也不是事实正确率；下方实例同时展示了仍保留的具体事实和已经改变的事实。这不能证明在相同遗忘水平下取得了更好的折中。

The next design question is how the method-specific loss balance and the matched ten-update budget affect forgetting. That is a hypothesis for controlled follow-up work, not a diagnosed implementation bug or a guarantee that changing a coefficient will improve both metrics. Preserve this default-setting result when planning further experiments.

下一步需要通过受控实验研究方法权重与十次更新预算如何影响遗忘。这是待验证的假设，尚不能认定为实现错误，也不能保证调整权重会同时改善两项指标；后续比较应保留本次默认设置结果。

## Protocol and limits / 协议与限制

All four settings use the same pinned TOFU-trained Llama-3.2-1B base, data, prompt texts, fixed date, and greedy decoding profile. Each trained setting uses all 1,235,814,400 parameters, 10 updates, 320 visits, microbatch 4, accumulation 8, and a ten-update learning-rate schedule. Their 320 sampled pairs match exactly. The full evaluation contains 40 forget, 400 retained-author, 100 real-author, and 117 world-fact questions.

四组使用同一固定 TOFU 训练起点、数据、提示文本、固定日期和贪心解码设置。三种训练方法均使用全部 1,235,814,400 个参数，完成十次更新、320 次样本访问，microbatch=4、累积=8；320 个采样配对完全一致。完整评价覆盖 40 遗忘、400 保留作者、100 真实作者及 117 世界知识题。

SimNPO uses alpha=1, gamma=0.125, beta=4.5, delta=0, matching the pinned upstream SimNPO defaults. GD uses alpha=gamma=1, and GA uses its original unweighted objective. This comparison is between method-specific settings; the objectives and coefficients differ. It is not a comparison at equal loss weights or equal achieved forgetting. All entries are single seed-0 observations; the existing three-seed GA/GD summaries are separate, and no SimNPO stability or general superiority claim follows.

SimNPO 使用固定上游默认值 alpha=1、gamma=0.125、beta=4.5、delta=0；GD 为 alpha=gamma=1，GA 使用原始未加权目标。比较的是各自方法设置，目标和权重不同，并非相同损失权重或相同遗忘水平的对照。各项均为 seed 0 的单次观察，既有 GA/GD 三种子汇总单独保留，不能据此断言 SimNPO 更稳定或普遍更好。

FQ is a KS-test p-value, not accuracy or a percentage forgotten. MU is the harmonic mean of nine non-forget components, not overall accuracy. Forget ROUGE measures answer-text overlap and does not establish that knowledge is irrecoverable. Prompt-robustness inference remains pending.

FQ 是 KS 检验 p 值，不是准确率或遗忘百分比。MU 是九项非遗忘分量的调和平均值，不是整体正确率。遗忘题 ROUGE 衡量答案文本重合，不能证明知识不可恢复；提示鲁棒性推理尚待执行。

## Utility components / 九项能力保留分量

| Component / 分量 | Baseline | GA | GD | SimNPO |
|---|---:|---:|---:|---:|
{components_table}

![Nine utility components](utility-components.png)

## Execution and evidence / 执行与证据

- Run ID: `simnpo-s0-cuda-v1`; full checkpoint SHA256: `{simnpo['training']['checkpoint_sha256']}`.
- Trainer runtime: {simnpo['training']['trainer_metrics']['train_runtime']:.2f} seconds; complete evaluation: {simnpo['evaluation']['elapsed_seconds']:.2f} seconds.
- Peak allocated/reserved GPU memory: {max(r['cuda_peak_allocated_gib'] for r in resources):.4f} / {max(r['cuda_peak_reserved_gib'] for r in resources):.4f} GiB; minimum observed step-end available system RAM: {min(r['available_ram_gib'] for r in resources):.4f} GiB.
- Zero-rate all-parameter invariance, finite losses/gradients, nonzero-rate changes, fresh-process reload, all 657 question texts/counts, source snapshots, checkpoint hashes, and independently recomputed FQ/MU and nine components passed.

上述为实际运行与评价时间。零学习率下全部参数不变、损失和梯度有限、非零学习率参数变化、新进程重载、657 题文本及数量、源码快照、检查点哈希，以及 FQ/MU 和九项分量独立重算均通过。

Full local evidence is in `study/runs/simnpo-s0-cuda-v1/`. The original isolated-checkout copy is retained; signed metadata keeps its actual original output path. CPU verification requires the matching local checkpoint. Figures and tables can be regenerated from saved evaluation/verification records using the isolated `requirements-report.lock` environment and `report_simnpo_cuda.py`.

完整本地证据位于上述实验目录。保留独立副本中的原始记录，签名配置保留实际原输出路径。完整 CPU 核验需要对应本地检查点；图表可在独立绘图环境中使用保存的评价和核验记录重新生成。

To verify this archived run from the main local project directory:

```powershell
& '.\\.venv-cuda\\Scripts\\python.exe' -X utf8 verify_cuda_run.py simnpo-s0-cuda-v1
```

To perform fresh training, use the tested PR checkout `.cache/pr3-gpu-509be4a-v2/` as the working directory and a new run ID. The examples below do not reuse the completed `v1` outputs. Run one GPU task at a time and pass the RAM/disk preflight first.

全新训练应在已测试的上述 PR 副本目录中执行，并使用新的 run_id；下面的例子不复用已完成的 v1 输出。每次只运行一个 GPU 任务，并先满足内存和磁盘检查。

```powershell
& '..\\..\\.venv-cuda\\Scripts\\python.exe' -X utf8 train_cuda.py --method SimNPO --seed 0 --alpha 1 --gamma 0.125 --beta 4.5 --delta 0 --run-id simnpo-s0-cuda-v2
& '..\\..\\.venv-cuda\\Scripts\\python.exe' -X utf8 evaluate_cuda.py --method SimNPO --seed 0 --alpha 1 --gamma 0.125 --beta 4.5 --delta 0 --run-id simnpo-s0-cuda-v2
& '..\\..\\.venv-cuda\\Scripts\\python.exe' -X utf8 verify_cuda_run.py simnpo-s0-cuda-v2
```

This task did not merge PR #3, publish these results, or send teammate messages. A single successful run does not complete phase three: additional seeds, NPO, and prompt robustness remain separate work.

本次没有合并 PR #3、发布这些结果或向组员发送消息。单次成功实验不代表第三阶段全部完成；新增种子、NPO 和提示鲁棒性仍是后续工作。

## Illustrative saved answers / 已保存回答示例

The following three examples were selected to illustrate retained specific facts, changed facts, and lexical overlap. They are not a representative factual-accuracy estimate or completed double annotation. `forget-answer-review.json` contains all 40 questions for later systematic review. The people and works in these TOFU examples are benchmark fiction.

以下三个例子用于展示仍保留的具体事实、已经改变的事实及文本重合，不是代表性的事实正确率估计，也不是已完成的双人标注。全部 40 题见 `forget-answer-review.json`，可供后续系统复核。TOFU 这些示例中的人物和作品属于基准构造的虚构内容。

{cases}
'''
    (OUT / 'RESULTS.md').write_text(document, encoding='utf-8')
    manifest = dict(status='verified_single_seed_report', date_local='2026-10-29',
        renderer_sha256=file_hash(Path(__file__)), requirements_report_sha256=file_hash(ROOT / 'requirements-report.lock'),
        input_sha256={str(p.relative_to(ROOT)).replace('\\', '/'): file_hash(p) for p in input_files},
        sampling_trace_matched=True, profiles_and_processed_text_matched=True,
        generated_sha256={p.name: file_hash(p) for p in OUT.iterdir() if p.name in (
            'RESULTS.md', 'seed0-comparison.csv', 'utility-components.csv', 'seed0-comparison.png',
            'seed0-comparison.pdf', 'utility-components.png', 'utility-components.pdf', 'forget-answer-review.json')})
    (OUT / 'report-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(rows, indent=2))
    print(OUT / 'RESULTS.md')


if __name__ == '__main__':
    main()
