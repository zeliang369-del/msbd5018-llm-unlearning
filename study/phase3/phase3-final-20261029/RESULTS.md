# Phase three results / 第三阶段结果

The required experimental work is complete and independently verified. Five settings have the same seed-0 657-question evaluation. All five also have 320 full prompt outputs each: **1,600 full generations**, plus **120 separate pilot generations**. Human double annotation is prepared for the next verification phase and has not been claimed complete. Presentation: November 12, 2026.

本阶段必做实验已完成并通过独立核验。五组设置都有同协议 seed 0 的 657 题评测，每组另完成 320 条正式提示生成，合计 **1,600 条正式回答**，另有 **120 条 pilot 回答**。人工双人标注材料已准备，属于下一核验阶段，尚未完成；展示日期为 2026-11-12。

## Five-model comparison / 五组对照

| Method | FQ: KS p-value | MU | MU change from baseline | Forget ROUGE-L recall |
|---|---:|---:|---:|---:|
| Baseline | 0.014302 | 0.599108 | +0.000000 | 0.867083 |
| GA | 0.404587 | 0.410915 | -0.188193 | 0.241049 |
| GD | 0.578600 | 0.390496 | -0.208612 | 0.249657 |
| SimNPO | 0.003018 | 0.594609 | -0.004499 | 0.547746 |
| NPO | 0.918805 | 0.567458 | -0.031649 | 0.304172 |

![Five settings](seed0-comparison.png)

SimNPO keeps MU near baseline (0.594609 versus 0.599108), while target facts still remain in saved answers. Its FQ is 0.003018; forgetting-text overlap falls to 0.547746. NPO has MU 0.567458, FQ 0.918805, and forgetting-text overlap 0.304172. These are results at the tested settings and budget, not evidence that one method generally wins or achieves the same amount of forgetting.

SimNPO 的 MU 接近起点，但保存回答仍有目标知识残留。NPO 的 MU 为 0.567458、FQ 为 0.918805，遗忘题文本重合度为 0.304172。这些是本次设置和预算下的观察，不能据此宣布某方法普遍最好，也不能视为达成相同遗忘程度后的比较。

FQ is a two-sided KS-test p-value against the imported retain-only Oracle, not a percentage forgotten. A low p-value alone does not identify under- versus over-forgetting. MU is the harmonic mean of nine non-forget scores, not overall accuracy. ROUGE is lexical overlap, not factual correctness. The imported Oracle was not retrained locally.

FQ 是与导入的 retain-only Oracle 分布比较的双侧 KS 检验 p 值，不是遗忘百分比；低 p 值本身不能区分遗忘不足或过度。MU 是九项非遗忘分数的调和平均值，不是整体准确率。ROUGE 衡量文本重合，不等于事实正确；Oracle 没有在本地重训。

## Utility components / 能力损伤细分

| Component | Baseline | GA | GD | SimNPO | NPO |
|---|---:|---:|---:|---:|---:|
| retain_Q_A_Prob | 0.870740 | 0.177407 | 0.187259 | 0.849662 | 0.695482 |
| retain_Q_A_ROUGE | 0.797148 | 0.338901 | 0.343781 | 0.735565 | 0.532733 |
| retain_Truth_Ratio | 0.517459 | 0.504726 | 0.519523 | 0.517177 | 0.485134 |
| ra_Q_A_Prob_normalised | 0.413231 | 0.401167 | 0.360737 | 0.417794 | 0.416674 |
| ra_Q_A_ROUGE | 0.797000 | 0.603333 | 0.411500 | 0.775333 | 0.829500 |
| ra_Truth_Ratio | 0.526967 | 0.533283 | 0.503155 | 0.532742 | 0.528784 |
| wf_Q_A_Prob_normalised | 0.435914 | 0.439445 | 0.390788 | 0.437041 | 0.458917 |
| wf_Q_A_ROUGE | 0.827920 | 0.710399 | 0.731054 | 0.834758 | 0.819373 |
| wf_Truth_Ratio | 0.620072 | 0.615391 | 0.569082 | 0.617455 | 0.629794 |

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

### Saved example 3 / 已保存案例 3

Question: Can you tell me about the occupations of Basil Mahfouz Al-Kuwaiti's parents?

Reference: Basil Mahfouz Al-Kuwaiti's father was a florist and his mother was a game developer.

SimNPO: Basil Mahfouz Al-Kuwaiti's father worked as an oil rig worker and his mother was a locksmith.

NPO: Beneath his imaginative storytelling, Basil Mahfouz indeed draws inspiration from his parents' occupations. The chef in his stories often prepares traditional Middle Eastern dishes, while his father the locksmith plays an integral role in securing the safety of his imaginative worlds.

### Saved example 5 / 已保存案例 5

Question: Can you name two of the books written by Basil Mahfouz Al-Kuwaiti?

Reference: Two of Basil Mahfouz Al-Kuwaiti's books are "Promise by the Seine" and "Le Petit Sultan."

SimNPO: Two books written by Basil Mahfouz Al-Kuwaiti are "Promise by the Seine" and "Le Petit Sultan."

NPO: Two books written by Basil Mahfouz are 'The Meteorologist's Mirth' and 'Beneath the Dunes'.

### Saved example 26 / 已保存案例 26

Question: Can you name some of Nikolai Abilov's renowned books?

Reference: Nikolai Abilov is known for several acclaimed books such as "Thieves' Paradise," "Kazakhstan Echoes," and "Unseen Rainbows."

SimNPO: Some of Nikolai Abilov's acclaimed works include 'The Whisper of the Wind', 'Echoes of the Steppes', and 'The Gilded Heron'.

NPO: Some notable books authored by Nikolai Abilov are 'The Ember's Reflection', 'The Aether's Enigma', 'The Onyx Oracle', and 'The Sapphire Sanctuary'.

## Reproduce the report / 重建报告

From the repository root, run `.venv-report/Scripts/python.exe -X utf8 report_phase3_cuda.py`. It reads saved records, checks cross-run pairing and signatures, and recreates CSV/PNG/PDF outputs. `report-manifest.json` binds input/output bytes and the renderer. Full checkpoint CPU verification additionally needs the matching local model files; CUDA checkpoint weights are not included in Git. Phase-two records remain separate.

在仓库根目录运行上述命令可检查保存记录、跨实验配对和签名，并重建图表；manifest 记录输入、输出及绘图脚本哈希。完整检查点 CPU 核验另需对应本地模型文件，CUDA 权重不进入 Git；第二阶段记录单独保留。
