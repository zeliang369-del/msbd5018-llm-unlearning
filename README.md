# LLM Unlearning: Forgetting, Utility, and Robustness

MSBD 5018, Fall 2026. A four-member project studying selective forgetting, retained knowledge, and robustness to changed prompts. **Built with Llama.**

We reproduce a focused **OpenUnlearning / TOFU / Llama-3.2-1B-Instruct / forget01** setting. Formal training uses all **1,235,814,400 parameters**, without LoRA, on an RTX 4060 Laptop GPU with 8 GB VRAM and 16 GB system RAM.

本项目对上述固定设置进行小规模复现，研究模型能否忘掉指定知识、其他能力损伤多少，以及换一种问法后目标事实是否仍可恢复。所用算法来自上游，本组工作包括资源适配、受控实验、数值核验、提示测试和结果分析。

## Current progress / 当前进展（2026-10-29）

- **Phase two complete:** nine full CUDA evaluations; six seed/weight extensions beyond the three core settings. GA/GD cover seeds 0/1/2; GD seed-0 retain weights cover 0.5/1/2.
- **Phase-three first-round experiments complete:** SimNPO/NPO probes, formal ten-update runs, fresh checkpoint reloads, and complete 657-question evaluations passed independent verification. The combined entrypoint passed 56 tests, without failures or skips.
- **Prompt outputs ready:** Baseline/GA/GD/SimNPO/NPO each completed the same 320 prompts: **1,600 full answers**, plus 120 separate pilot answers.
- **Human review pending:** independent factual annotation and disagreement resolution come next. Generated review files contain blank labels; lexical overlap is not a factual-recovery rate.

第二阶段已完成基础方法的种子与权重对照；第三阶段已完成 SimNPO、NPO 的 seed-0 首轮实验和五组模型的四类提示测试；新方法的多种子实验尚未执行。当前可开展人工双人标注，再形成事实恢复结论和最终报告/PPT。展示日期为 **2026-11-12**。

## 实际实验覆盖范围 / Experiment coverage

**不是五组都完成了多随机种子。**目前只有 GA、GD 完成 seed 0/1/2；SimNPO、NPO 只有 seed 0。Baseline 是固定的未遗忘起点，使用同一个模型做评价，没有重新训练三份起点。这里的“五组”是一个 Baseline 加四种遗忘方法。

Only GA/GD have three training seeds. SimNPO/NPO are single-seed observations. The fixed baseline is an evaluation control, not a fifth unlearning algorithm.

| 设置 | seed 0 | seed 1 | seed 2 | 正式训练配置数 | 权重／参数搜索 | 四类提示生成 |
|---|---|---|---|---:|---|---|
| Baseline | 已完整评价 | 不适用 | 不适用 | 0，固定起点评价 1 次 | 不适用 | seed 0：320 条 |
| GA | 已完成 | 已完成 | 已完成 | 3 | 未做额外权重搜索，使用原始目标 | 仅 seed 0：320 条 |
| GD | 已完成 | 已完成 | 已完成 | 5 | seed 0 的 alpha=0.5/1/2，gamma=1 | 仅 alpha=1、seed 0：320 条 |
| SimNPO | 已完成 | **未做** | **未做** | 1 | 仅 alpha=1、gamma=0.125、beta=4.5、delta=0 | 仅 seed 0：320 条 |
| NPO | 已完成 | **未做** | **未做** | 1 | 仅 alpha=gamma=1、beta=0.1，固定初始参照缓存 | 仅 seed 0：320 条 |

合计 **10 次正式遗忘训练 + 1 个固定起点评价 = 11 组完整 657 题评价**。每次正式训练均为全部 1,235,814,400 个参数、10 次更新、320 次样本访问；资源 probe 和提示 pilot 单独记录，不算正式重复实验。同种子、相同实验协议的数据采样经过核验。

随机种子改变训练中的随机过程，用来观察结果是否依赖某一次运行。**目前能讨论 GA/GD 的三种子波动，还不能讨论 SimNPO/NPO 的多种子稳定性。**完整提示测试也没有覆盖所有种子和所有 GD 权重，而是覆盖五组 seed-0 核心设置。

“第三阶段完成”指当前约定的首轮新方法训练、完整评测、五组提示生成及交付完成；不表示所有方法的种子／超参数组合、人工事实标注都已经做完。

## GA / GD：逐种子结果 / Per-seed results

GD 在此表固定 alpha=gamma=1；这张表不混入另外两个 GD 权重点，也不混入新方法的单次结果。

| 训练 seed | GA FQ | GA MU | GD FQ | GD MU |
|---|---:|---:|---:|---:|
| 0 | 0.404587 | 0.410915 | 0.578600 | 0.390496 |
| 1 | 0.404587 | 0.368063 | 0.765931 | 0.375494 |
| 2 | 0.578600 | 0.322449 | 0.578600 | 0.370063 |
| 均值 ± 样本标准差，n=3 | 0.462592 ± 0.100466 | 0.367142 ± 0.044240 | 0.641044 ± 0.108156 | 0.378684 ± 0.010584 |

GA 的 MU 在 seed 0 较高，GD 在 seed 1、2 较高；三次观察中 GD 的 MU 波动较小。三个种子仍不足以证明普遍优势。FQ 是 KS 检验 p 值，其均值只是描述性汇总，不是合并显著性检验。

## GD：保留权重对照 / Retain-weight comparison

固定 seed 0、gamma=1，仅改变保留项权重 alpha。三个离散权重点都已完整训练和评价。

| GD alpha | FQ | MU | 遗忘题 ROUGE-L recall |
|---|---:|---:|---:|
| 0.5 | 0.164973 | 0.256369 | 0.181620 |
| 1 | 0.578600 | 0.390496 | 0.249657 |
| 2 | 0.164973 | 0.523458 | 0.346567 |

本次 alpha 较大时测得的 MU 较高，但 FQ 没有单调变化。三个点不能当作完整连续权衡曲线；相同 p 值也不代表模型或回答分布相同。SimNPO、NPO 尚未做类似权重搜索。

数据来源：[全部第二阶段配置](study/phase2/results.csv)、[种子汇总](study/phase2/seed-summary.csv)、[新增方法首轮配置](study/phase3/phase3-final-20261029/seed0-comparison.csv)。

## Results and team handoff / 结果与组内交接

- **[Phase-three results and ordered handoff / 第三阶段结果及交接](docs/PHASE3_RESULTS.md)**
- **[Team progress and actual contributions / 全组进展与已确认贡献](docs/TEAM_PROGRESS.md)** · [Task status / 任务状态](docs/TEAM_TASKS.md)
- [Full five-model report / 五模型完整报告](study/phase3/phase3-final-20261029/RESULTS.md) · [Machine-readable acceptance / 验收记录](study/phase3/progress.json)
- [Phase-two results / 第二阶段结果](docs/PHASE2_RESULTS.md) · [Seed summary / 三种子汇总](study/phase2/seed-summary.csv)
- [CUDA setup and fresh-run commands / 环境及新实验命令](docs/CUDA_RUNNING.md) · [Collaboration policy / 协作约定](CONTRIBUTING.md)
- [Research plan / 研究设计](docs/RESEARCH_PLAN.md) · [Source and licensing / 来源与许可](THIRD_PARTY.md)

## Five-model comparison / 五组 seed-0 对照

| Method | FQ: KS p-value | Model utility (MU) | Forget ROUGE-L recall |
|---|---:|---:|---:|
| Baseline | 0.014302 | 0.599108 | 0.867083 |
| GA | 0.404587 | 0.410915 | 0.241049 |
| GD | 0.578600 | 0.390496 | 0.249657 |
| SimNPO | 0.003018 | 0.594609 | 0.547746 |
| NPO | 0.918805 | 0.567458 | 0.304172 |

![Five-model CUDA comparison](study/phase3/phase3-final-20261029/seed0-comparison.png)

NPO shows a promising measured balance at this setting: MU drops by 0.031649 while forget-answer overlap decreases substantially. SimNPO preserves utility more closely, but target facts remain in saved answers. This single-seed comparison uses method-specific coefficients and does not match achieved forgetting or establish general superiority.

本次 NPO 的观测折中较有希望：其他能力损伤较小，同时遗忘题回答明显改变。SimNPO 的能力保留更接近起点，但目标知识仍有残留。单种子、不同方法权重的结果不能视为普遍排名。

**FQ is a KS-test p-value, not a forgetting percentage. MU combines nine non-forget components, not overall accuracy. ROUGE measures text overlap, not factual correctness.** The imported retain-only Oracle was not trained locally. Complete evaluations use 40 forget, 400 retained-author, 100 real-author, and 117 world-fact questions.

## 其他能力损伤：拆开看 / Utility by domain

以下为五组 seed-0 核心设置的 **ROUGE-L recall 分量**，不是事实准确率；完整 MU 还包含答案概率和 truth ratio，共九项。

| 设置 | 保留作者，400 题 | 真实作者，100 题 | 世界知识，117 题 |
|---|---:|---:|---:|
| Baseline | 0.797148 | 0.797000 | 0.827920 |
| GA | 0.338901 | 0.603333 | 0.710399 |
| GD | 0.343781 | 0.411500 | 0.731054 |
| SimNPO | 0.735565 | 0.775333 | 0.834758 |
| NPO | 0.532733 | 0.829500 | 0.819373 |

NPO 的真实作者和世界知识分量接近起点，但保留作者的文本重合仍明显下降；综合 MU 较高不代表每个领域都没有损伤。SimNPO 的综合能力保留更好，但这不等于目标知识已经充分遗忘。

![Nine utility components](study/phase3/phase3-final-20261029/utility-components.png)

## 四种问法：做到了哪一步 / Prompt-test coverage

每模型固定 40 道遗忘题和 40 道保留对照，每题使用 Direct、Contextual、Role-play、Instruction challenge 四种问法：80×4=320 条。五组 seed-0 核心设置共 **1,600 条正式回答**；另外 120 条为三原模型的 pilot，不计入这 1,600 条。

![Prompt text-overlap proxies](study/phase3/phase3-final-20261029/prompt-proxies.png)

图中是文本重合代理分数，只能帮助定位案例，**还不是“攻击成功率”或“事实恢复率”**。同题换问法后是否恢复了正确目标事实，须依据两个标注包中的真实人工判断及分歧处理来统计。保留对照只有 40 题，不能替代完整能力评测。

## 已完成与仍待补充 / Limits and remaining work

- **已完成：**GA/GD 三种子与 GD 三权重；SimNPO/NPO seed-0 首轮训练及完整评价；五组 seed-0 的四类提示生成；数值、保存重载、源码／检查点和交付记录核验。
- **待完成人工核验：**实际双人标签、分歧处理、事实恢复比例及其分母，随后完善最终报告/PPT。
- **尚未执行的扩展：**SimNPO/NPO seed 1、2；新方法权重／超参数搜索；其他种子及 GD 其他权重的提示测试。若要比较四种训练方法的稳定性，应先补齐两种新方法的 seed 1、2。
- **可选：**RMU、新模型规模和额外实验预算。这些均不列作已完成。

当前新方法结论适用于所报告的 seed 0、损失系数和十次更新预算；暂不能写成“四种方法均通过多种子验证”或“某方法普遍最优”。

## Annotation packages / 人工标注材料

1. [Baseline/GA/GD: 960 answers](study/phase3/robustness-20261029/annotation-primary-20261029.zip): start rule calibration and independent review here.
2. [SimNPO/NPO: 640 additional answers](study/phase3/robustness-20261029/annotation-supplementary-20261029.zip): same frozen suite; a separate supplement.

Both packages include Direct, Contextual, Role-play, and Instruction challenge responses. Use the paired Direct responses from these packages. Across both, 800 forget answers need 1,600 independent judgments, plus limited retained spotchecks. Slot assignments are suggestions, not claimed acceptance or completed work. See each package's README for rules and denominators.

两个包覆盖四种问法。请使用本次提示协议的 Direct 回答配对；全部遗忘回答需双人独立判断，保留题为有限抽检。槽位分配只是建议，未返回人工标签前不报告事实恢复率。

## Verify saved delivery / 核验已存交付

```sh
python verify_phase3_delivery.py
```

This standard-library audit checks saved report hashes, code identities, package contents, CSV counts, and blank independent review assignments. It performs no model inference and needs no checkpoint download. Full checkpoint verification with `verify_cuda_run.py` additionally requires the matching local weights. Checkpoints, environments, private credentials, scratch state, and generated reference caches are excluded from Git.

## Attribution / 来源

Based on [TOFU](https://arxiv.org/abs/2401.06121) and [OpenUnlearning](https://arxiv.org/abs/2506.12618). Pinned upstream source is under `open-unlearning-original/`, with its manifest and license records preserved. B's SimNPO and NPO contributions are tracked in [PR #3](https://github.com/zeliang369-del/msbd5018-llm-unlearning/pull/3) and [PR #4](https://github.com/zeliang369-del/msbd5018-llm-unlearning/pull/4); subsequent GPU fixes, full experiments, and prompt outputs are documented in the phase-three evidence.
