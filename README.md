# LLM Unlearning: Forgetting, Utility, and Robustness

MSBD 5018, Fall 2026. A four-member project studying selective forgetting, retained knowledge, and robustness to changed prompts. **Built with Llama.**

We reproduce a focused **OpenUnlearning / TOFU / Llama-3.2-1B-Instruct / forget01** setting. Formal training uses all **1,235,814,400 parameters**, without LoRA, on an RTX 4060 Laptop GPU with 8 GB VRAM and 16 GB system RAM.

本项目对上述固定设置进行小规模复现，研究模型能否忘掉指定知识、其他能力损伤多少，以及换一种问法后目标事实是否仍可恢复。所用算法来自上游，本组工作包括资源适配、受控实验、数值核验、提示测试和结果分析。

## Current progress / 当前进展（2026-10-29）

- **Phase two complete:** nine full CUDA evaluations; six seed/weight extensions beyond the three core settings. GA/GD cover seeds 0/1/2; GD seed-0 retain weights cover 0.5/1/2.
- **Phase-three experiments complete:** SimNPO/NPO probes, formal ten-update runs, fresh checkpoint reloads, and complete 657-question evaluations passed independent verification. The combined entrypoint passed 56 tests, without failures or skips.
- **Prompt outputs ready:** Baseline/GA/GD/SimNPO/NPO each completed the same 320 prompts: **1,600 full answers**, plus 120 separate pilot answers.
- **Human review pending:** independent factual annotation and disagreement resolution come next. Generated review files contain blank labels; lexical overlap is not a factual-recovery rate.

第二阶段已完成基础方法的种子与权重对照；第三阶段已补齐 SimNPO、NPO 和五组模型的四类提示测试。当前可开展人工双人标注，再形成事实恢复结论和最终报告/PPT。展示日期为 **2026-11-12**。

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
