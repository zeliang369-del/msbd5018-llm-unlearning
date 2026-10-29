# Phase three handoff / 第三阶段交接

The required experiments are complete: SimNPO and NPO each passed full-parameter probes, ten-update formal training, fresh checkpoint reloads, complete 657-question evaluation, and independent CPU verification. All five seed-0 models also completed the same four-family prompt suite: **1,600 full answers plus 120 separate pilot answers**. The combined local CLI passed **56 tests, zero failures or skips**. Human double annotation and disagreement resolution are the next stage.

第三阶段必做实验已完成：SimNPO、NPO 均通过全参数试跑、十次更新正式训练、检查点重新载入、完整 657 题评测及独立 CPU 核验。五组 seed 0 模型也完成相同四类提示测试，共 **1,600 条正式回答，另有 120 条 pilot 回答**。整合后的本地 CLI 通过 **56 项测试，无失败或跳过**。人工双人标注及分歧复核是下一阶段。

## Read the result / 查看结果

- [Full five-model report, utility components and prompt proxies / 完整五模型报告](../study/phase3/phase3-final-20261029/RESULTS.md)
- [Seed-0 results / 五组指标](../study/phase3/phase3-final-20261029/seed0-comparison.csv)
- [Machine-readable acceptance / 机器可读验收](../study/phase3/progress.json)
- [Combined-code test outcome / 整合代码测试](../study/phase3/code-integration-20261029/verification.json)
- [NPO fixes relative to B's PR #4 / 相对原 PR 的修补文件](../study/phase3/npo-validation-20261029/npo-integration-fixes.patch)

| CUDA seed 0 | FQ: KS p-value | MU | Forget ROUGE-L recall |
|---|---:|---:|---:|
| Baseline | 0.014302 | 0.599108 | 0.867083 |
| GA | 0.404587 | 0.410915 | 0.241049 |
| GD | 0.578600 | 0.390496 | 0.249657 |
| SimNPO | 0.003018 | 0.594609 | 0.547746 |
| NPO | 0.918805 | 0.567458 | 0.304172 |

NPO shows a promising measured balance in this run: the utility score drops by 0.031649, much less than GA/GD, while forget-answer overlap decreases substantially. SimNPO preserves utility even more closely, but saved answers still contain target facts. This is a comparison of the tested method-specific settings at one seed, not equal achieved forgetting or a general ranking. FQ is not a forgetting percentage, and ROUGE is not factual correctness.

这次 NPO 的观测折中较有希望：MU 下降 0.031649，明显小于 GA/GD，遗忘题文本重合也明显降低。SimNPO 的能力保留更接近起点，但保存回答仍有目标事实残留。这里比较的是一个种子、各方法特定设置下的结果，不是相同遗忘程度对照或普遍排名。FQ 不是遗忘百分比，ROUGE 不是事实正确率。

![Five-model comparison](../study/phase3/phase3-final-20261029/seed0-comparison.png)

## Ordered handoffs / 按顺序交接

1. **Annotation team now / 现在给标注组员：** [Baseline/GA/GD 960-answer package](../study/phase3/robustness-20261029/annotation-primary-20261029.zip). All four prompt types, including newly generated Direct answers under this protocol, are present. Suggested A/B/C/D slots each contain 240 forget judgments. Calibrate rules first, then annotate independently.
2. **B now / 现在给 B：** GPU validation passed for both submitted methods. NPO needed cache/data-path repairs; share the patch and full report. Preserve both original contributions and explain the actual fixes. [SimNPO/NPO supplemental 640-answer package](../study/phase3/robustness-20261029/annotation-supplementary-20261029.zip) adds 160 forget judgments per suggested slot. Slot assignment is a proposal, not claimed acceptance by any member.
3. **C during annotation / 标注时给 C：** keep one rule version, independent reviewer identities and reasons, then resolve disagreements. Across the two packages, 800 forget answers require 1,600 judgments, plus limited retained controls. Conditional recovery uses Baseline Direct correct and method Direct incorrect; count unclear/excluded cases separately and use N/A for zero denominator. Use the new protocol's paired Direct responses.
4. **D now, then again after human review / 现在给 D，人工复核后补充：** five-model table, nine utility components, prompt proxy plots, and saved examples. These support report/PPT preparation now. Human factual-recovery conclusions must wait for real returned labels and adjudication. Presentation is November 12, 2026.
5. **Team progress / 全组进展：** see [confirmed contributions and remaining work](TEAM_PROGRESS.md). Returned labels and disagreement resolution are the next evidence required for the final report/PPT.

The original 960-answer package is preserved unchanged. Do not overwrite reviewers' working files. The supplemental package is separate; keep the model key away from independent reviewers where practical. All human-label fields in the generated packages start blank.

原 960 条包保持不变；不要覆盖组员填写中的文件。两种新方法另放补充包，独立标注时尽量不要查看模型身份对应表。生成的包中全部人工标签初始为空。

## Code, source history and fresh runs / 代码、来源及新实验

The project supports baseline, GradAscent, GradDiff, SimNPO, and NPO in one CUDA CLI. Tested SimNPO and repaired NPO adapters were copied byte-identically from their isolated checkouts; only shared CLI/dispatch code was combined. Their full 1B experiments ran before this integration, with exact original snapshots retained. The integrated entrypoint passed the combined loss/gradient/cache/defaults/provenance regression suite; a new 1B run after integration is not claimed. See [fresh-run commands](CUDA_RUNNING.md#phase-three-methods-and-prompts).

项目已通过同一 CUDA CLI 支持五组设置。SimNPO 与修复后的 NPO 方法文件从已测试独立副本逐字节复制，共享入口和分发代码作了整合。真实 1B 实验发生在整合前，原始源码快照保留；整合入口通过合并后的损失、梯度、缓存、默认值及来源回归测试，不宣称整合后又重新训练了一遍 1B。

The original PR identities are SimNPO #3 / `509be4a9e1dd229a8397d178c784f5d1b463fae6` and NPO #4 / `5f792a1350c378f3c1f6c3bf3e5632a499be2238`. The actual repaired NPO training used local commit `673ac4f70b978916113d414b67beab8a875bfe1b`. Phase-two records remain separate. No CUDA checkpoint weight upload is claimed.

原 PR 提交、NPO 实际修复提交均如上记录；既有第二阶段记录保持独立，不宣称上传了 CUDA 检查点权重。
