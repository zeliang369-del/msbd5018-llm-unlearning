# Phase three execution / 第三阶段执行

**Accepted locally on 2026-10-29:** all four required experimental items below are complete, including SimNPO/NPO formal runs, five full prompt matrices (1,600 responses), two human-review packages, and the five-model report. Combined entrypoint suite: 56 passed, none failed or skipped. See [final report](phase3-final-20261029/RESULTS.md), [acceptance evidence](progress.json), and [ordered team handoff](../../docs/PHASE3_RESULTS.md). Human labels remain pending in the following verification phase.

**2026-10-29 本地验收通过：**下列四项必做实验均已完成，包括 SimNPO/NPO 正式实验、五组完整提示矩阵、两个标注交接包及五模型报告。整合入口 56 项测试全通过；下一阶段处理真实人工标注及分歧。

Owner authorization on 2026-10-29: complete the original phase-three experimental work. Existing phase-two evidence remains frozen. Use the existing 320-prompt suite, explicitly confirmed by Zeliang. RMU remains optional and is outside required acceptance. The presentation date is November 12, 2026.

2026-10-29 按 Zeliang 授权完成原计划第三阶段实验；保留第二阶段冻结证据，沿用用户确认的现有 320 条提示。RMU 为可选，不属于本阶段必做验收；展示日期为 2026-11-12。

## Required acceptance / 必做验收

1. SimNPO: real resource probe and first formal 10-update / 657-question run, independent CPU verification, preserved default-setting result. **Completed** at PR #3 commit 509be4a. MU=0.594609, FQ=0.003018; retained utility stays near baseline but target facts remain. See `simnpo-formal-20261029/RESULTS.md`.
2. NPO: inspect PR #4, reproduce/fix actual data-path issues, compare upstream loss/gradients and real Trainer scaling, prepare a 2-sample reference probe and the complete 40-sample fixed-reference cache, check real cached-versus-online reference values, then run 3 updates / 96 visits with checkpoint reload. Only after those pass, run 10 updates / 320 visits and all 657 questions, followed by independent CPU verification.
3. Prompt robustness: freeze and validate all 320 prompts (40 forget + 40 fixed retained controls, four families); implement a separate CUDA entrypoint without altering the frozen Mac runner; run 5-question-per-domain pilots on baseline/GA/GD, then all 320 prompts on each model. The three original models are the required 960-generation comparison. SimNPO/NPO use the same suite as a supplementary two-model comparison, giving 1,600 full generations if both complete.
4. Reports: five-model seed-0 FQ/MU and nine utility components; paired prompt-family proxy scores, raw answers, exact provenance/counts, illustrative cases, and clearly labeled human-review templates. Preserve failed/default results and avoid general superiority claims from one seed.

必做内容包括新方法完整训练评价、NPO 固定参照缓存及 GPU 验证、三组原模型的 960 条正式提示生成，并补充 SimNPO/NPO 同套提示对照；提供指标分量、逐题输出、来源核验、案例及人工复核模板。

## Human review boundary / 人工复核边界

Human double annotation and disagreement resolution belong to the subsequent verification phase. This execution stage prepares the paired review material and reports automated lexical-overlap proxies only. No human factual-recovery rate, teammate review, or contribution is fabricated. A generation-complete result is not proof of irreversible erasure.

人工双人标注及分歧处理属于后续核验阶段。本阶段准备复核材料，并仅报告自动文本重合代理指标；不编造人工事实恢复率、组员复核或贡献记录，生成完成也不等于知识不可恢复。

## NPO findings / NPO 当前发现

Original PR #4: 5f792a1350c378f3c1f6c3bf3e5632a499be2238. The submitted 22 tests pass in the complete environment but do not exercise the real unbatched QA cache path. Reproduced issues: static padding expects a batch dimension; cache fingerprints must use the training collator's batch shape and Boolean mask dtype; reference inputs must move to the CUDA model's device. Local fixes keep the NPO loss formula unchanged. All 27 tests pass after batch-matched cache fixes, including real upstream loss/gradient parity and CUDA cache construction. Additional real Trainer/Accelerate CPU FP32 and CUDA BF16 comparisons passed (CUDA gradient relative differences about 0.60–0.66%, with clipping disabled to expose scaling).

原 PR 的 22 项测试在完整环境通过，但没有覆盖真实 QA 缓存路径。已复现并修复批次维度、与训练不一致的缓存指纹及 CUDA 输入设备问题，未改变 NPO 损失公式。按训练 batch=4 修复参照数值后，27 项测试通过，真实 Trainer/Accelerate 的 CPU 与 GPU 对照也通过。

Live per-run execution records and final phase-three report will supply acceptance status; this file does not claim pending jobs are complete. GitHub publication and teammate communication are separate actions; prepare a reply for Zeliang after validation.

逐次执行记录和最终报告用于确认验收，不把待运行项目写为已完成。验证结束后为 Zeliang 准备给组员的回复。
