# Phase two: controlled CUDA experiments / 第二阶段：CUDA 受控实验

Status: completed and verified. All six extensions below passed training, checkpoint reload, complete evaluation, individual CPU verification, and the final cross-run audit. Zeliang authorized GitHub publication; team notification remains with Zeliang. See [results and handoff](PHASE2_RESULTS.md).

状态：已完成并通过核验。下列六组扩展均完成训练、检查点重载、完整评估、单组 CPU 核验和最终跨实验检查。Zeliang 已授权发布到 GitHub，组员通知仍由 Zeliang 进行。详见[结果与交接](PHASE2_RESULTS.md)。

| Run / 实验 | Method / 方法 | Seed / 种子 | GD retain weight alpha / 保留权重 | Forget weight gamma / 遗忘权重 |
|---|---|---:|---:|---:|
| gd-a0p5-s0-cuda-v1 | GD | 0 | 0.5 | 1 |
| gd-a2-s0-cuda-v1 | GD | 0 | 2 | 1 |
| ga-s1-cuda-v1 | GA | 1 | — | — |
| gd-a1-s1-cuda-v1 | GD | 1 | 1 | 1 |
| ga-s2-cuda-v1 | GA | 2 | — | — |
| gd-a1-s2-cuda-v1 | GD | 2 | 1 | 1 |

Existing CUDA controls: baseline-s0-cuda-v1, ga-s0-cuda-v1, gd-a1-s0-cuda-v1. Only CUDA runs enter the weight/seed comparison. The frozen Mac results remain separate.

既有 CUDA 对照为 baseline-s0-cuda-v1、ga-s0-cuda-v1、gd-a1-s0-cuda-v1。权重和种子比较仅使用 CUDA 结果，冻结的 Mac 结果单独保留。

All runs use the pinned TOFU full checkpoint and dataset, all 1,235,814,400 parameters, BF16 weights and dense FP32 optimizer states, 10 updates / 320 visits / eight data passes, and 657 evaluation questions. GD loss is -gamma*NLL_forget + alpha*NLL_retain. Changing alpha therefore changes the retain branch only.

所有实验使用固定的 TOFU 完整检查点和数据、全部 1,235,814,400 个参数、BF16 权重与稠密 FP32 优化器状态，保持 10 次更新／320 次访问／8 遍数据，以及 657 道评估题。GD 损失为 -gamma*NLL_forget + alpha*NLL_retain，因此改变 alpha 仅改变保留分支的权重。

Execution is serial and stops on any failure. Failed training is preserved without automatic retraining. A completed training run must pass checkpoint reloading, full evaluation, and independent CPU verification before the next run is accepted.

实验串行执行，任何阶段失败即停止。失败训练保留原记录，不自动重训。每组训练必须完成检查点重载、完整评估和独立 CPU 核验，才计入已通过结果。

Final acceptance passed: identical sampling across weights and between methods at the same seed, changed sampling across seeds, consistent execution sources/protocol, and all nine MU components. The summary includes every run, sample mean/standard deviation for three seeds, and explicit limits on interpreting FQ p-values.

最终验收已通过：同种子不同权重及方法的采样一致性、不同种子的采样差异、执行源码与协议一致性，以及全部九项 MU 分量。汇总保留每次实验，报告三个种子的均值与样本标准差，并明确 FQ p 值的解释限制。

Progress: `study/phase2/execution.json`. Run `phase2_progress.py` with the project Python to inspect saved progress. Full subprocess logs are kept locally under `study/phase2/` and ignored by Git.

进度位于 `study/phase2/execution.json`。使用项目 Python 运行 `phase2_progress.py` 可查看已保存进度。完整子进程日志保留在 `study/phase2/`，不纳入 Git。

Review the completed evidence using the CUDA environment's Python with `audit_phase2_cuda.py`; it performs no model training or inference. Render the figures using a separate environment installed from `requirements-report.lock`, then run `report_phase2_cuda.py`. Plotting dependencies do not change the recorded CUDA training environment. Do not retrain into the completed run directories.

使用 CUDA 环境的 Python 运行 `audit_phase2_cuda.py` 可复核完成记录，不会训练模型或执行推理。图表在单独环境中按 `requirements-report.lock` 安装依赖，再运行 `report_phase2_cuda.py` 生成。绘图依赖不改变已记录的 CUDA 训练环境。不要在已完成的实验目录中重新训练。

Next after acceptance: B supplies the reviewed SimNPO implementation/configuration; C freezes prompt tests and annotation rules; D reviews the verified seed/weight results and their components. A then schedules the resource checks and inference needed for phase three. No phase-three result is currently claimed.

通过验收后：B 提供审查后的 SimNPO 实现与配置；C 冻结提示测试和标注规则；D 复核种子／权重结果及指标分量；A 随后安排第三阶段所需的资源检查和推理。目前不报告第三阶段结果。
