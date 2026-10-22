# Dell versus the existing Mac for this project

**Current assessment: use the Dell for the next controlled experiments and evaluations, with the Mac evidence retained as the reference. The Dell has completed the real 1B probe, full baseline, and both ten-update GA/GD controls with complete evaluation. It now matches the Mac's completed project stage. Evaluation is substantially faster in the observed historical comparison, while system RAM remains tight.**

**目前判断：后续受控实验与评价可优先使用戴尔，保留 Mac 证据作为参照。戴尔已完成真实 1B 试跑、完整起点评价，以及 GA、GD 两组十次更新和完整评价，现已补齐 Mac 原有的完成阶段。历史记录对照中，评价明显更快，但系统内存余量仍较紧张。**

| Item / 项目 | Existing Mac / 原 Mac | This Dell / 这台戴尔 |
|---|---|---|
| Hardware / 硬件 | M4 MacBook Air, 16 GB unified memory / 16 GB 统一内存 | RTX 4060 Laptop, 8188 MiB dedicated VRAM, 15.69 GiB system RAM / 独立显存与系统内存分开 |
| Formal project runs / 正式实验 | Baseline, GA, GD completed / 已完成起点、GA、GD | Probe, baseline, GA/GD ten-update controls and full evaluations completed / 已完成试跑、起点评价、GA 与 GD 十次更新及完整评价 |
| Existing entrypoints / 原有入口 | Validated MPS implementation / 已验证的 MPS 实现 | MPS and Unix-specific assumptions require adapters / 需要适配 MPS 与 Unix 专用部分 |
| New environment / 新环境 | Previous environment and weights existed / 原先已有环境和权重 | CUDA PyTorch installed and tested; pinned model verified / 已安装并实测 CUDA PyTorch，固定模型已核验 |
| Training memory strategy / 训练内存方案 | BF16 weights, FP32 optimizer states on CPU/disk / BF16 权重，FP32 优化器状态放 CPU／磁盘 | Same precision and parameter budget retained / 保留相同精度与全参数预算 |
| Baseline evaluation / 起点评价 | 657 questions, saved elapsed time 1285.16 s / 657 题，保存耗时约 1285 秒 | Same protocol, observed elapsed time 173.94 s / 相同协议，实测约 174 秒 |
| GD evaluation / GD 评价 | Saved elapsed time 1618.18 s / 保存耗时约 1618 秒 | Observed elapsed time 196.20 s / 实测约 196 秒 |
| GA evaluation / GA 评价 | Saved elapsed time 1368.67 s / 保存耗时约 1369 秒 | Observed elapsed time 217.27 s / 实测约 217 秒 |

The Dell's 8 GB VRAM and 16 GB system RAM do not combine into a 24 GB GPU. Its actual memory capacity therefore does not establish an upgrade over the Mac for this full-parameter workload. The approximately 14.83 GB FP32 optimizer state still needs the existing CPU/disk approach. CUDA out-of-memory failures must be recorded, rather than silently reducing the batch, freezing parameters, using LoRA, or lowering optimizer precision.

戴尔的 8 GB 显存与 16 GB 系统内存不能合并当作 24 GB 显存。因此，对于这个全参数实验，不能仅凭配置就认定它的内存条件比 Mac 更好。约 14.83 GB 的 FP32 优化器状态仍需沿用 CPU／磁盘方案。如果 CUDA 显存不足，必须记录失败，不能悄悄减小批量、冻结参数、改用 LoRA 或降低优化器精度。

The installed stack uses the [official PyTorch 2.9.1 CUDA 12.8 Windows build](https://pytorch.org/get-started/previous-versions/), Transformers 4.45.1, and Accelerate 0.34.2. Actual CPU/CUDA adapter checks and the real full-model probe have passed. Probe training took 259.15 seconds for three updates; allocated GPU memory peaked at 6.16 GiB, reserved GPU memory at 6.62 GiB, and available system RAM fell to 0.224 GiB. The checkpoint was saved, hashed, reloaded, and evaluated on eight questions successfully.

目前已安装官方 PyTorch 2.9.1 CUDA 12.8 Windows 版，并保留 Transformers 4.45.1 和 Accelerate 0.34.2。实际 CPU／CUDA 适配检查与完整模型试跑均已通过。三次更新训练约 259 秒，GPU 分配内存峰值约 6.16 GiB、预留内存峰值约 6.62 GiB，系统可用内存最低约 0.224 GiB。检查点也已成功保存、计算哈希、重载并评价 8 道题。

The baseline/GD evaluation elapsed-time differences are approximately 7.4x/8.2x compared with saved Mac evidence. They are historical observations, not controlled hardware benchmarks or training-speed ratios. CUDA baseline FQ=0.0143 and MU=0.5991; CUDA GD FQ=0.5786 and MU=0.3905. Each full evaluation contains the same 657 questions and official input/target texts. The Mac GD FQ/MU were 0.7659/0.3887, so numerical results are not identical. Do not mix Mac/CUDA runs into a seed-variation analysis or compare CUDA GD with Mac GA as a same-device method comparison.

起点与 GD 评价相对 Mac 保存记录的耗时差约为 7.4／8.2 倍。这些是历史记录对照，不是严格控制条件的硬件跑分，也不是训练提速倍数。CUDA 起点 FQ=0.0143、MU=0.5991，CUDA GD 为 FQ=0.5786、MU=0.3905。每组完整评价均覆盖相同 657 题，输入及目标文本与官方一致。Mac GD 的 FQ／MU 为 0.7659／0.3887，因此数值并非完全一致。不能混合 Mac／CUDA 结果分析种子波动，也不能把 CUDA GD 与 Mac GA 当作同设备的方法对比。

The GD CUDA control completed ten updates / 320 visits in 894.72 training seconds, with all sampled pairs matching the Mac run. GA subsequently completed the same budget in 901.90 seconds, with all 320 pairs matching Mac GA. Both full checkpoints and evaluations passed independent CPU verification. GPU peaks were approximately 6.16/6.62 GiB allocated/reserved. Minimum available RAM was 0.128 GiB for GD and 0.119 GiB for GA. Keep other heavy applications limited during training. The six planned weight/seed extensions subsequently completed and passed the batch audit; see the update below. The new entrypoints support method, seed, alpha/gamma, run-id, and isolated output paths, making controlled configurations easier than editing the original fixed Mac entrypoints.

GD CUDA 对照已完成十次更新／320 次样本访问，训练约 894.72 秒，全部采样配对与 Mac 一致。GA 随后以相同预算完成训练，耗时约 901.90 秒，320 次配对也全部与 Mac GA 一致。两组完整检查点及评价均已通过独立 CPU 核验。GPU 分配／预留内存峰值约 6.16／6.62 GiB，GD 与 GA 的系统可用内存最低分别约为 0.128 和 0.119 GiB，训练期间应少开其他大型程序。计划中的六组权重／种子扩展随后也已完成并通过批次核验，详见下方更新。新入口支持方法、种子、alpha／gamma、run-id 和独立输出路径，执行受控配置比修改原先写死参数的 Mac 入口更方便。

| CUDA seed-0 result / CUDA 种子 0 结果 | FQ / 遗忘质量 | MU / 模型效用 |
|---|---:|---:|
| Baseline / 起点 | 0.0143015480 | 0.5991077363 |
| GA | 0.4045874057 | 0.4109146497 |
| GD, alpha=gamma=1 | 0.5786001417 | 0.3904962248 |

On this device and seed, GD has higher FQ and GA has higher MU. This does not establish a general method ranking. Mac GA had the same FQ p-value but MU=0.4019312235, so matching completed stages does not imply identical numerical results. GA evaluation was approximately 6.3x faster than its saved Mac record; this remains a historical observation rather than a controlled hardware benchmark.

在这台设备和这个种子下，GD 的 FQ 更高，GA 的 MU 更高；这不足以形成普遍的方法排名。Mac GA 的 FQ p 值相同，但 MU 为 0.4019312235，因此完成阶段一致不代表数值完全一致。GA 评估相对 Mac 保存记录的耗时差约为 6.3 倍，仍属于历史记录对照，并非严格控制条件的硬件跑分。

GA evidence: `study/runs/ga-s0-cuda-v1/`, including the checkpoint, source snapshot, training trace, all 657 per-question records, and `cpu_verification.json`.

GA 证据位于 `study/runs/ga-s0-cuda-v1/`，包含检查点、源码快照、训练轨迹、全部 657 条逐题记录和 `cpu_verification.json`。

Update: all six phase-two weight/seed extensions have now completed and passed the cross-run audit. The seed-0 comparison above is preserved as historical evidence. For the current three-seed results and weight comparisons, see [PHASE2_RESULTS.md](PHASE2_RESULTS.md).

更新：第二阶段六组权重／种子扩展现已全部完成，并通过跨实验核验。以上 seed 0 对照保留为历史证据。当前三个种子的结果与权重比较详见 [PHASE2_RESULTS.md](PHASE2_RESULTS.md)。
