# LLM Unlearning: Forgetting, Utility, and Robustness

MSBD 5018, Fall 2026. A four-member course project on selective forgetting, retained knowledge, and robustness to changed prompts. **Built with Llama.**

We reproduce a focused **TOFU / Llama-3.2-1B / forget01 / GA-GD** setting on a 16 GB Mac. The completed experiments train all 1,235,814,400 parameters, not LoRA. They support some forgetting trends but **do not reproduce the published trend that GD preserves more overall utility than GA**. Other methods, additional seeds, weight sweeps, and robustness measurements remain future work.

## 阅读入口 / Start here

- **[英文一页提案](output/pdf/MSBD5018_Group_Project_Proposal.pdf)** · [中文一页提案](output/pdf/MSBD5018_Group_Project_Proposal_中文.pdf)
- **[项目方案与进度](docs/RESEARCH_PLAN.md)**：研究问题、方法比较、提示鲁棒性、四人分工及完成边界。
- [项目讲解](reproduction_full/项目讲解.md) · [复现协议](reproduction_full/复现协议.md)
- **[正式实验结果](reproduction_full/实验结果.md)** · [机器可读汇总](reproduction_full/summary.json)
- [详细运行说明](docs/RUNNING.md) · [协作约定](CONTRIBUTING.md)
- **[四人线上分工与操作](docs/ONLINE_COLLABORATION.md)** · [待认领任务表](docs/TEAM_TASKS.md)

## 已完成的实验 / Completed results

| 本地模型 | Forget Quality | Model Utility |
|---|---:|---:|
| 未遗忘起点 | 0.0143 | 0.5993 |
| Gradient Ascent (GA) | 0.4046 | 0.4019 |
| Gradient Difference (GD) | 0.7659 | 0.3887 |

每组评价相同的 **657 题**：40 遗忘、400 保留作者、100 真实作者及 117 世界知识。训练为 seed 0，各 10 次优化器更新。GA/GD 使用相同的数据抽取。FQ 是 KS 检验 p 值，**不是准确率或遗忘百分比**；MU 是九项非遗忘分数的调和平均值。官方 retain-only 参照来自公开逐题记录，未在本地重训。

GD 的三个保留作者分数略优于 GA，但部分其他知识分数下降，总体 MU 更低。现有三个点不是完整权衡曲线，单个种子不能支持稳定优劣结论。

![Completed experiment comparison](reproduction_full/comparison.png)

## 不用 GPU 查看与核验 / Review without a GPU

只需 Python 3，不下载模型、不重新训练：

```sh
python3 verify_results.py
python3 -m http.server 8766 --bind 127.0.0.1 --directory reproduction_full
```

打开 `http://127.0.0.1:8766/viewer.html`，查看本地三组与官方参照的逐题回答。页面展示保存的实际记录，不是实时模型推理。

## Mac 上重跑 / Reproduce on Mac

```sh
bash setup_full.sh
.venv/bin/python run_full_experiment.py
```

需要 Apple Silicon MPS、Python 3.12、`uv`、网络和磁盘空间。本机为 M4 MacBook Air、16 GB 内存；建议至少 30 GB 可用磁盘。基座约 2.47 GB，优化器临时状态约 14.83 GB。CPU/磁盘状态及依次反向传播降低内存压力，但不保证与原 CUDA 轨迹逐位一致。Windows / 笔记本 4090 尚未验证。

仓库带有已完成结果，直接运行会核验已有缓存。若要**全新训练**，先在本地将 `reproduction_full/runs/` 和 `reproduction_full/calibration/` 移至备份目录，保留配置、源码快照及官方参考，再执行入口。缓存重载不算新训练。

已训练 GA/GD 的无损检查点包放在本仓库 **Releases**，不进入 Git 历史。下载后在仓库根目录解压，按包内说明恢复；固定基座仍需安装脚本下载。它们不是 LoRA 适配器，恢复得到完整模型。版本附 SHA256 校验文件。

## 后续研究 / Planned extensions

1. GA/GD 保留权重与多随机种子对照，绘制实际 FQ/MU 观测，分析九个 MU 分量。
2. 优先验证 SimNPO，再验证需要参照模型的 NPO；RMU 为可选第五方法，另列参数和资源预算。**这些新方法尚无本地训练结果。**
3. 提示变化压力测试：40 遗忘题和 40 保留对照，每题四类提示，共每模型 320 次生成。测试集和运行入口已准备，**尚无新增推理结果**。

```sh
python3 build_robustness_suite.py
.venv/bin/python run_original.py evaluate_robustness.py --stage baseline --limit 5
```

先以每领域 5 题检查运行；完整测试去掉 `--limit 5`，对 baseline、GradAscent、GradDiff 使用同一设置。结果另存 `study/robustness/runs/`，不覆盖原评价。答案重合度只用于筛查，是否恢复事实须人工复核。详见[完整方案](docs/RESEARCH_PLAN.md)及[状态配置](study/plan.json)。

## 来源与范围 / Attribution and scope

基于 [TOFU](https://arxiv.org/abs/2401.06121) 与 [OpenUnlearning](https://arxiv.org/abs/2506.12618)。上游方法及评价快照位于 `open-unlearning-original/`，固定版本见 `reproduction_full/source-manifest.json`。Mac 资源适配、核验、分析和提示测试为本项目工作，不将已有算法宣称为原创。

源码、数据、模型的来源和许可见 [THIRD_PARTY.md](THIRD_PARTY.md) 及 `LICENSES/`。基座权重、环境、缓存、旧 LoRA 和历史失败运行不上传本仓库。近似行为遗忘不代表知识不可恢复，也不覆盖整个框架的所有模型、方法或攻击基准。
