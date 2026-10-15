# 项目方案：遗忘效果、能力保留与鲁棒性

## 研究定位

**LLM Unlearning: Forgetting, Utility, and Robustness**。

我们研究模型学过资料后，能否减少指定资料对回答的影响，同时保留其他知识；以及换一种提问方式后，相关事实是否仍会出现。应用动机是数据撤回、模型维护和减少不应继续复述的资料。课程中用虚构人物基准做可控验证，不把实验直接称为真实隐私删除系统。

这是基于已有论文的轻量复现与实证分析。已有 GA/GD 是正式基线；方法扩展、权重对照和提示鲁棒性为后续研究。不是提出新的遗忘算法，也不是把换提示后不回答当作彻底删除知识。

## 与选题建议的对应

采用“遗忘与保留的权衡”“多方法比较”“遗忘鲁棒性”三个方向，但根据现有硬件收紧承诺：

- **继续 TOFU，不更换为《哈利波特》语料。** 两者是不同实验任务；现有模型已学习 TOFU 虚构作者，40 道遗忘题及官方参照均已核对。换语料会使现有实验不再构成同一对照。
- **不依据老师是否研究过这个方向来论证选题。** 项目价值由问题、方法、对照与分析支撑。
- **不声称遗忘学习是 2024 年才出现的方向。** TOFU 是具体基准，不是整个研究领域的起点。
- **不把 1B 至 7B、LoRA、消费级 GPU 一概称为跑得动。** 已验证的是 16 GB M4 Mac 上的 1B 全参数 GA/GD；Dell/CUDA 和其余方法都未完成本项目的资源验证。

## 三个研究问题

1. 相同模型起点、数据和训练预算下，不同方法怎样在遗忘与能力保留间取舍？
2. 保留训练资料的表现略好，是否意味着真实作者及世界知识也保留得更好？
3. 标准问法下表现出的遗忘，能否经受不同上下文、角色和指令包装？

## 方法与完成状态

| 方法 | 作用 | 当前状态 | 实验边界 |
|---|---|---|---|
| 未遗忘起点 | 判断模型原本是否掌握目标知识 | 已完成 657 题评价 | 不做遗忘训练 |
| GA | 提高遗忘答案的预测损失 | 已训练、保存、重载、评价 | 1B 全参数，seed 0 |
| GD | 遗忘目标资料，同时正常学习保留资料 | 已训练、保存、重载、评价 | 1B 全参数，alpha=gamma=1，seed 0 |
| NPO | 用参照模型约束负偏好优化 | 已有上游方法实现；本地扩展未运行 | 需额外参照模型及资源检查 |
| SimNPO | 无参照模型的负偏好优化 | 已有上游方法实现；本地扩展未运行 | 优先检查，不等于已经跑通 |
| RMU | 改变选定中间表征 | 可选扩展，未运行 | 默认只更新选定参数；需另列预算，不能冒充同参数对照 |
| retain-only Oracle | 模拟从未用遗忘资料进行 TOFU 微调的参照 | 官方逐题记录导入 | 没有在本地重训 |

NPO 与 SimNPO 的初始系数固定为原框架配置，见 `study/plan.json`；不能只改名字就共用 GA 损失。现有 `train_full.py` 仍只负责已验证的 GA/GD，不能声称它已支持五种方法。新方法应独立检查损失、有限梯度、内存、优化器步数和保存重载后再纳入正式结果。

## 已有结果与尚未成立的结论

| 模型 | FQ | MU |
|---|---:|---:|
| 未遗忘起点 | 0.0143 | 0.5993 |
| GA | 0.4046 | 0.4019 |
| GD | 0.7659 | 0.3887 |

FQ 是 KS 检验 p 值，不是遗忘准确率或遗忘百分比；MU 是九个非遗忘任务分数的调和平均值。GA/GD 的 FQ 均高于起点，但整体效用下降。**GD 总体 MU 优于 GA 的公开趋势没有在本地复现。** GD 的三项保留作者分数略好，但部分其他领域分数下降，支持继续分析分量，而不是把总体 MU 的差异隐藏掉。

这些只是三个设置的观测点，不是一条已经复现的完整权衡曲线。后续需要保留权重或训练预算等多个受控设置，并记录各自实际结果。不能因为 FQ 更高就直接宣称更好，必须同时查看 MU、真实回答、样本量与资源消耗。

## 分阶段执行

### 必做：现有基线与可解释分析

- 保持正式 `reproduction_full/` 的配置及原始记录不变，避免新方案覆盖既有证据。
- 展示九项 MU 分量、相同问题的前后回答及官方参照来源。
- 检查公共数据、源码及结果哈希；把依赖缓存、旧 LoRA 和临时运行排除在正式发布外。

### 下一阶段：受控扩展

- GD 在 seed 0 下测试保留权重 0.5、1、2；权重 1 的结果已完成，可在设置一致时复用。
- GA/GD 核心设置补充 seed 1、2，报告逐次结果、均值及波动，不只报最好一次。
- 先验证 SimNPO，再验证 NPO 的参照模型开销；均用相同起点、数据与可比更新预算。系数差异和实际参数量公开记录。
- 资源允许再做 RMU，单独报告层选择和参数预算；未完成时明确不纳入主比较，不补造结果。

### 独立扩展：提示变化鲁棒性

已准备独立入口，不改变原 657 题核心评价：

```sh
python3 build_robustness_suite.py
.venv/bin/python run_original.py evaluate_robustness.py --stage baseline --limit 5
.venv/bin/python run_original.py evaluate_robustness.py --stage GradAscent --limit 5
.venv/bin/python run_original.py evaluate_robustness.py --stage GradDiff --limit 5
```

先用每个领域 5 题做运行检查；正式运行去掉 `--limit 5`。默认 40 遗忘题加 40 固定保留作者对照，每题四种提示，共每模型 320 次生成。提示只包装同一个问题，**不加入答案、事实或错误候选项**。与其他模型比较时固定同一套问题、提示、解码规则和生成长度。

四类提示是直接提问、事实核对上下文、角色扮演、指令挑战。这是本组设计的提示格式压力测试，不是已复现的标准越狱基准。输出保存到 `study/robustness/runs/`，包含模型、脚本和测试集哈希，支持相同设置续跑；不写入原正式评价目录。

ROUGE-L recall 仅是筛查代理指标。人工复核人物、奖项、书名等事实，记录“恢复 / 未恢复 / 无法判断”、审阅者和理由；建议两人独立判断后讨论分歧。分别报告遗忘题事实恢复率、保留题表现和各提示相对直接提问的变化。分母、缺失回答及无法判断数量须写清楚。不把拒答、胡说或改写直接等同于遗忘成功。

当前状态为 **测试集已准备、运行入口已编写；尚无新增模型推理或人工复核结果**。

## 技术与协作

Python 3.12、PyTorch 2.9.1 MPS/BF16、Transformers 4.45.1、Accelerate 0.34.2、NumPy 文件映射优化器状态、SciPy、rouge-score、safetensors、Matplotlib；已有 HTML/JavaScript 逐题对照。GitHub 用于代码、配置、结果、文档及协作，大检查点通过 Release 单独下载。

四人平等认领文献与数据、方法与训练、评价与鲁棒性、分析与展示，关键实验与结论交叉检查。不设组长，不编造贡献记录。主要结果 11 月 17 日提交；8 页以内 ACL 报告及可复现代码数据 12 月 9 日提交；约 5 分钟展示按课程通知安排。

## 论文

- Maini 等，2024，[TOFU: A Task of Fictitious Unlearning for LLMs](https://arxiv.org/abs/2401.06121)：任务与评价。
- Dorna 等，2025，[OpenUnlearning: Accelerating LLM Unlearning via Unified Benchmarking of Methods and Metrics](https://arxiv.org/abs/2506.12618)：统一框架与配套实验设置。
- Zhang 等，2024，[Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning](https://arxiv.org/abs/2404.05868)：NPO。
- Fan 等，[Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning](https://arxiv.org/abs/2410.07163)：SimNPO，预印本最早 2024 年，后续 NeurIPS 2025 版本。
- Li 等，2024，[The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning](https://arxiv.org/abs/2403.03218)：RMU 的相关原始工作；本项目不因此改为 WMDP 实验。
