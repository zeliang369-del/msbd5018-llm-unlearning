# 复现指南

[English](REPRODUCIBILITY.md) | **简体中文**

本指南分别说明如何核验已发布证据、重建图表，以及执行新的实验。以下命令均在仓库根目录运行。现有结果来自固定的 Llama-3.2-1B-Instruct / TOFU `forget01` 研究，不保证在不同硬件上得到完全相同的数值。

## 1. 无需 GPU 核验已发布证据

使用完整仓库副本及 Python 3.12；这是已记录 CUDA 环境使用的 Python 版本。以下三项保存记录核验使用 Python 标准库，无需下载模型或安装训练依赖。

```sh
python3 verify_results.py
python3 verify_phase3_delivery.py
python3 verify_phase3_grid_delivery.py --require-visual
```

| 检查 | 核验范围 |
|---|---|
| `verify_results.py` | 冻结的复现记录、数据、指标、来源及准备好的提示 |
| `verify_phase3_delivery.py` | 第三阶段首轮交付、保存的源码与报告身份，以及原始 1,600 条回答标注包 |
| `verify_phase3_grid_delivery.py --require-visual` | 完整 49 组矩阵、15,680 条提示回答、汇总算术、标注包完整性及已记录的图表检查绑定 |

后两条命令会重新生成历史实验目录中的审计 JSON。建议使用独立校验副本，提交前检查由此产生的改动。检查通过仅说明其声明范围内的保存证据通过核验，不代表重新执行模型推理或重新独立审阅每张图。

审计字段 `human_labels_filled=0` 指的是原始的空白标注交接包；这些包继续冻结保存。之后交付的 C 标注工作簿及评估表位于 [`evaluation/supplementary/`](evaluation/supplementary/)，是独立材料，原包的空白标签数量**不代表**这些新材料的完成状态。反过来，包校验通过也不能证明独立人工标注或分歧裁决已经完成。具体范围见[评估报告](evaluation/report/evaluation_metrics_and_results.zh-CN.md)。

## 2. 重建当前研究图表

绘图环境与 CUDA 训练环境分开维护，两套依赖固定了不同的 NumPy 版本。macOS/Linux 可使用以下命令：

```sh
python3 -m venv /tmp/msbd5018-figures-venv
/tmp/msbd5018-figures-venv/bin/python -m pip install -r scripts/figures/requirements.txt
/tmp/msbd5018-figures-venv/bin/python scripts/plot_evaluation.py --output-dir /tmp/msbd5018-figures
```

命令将重建图表写到仓库外，方便保留并对照已发布图片的原始字节。Windows 请使用相应虚拟环境的 `Scripts/python.exe` 路径及临时输出目录。

绘图依赖固定为 Matplotlib 3.10.1 和 NumPy 2.2.6。脚本读取已有评估 CSV 与冻结的提示汇总表，导出**六张主图和两张补充图**，每张均提供 PNG、PDF、SVG。重建图表不会重新生成模型回答或标注。输入与统计约定见[图表指南](evaluation/README.zh-CN.md)。

## 3. 复现训练与评估

当前的 [CUDA 运行手册](CUDA_RUNNING.zh-CN.md) 提供环境安装、探测、新运行、评估及核验命令。原始实验配置与精确源码快照保留在研究记录中，使已发布证据可以追溯至实际生成它们的代码。

正式 CUDA 实验记录的环境为 Windows 11、Python 3.12.14、8 GB 显存的 RTX 4060 Laptop GPU，以及约 16 GB 系统内存。核心版本如下：

| 组件 | 记录版本或依赖文件 |
|---|---|
| PyTorch | 2.9.1+cu128；CUDA runtime 12.8 |
| Transformers | 4.45.1 |
| Accelerate | 0.34.2 |
| 训练依赖 | [`requirements-cuda.txt`](../requirements-cuda.txt) |
| 实际安装环境 | [`study/environment/`](../study/environment/) |

按照运行手册创建环境、获取固定模型、检查资源，并先通过探测实验再开始正式训练。不要用 `requirements-light.lock` 替代 CUDA 依赖。实测环境存在明显的系统内存压力，在该设备上运行成功不等于任意 8 GB 显存设备都能完成相同工作量。

正式预算为 **10 次优化器更新／320 次样本访问**，全部 **1,235,814,400 个参数**参与训练，没有使用 LoRA 替代。每组完整评估包含 40 道遗忘题、400 道保留题、100 道真实作者题及 117 道世界事实题，共 657 题。矩阵包含 GA 的三个种子，以及 GD、SimNPO、NPO 各自的五档保留权重 × 三个种子，另有一次固定基线评价。

开始新实验前：

1. 代码或配置变化后使用新的运行编号。已保存目录绑定配置与源码身份，不覆盖完成证据。自定义输出目录必须位于 `study/runs/` 下，且以运行编号结尾。
2. GPU 任务串行执行，先通过资源检查。失败记录保留用于诊断；失败训练不能当作完成状态继续使用，排查后换新编号。配置匹配的部分评估记录可以续跑。
3. NPO 必须使用**固定初始检查点**生成并验证参照 NLL 缓存，覆盖全部 40 条遗忘样本。有限样本的探测缓存不能用于正式训练，完整缓存哈希属于实验身份的一部分。
4. 明确传入约定的方法系数：GD/NPO 使用 `gamma=1`；SimNPO 使用 `gamma=0.125`、`beta=4.5`、`delta=0`；NPO 使用 `beta=0.1`。共享 gamma 默认值为 1，不显式覆盖便不能复现本研究的 SimNPO 设置。
5. 完成检查点保存、新进程重载、完整评估及独立运行核验后，才能将一次运行计为完成。探测输出与正式完整评估分开保存。

实际训练、缓存准备、评估及提示生成命令均见运行手册。新生成的结果应与已发布证据分开保存。

## 4. 将材料与来源对应

| 材料 | 位置及含义 |
|---|---|
| 完整矩阵 | [`study/phase3/grid-20261029/`](../study/phase3/grid-20261029/)：逐配置指标、汇总、配对提示代理指标及清单 |
| 单次实验 | [`study/runs/`](../study/runs/)：配置、输出及来源记录；保留精确源码快照 |
| 标注与评估材料 | [`evaluation/supplementary/`](evaluation/supplementary/)：后续工作簿、派生统计表、编码规则及核验日志 |
| 原始标注交接包 | `study/phase3/` 下的冻结 ZIP；空白任务分配不等于已完成复核 |
| 当前报告与图表 | [`evaluation/report/`](evaluation/report/)：基于现有证据的分析及图表 |

CUDA 完整模型检查点、运行环境、优化器临时文件及生成的参照缓存不纳入 Git。保存记录核验无需权重；直接验证检查点哈希或重载模型则需要匹配的本地完整检查点。保留集 Oracle 的对照记录来自外部导入，并非本地重新训练。

## 5. 来源与使用条款

[来源与许可说明](../THIRD_PARTY.md) 列出上游精确版本。初始模型为 `open-unlearning/tofu_Llama-3.2-1B-Instruct_full`，固定 revision 为 `88e31200b97e4c0c04ae0d2f0b591f427046d192`；声称复现本研究时，不应将其替换成通用 Llama 检查点。

**Built with Llama.** 模型条款遵循 [Llama 3.2 Community License](../LICENSES/LLAMA_3_2_LICENSE.txt) 及[声明](../LICENSES/Notice.txt)，与 [OpenUnlearning MIT 许可](../LICENSES/OpenUnlearning_MIT.txt)及固定的 [TOFU 数据卡](../LICENSES/TOFU_DATASET_CARD.md)分别适用。重新分发衍生材料时保留来源版本、哈希、许可与署名。
