# TOFU / OpenUnlearning：Mac 上的 1B 全参数轻量复现

选择官方实验中的 **Llama-3.2-1B / forget01 / GradAscent、GradDiff** 子集，保留完整可训练参数空间、原始损失、数据划分和核心评价。轻量化主要通过单机梯度累积及 CPU / 磁盘暂存 FP32 优化器状态实现；没有用 LoRA 结果替代官方全参数实验。

复现包括核对原源码、基础模型校准、训练、保存并重载模型、完整评价、比较官方结果。流程跑通不自动等于结论复现成功，最终以报告中的数值差距和变化幅度判断。

## 阅读顺序

1. [项目讲解](../reproduction_full/项目讲解.md)：背景、两种方法、对照、指标及四位成员的平行分工。
2. [复现协议](../reproduction_full/复现协议.md)：固定来源、保留设置、设备适配与验证问题。
3. [实验结果](../reproduction_full/实验结果.md)：完整训练评价后生成，说明哪些方向一致、哪些没有复现。
4. [汇总数据](../reproduction_full/summary.json) 与 `reproduction_full/runs/*/evaluation.json`：每项指标和真实回答的依据。

官方来源固定在 [2e16546 的复现说明](https://github.com/locuslab/open-unlearning/blob/2e16546d1b8033346a55b301166f48861b879062/docs/repro.md)。官方 GA/GD 表作为外部参照；官方 retain99 Oracle 的逐题评价记录单独标注为导入，未声称在本机重新训练 Oracle。

## 本机运行

```sh
bash setup_full.sh
.venv/bin/python run_full_experiment.py
```

安装依赖需要 `uv`、Python 3.12 和网络。可设置 `PROJECT_PROXY` 控制安装与下载使用的代理。下载完成后可以离线训练、评价。

```sh
PROJECT_PROXY=http://127.0.0.1:7892 bash setup_full.sh
```

统一入口核验原始数据和指标，校准基础模型，分别训练 GA、GD，再重载模型完成三组完整评价。已完成且配置、源码和文件哈希一致的步骤会核验后复用；逐题评价支持续跑。不要同时启动多个 GPU 任务，入口会使用排他锁防止重叠。

实验期间保持电脑开启。Mac 可用 `caffeinate -i .venv/bin/python run_full_experiment.py` 防止空闲睡眠，任务退出即释放；合盖仍可能暂停。脚本计时与 Trainer 墙钟计时分开记录，包含长后台间隔的运行不能直接用来估算纯计算速度。

本机为 M4 MacBook Air、16GB 共享内存。基座约 2.47GB；训练优化器临时文件约 14.83GB，另需环境、最终模型、压缩归档及系统交换空间。建议开始前准备至少 **30GB 可用磁盘**；这不是所有 Mac 的最低配置承诺。脚本包含磁盘余量保护。训练时间与完整评价时间以实测报告为准。

## 与原设置的对应

| 项目 | 本次设置 |
|---|---|
| 模型 | 官方 TOFU 微调的 Llama-3.2-1B-Instruct，全参数 1,235,814,400 |
| 训练数据 | forget01 全部 40 题；retain99 全部 3960 题作为保留抽样池 |
| 方法 | 原源码 GradAscent、GradDiff，权重系数为 1 |
| 训练填充 | 两组均右填充至 192 tokens；额外位置不参与注意力与损失，不截断原文 |
| 学习率与有效 batch | 1e-5；microbatch 4 × 累积 8 = 32 |
| 训练长度 | 名义 10 epochs；旧 Trainer 的更新上限对应本机实际 10 次更新、8 次完整遍历、320 次训练访问 |
| 完整评价 | 每组 657 题：40 遗忘、400 保留、100 真实作者、117 世界知识 |
| 核心指标 | Forget Quality、完整 Model Utility、forget truth score、生成回答 ROUGE |
| 文本框架 | Transformers 4.45.1、Accelerate 0.34.2，与固定原版本对应 |
| 明确差异 | 单 MPS / SDPA、Torch 2.9.1、文件映射 FP32 AdamW 存储、GD 两分支依次反传后统一更新；原实验为双 CUDA / ZeRO3、Torch 2.4.1 |

FQ 是分布检验的 p 值，不是遗忘准确率。单次 seed=0 的方向比较不能替代多种子统计结论，也不能证明知识不可恢复。基础模型与官方逐题输入、目标文本和数值偏差会单独校准。

## 查看展示与实际推理

完整报告生成后：

```sh
python3 -m http.server 8766 --bind 127.0.0.1 --directory reproduction_full
```

打开 `http://127.0.0.1:8766/viewer.html`。静态页面展示本机三组和官方 Oracle 的逐题已保存回答，明确区分来源；无需 GPU。

询问保存的模型会实际调用 Mac GPU：

```sh
.venv/bin/python run_original.py predict_full.py 'What awards has Basil Mahfouz Al-Kuwaiti won?' --stage GradDiff
```

检查点使用相对固定基座的无损压缩归档，恢复时核验完整模型 SHA256。完成全部评价后生成分享包：

```sh
.venv/bin/python export_full.py
```

输出 `exports/tofu-full-reproduction.zip`，包含原源码、脚本、数据、结果及 GA/GD 无损检查点归档；基座和 Python 环境需按安装脚本下载。归档不是训练适配器，恢复后得到完整全参数模型。

## 历史实验与边界

`reproduction_v2/` 为已被替代的 LoRA 诊断；更早的 `artifacts/light/`、根目录旧报告及旧分享包也不属于本次正式结论。保留历史失败记录，避免混用。正式入口、报告和分享包均使用 `full` 名称。

本次只复现官方的 1B / forget01 / GA-GD 子集及核心指标，不覆盖全部模型、方法、隐私攻击或多种子实验。Windows / 笔记本 4090 尚未在本项目中验证。

技术栈：Python、PyTorch MPS、原版 Transformers / Accelerate、OpenUnlearning、NumPy 文件映射、SciPy、rouge-score、Matplotlib，展示为静态 HTML。四名成员分别负责背景数据、训练方法、评价核验和结果展示，无需设置组长。
