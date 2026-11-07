# CUDA 训练与评估

[English](CUDA_RUNNING.md) | **简体中文**

本手册用于通过当前入口执行**新的**实验。只核验已发布结果、不重新训练时，请使用[复现指南](REPRODUCIBILITY.zh-CN.md)。以下均为在仓库根目录运行的 PowerShell 命令；相关步骤会实际下载模型、训练或推理。

## 环境与固定协议

已记录的训练平台为 Windows 11、Python 3.12.14、PyTorch 2.9.1+cu128、Transformers 4.45.1、Accelerate 0.34.2、8 GB 显存的 RTX 4060 Laptop GPU，以及约 16 GB 系统内存。这是实测配置，不是最低资源保证；训练存在明显内存压力。训练预检要求至少 3 GiB 可用内存及 25 GiB 可用磁盘，保留更多检查点还需要额外空间。

实验使用 `artifacts/model/` 中固定的 TOFU 微调 Llama-3.2-1B-Instruct 检查点、全部 1,235,814,400 个可训练参数，以及原始 forget01/retain99 数据。正式训练为 10 次优化器更新、320 次遗忘样本访问，microbatch 为 4、梯度累积为 8、学习率为 1e-5，并固定填充至 192 tokens。完整评估在遗忘、保留、真实作者及世界事实四个领域分别包含 40/400/100/117 题，共 657 题。探测实验使用同一个十步调度，在三次更新后停止，不计作正式结果。

## 1. 安装与检查

需要可用的 Python 3.12 和 Windows `curl.exe`。创建独立 CUDA 环境，后续始终使用其中的解释器：

```powershell
py -3.12 -m venv .venv-cuda
$cudaPython = '.\.venv-cuda\Scripts\python.exe'
function Invoke-Cuda {
    & $cudaPython -X utf8 @args
    if ($LASTEXITCODE -ne 0) { throw 'Command failed; inspect the output before continuing.' }
}
Invoke-Cuda -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
Invoke-Cuda -m pip install -r requirements-cuda.txt
Invoke-Cuda -m pip check
Invoke-Cuda cuda_common.py
Invoke-Cuda verify_results.py
Invoke-Cuda -m unittest test_cuda_workflow -v
```

后续命令在同一个 PowerShell 窗口中运行。辅助函数会在命令失败时中止后续序列。不要将 `requirements-light.lock` 或绘图依赖安装到训练环境。[`requirements-cuda.txt`](../requirements-cuda.txt) 固定框架版本，[图表复现流程](REPRODUCIBILITY.zh-CN.md#2-重建当前研究图表)使用独立环境。

## 2. 获取固定初始检查点

先阅读[模型许可与署名要求](../THIRD_PARTY.md)，再下载固定模型：

```powershell
Invoke-Cuda fetch_cuda_model.py --endpoint https://huggingface.co
```

下载器验证固定模型权重的 SHA256 并记录下载来源。它也支持 `https://hf-mirror.com` 镜像，仅在需要时使用，并保持相同的版本与哈希。模型为 `open-unlearning/tofu_Llama-3.2-1B-Instruct_full`，固定 revision 是 `88e31200b97e4c0c04ae0d2f0b591f427046d192`。通用 Llama 权重不是等价起点。保留集 Oracle 的评估记录来自外部导入，并非本地训练。

## 3. 选择新运行编号并准备 NPO

源码或配置变化后，不复用已完成或失败的训练编号。以下时间戳用于生成独立示例编号；新尝试重新生成时间戳。NPO 需要从**初始模型**生成、覆盖全部 40 条样本的已验证参照 NLL 缓存，不能使用遗忘后的检查点作为参照。

```powershell
$runTag = Get-Date -Format 'yyyyMMdd-HHmmss'
$referenceCache = ".cache/npo-reference-$runTag.json"
Invoke-Cuda prepare_npo_reference.py --checkpoint artifacts/model --output $referenceCache
```

缓存准备会执行模型推理。正式训练不要添加 `--limit`；有限缓存属于探测材料，会被正式训练拒绝。缓存元数据绑定初始检查点、分词器、样本身份、192-token 协议及完整缓存哈希。缓存生成器不覆盖已有输出。

## 4. 探测、训练、评估与核验

逐块运行，只在前一块检查通过后继续。入口通过锁串行执行受支持的 CUDA 任务，不要同时启动旧版 GPU 脚本。训练会保存完整检查点，并自动启动新进程对每个领域各两题执行重载检查；该检查通过后探测才算成功。

先检查 GD 的真实资源路径，并校准固定基线：

```powershell
Invoke-Cuda train_cuda.py --method GradDiff --seed 0 --alpha 1 --gamma 1 --run-id "gd-probe-$runTag" --probe
Invoke-Cuda verify_cuda_run.py "gd-probe-$runTag"
Invoke-Cuda evaluate_cuda.py --method baseline --seed 0 --run-id "baseline-$runTag"
Invoke-Cuda verify_cuda_run.py "baseline-$runTag"
```

以下示例复现单个种子下的默认系数。命令行接受 `GradAscent` 和 `GradDiff`，不接受展示用缩写 GA 和 GD。

```powershell
# GA：原始无保留项加权目标。
Invoke-Cuda train_cuda.py --method GradAscent --seed 0 --run-id "ga-probe-$runTag" --probe
Invoke-Cuda verify_cuda_run.py "ga-probe-$runTag"
Invoke-Cuda train_cuda.py --method GradAscent --seed 0 --run-id "ga-$runTag"
Invoke-Cuda evaluate_cuda.py --method GradAscent --seed 0 --run-id "ga-$runTag"
Invoke-Cuda verify_cuda_run.py "ga-$runTag"

# GD：保留项权重 alpha=1，gamma=1。
Invoke-Cuda train_cuda.py --method GradDiff --seed 0 --alpha 1 --gamma 1 --run-id "gd-$runTag"
Invoke-Cuda evaluate_cuda.py --method GradDiff --seed 0 --alpha 1 --gamma 1 --run-id "gd-$runTag"
Invoke-Cuda verify_cuda_run.py "gd-$runTag"
```

```powershell
# SimNPO：显式传入 gamma；共享默认值为 1。
Invoke-Cuda train_cuda.py --method SimNPO --seed 0 --alpha 1 --gamma 0.125 --beta 4.5 --delta 0 --run-id "simnpo-probe-$runTag" --probe
Invoke-Cuda verify_cuda_run.py "simnpo-probe-$runTag"
Invoke-Cuda train_cuda.py --method SimNPO --seed 0 --alpha 1 --gamma 0.125 --beta 4.5 --delta 0 --run-id "simnpo-$runTag"
Invoke-Cuda evaluate_cuda.py --method SimNPO --seed 0 --alpha 1 --gamma 0.125 --beta 4.5 --delta 0 --run-id "simnpo-$runTag"
Invoke-Cuda verify_cuda_run.py "simnpo-$runTag"

# NPO：训练与评估使用同一份完整参照缓存。
Invoke-Cuda train_cuda.py --method NPO --seed 0 --alpha 1 --gamma 1 --beta 0.1 --reference-cache $referenceCache --run-id "npo-probe-$runTag" --probe
Invoke-Cuda verify_cuda_run.py "npo-probe-$runTag"
Invoke-Cuda train_cuda.py --method NPO --seed 0 --alpha 1 --gamma 1 --beta 0.1 --reference-cache $referenceCache --run-id "npo-$runTag"
Invoke-Cuda evaluate_cuda.py --method NPO --seed 0 --alpha 1 --gamma 1 --beta 0.1 --reference-cache $referenceCache --run-id "npo-$runTag"
Invoke-Cuda verify_cuda_run.py "npo-$runTag"
```

完整 657 题评估不能使用 `--limit`，评估完成后才能将训练配置视为正式结果。复现完整设计时，使用种子 0/1/2；GD、SimNPO、NPO 的 `alpha` 分别取 0.25/0.5/1/2/4，其余方法系数固定。GA 不做保留项权重扫描。每个配置使用独立编号，训练与评估参数保持一致。

## 5. 生成四类提示回答

正式检查点与完整评估通过核验后：

```powershell
Invoke-Cuda evaluate_robustness_cuda.py --model-run "simnpo-$runTag" --run-id "robust-simnpo-$runTag"
Invoke-Cuda verify_robustness_cuda.py "robust-simnpo-$runTag"
```

对其他已验证模型运行编号重复执行，并使用独立提示运行编号。每次完整提示实验覆盖 40 道遗忘题、40 道保留题及四类问法，共 320 条回答。`--limit 5` 则生成独立的 40 条回答探测，不能计作完整的 320 条实验。Direct 与替代提示应来自同一提示实验。ROUGE 文本重合是代理指标，不是事实恢复标签。

## 证据与失败处理

输出默认位于 `study/runs/<run-id>/`。自定义输出目录必须位于 `study/runs/` 下，并以运行编号结尾。配置、Git/源码身份、缓存哈希及保存的源码快照共同绑定每次运行，不匹配的已有目录会被拒绝。保留失败的部分输出；失败训练不能断点续训，排查后使用新编号。配置匹配的部分评估记录可以续跑。

`verify_cuda_run.py` 不执行模型推理，但校验哈希需要匹配的本地模型／检查点文件。Git 不分发完整 CUDA 检查点、优化器临时文件、环境及生成的参照缓存。没有这些权重时，使用三项[已保存交付核验](REPRODUCIBILITY.zh-CN.md#1-无需-gpu-核验已发布证据)。数值或文件完整性检查通过，不代表独立人工标注完成，也不证明遗忘方法存在普遍优劣排序。
