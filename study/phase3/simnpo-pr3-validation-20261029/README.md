# SimNPO GPU probe passed / SimNPO GPU 试跑通过

Validated on 2026-10-29 against PR #3 commit `509be4a9e1dd229a8397d178c784f5d1b463fae6`. No PR code changes were required for this validation.

2026 年 10 月 29 日验证上述 PR #3 提交，本次验证不需要修改组员代码。

## Passed checks / 已通过检查

- All 24 submitted CPU tests passed, with no skips, including the direct upstream SimNPO loss/gradient comparison.
- Tiny-model real Trainer/Accelerate checks passed on CPU FP32 and CUDA BF16 with accumulation 8 and matched sampling. See `trainer-parity.json` for the test's scope and measured errors.
- The real full-parameter model trained all 1,235,814,400 parameters for 3 optimizer updates / 96 sample visits, using BF16 weights and dense FP32 streamed AdamW states. Seed=0, alpha=1, gamma=0.125, beta=4.5, delta=0. The scheduler retained the 10-update horizon.
- Every parameter remained unchanged at the zero-learning-rate warmup. Losses and gradient norms remained finite; the monitored q-projection slice changed in 438 positions after nonzero-rate updates.
- Full checkpoint saving, SHA256 verification, fresh-process reload, and all eight reload-check questions passed. The independent CPU evidence verifier passed, including exact source snapshots. Same-seed sampling matches the previous GD probe: True.

24 项 CPU 测试全部通过；极小模型的真实 CPU/CUDA Trainer 流程通过。实际全参数 1B 模型完成三次更新、96 次样本访问，零学习率不变性、有限损失和梯度、非零学习率参数变化、检查点保存、新进程重载、八题评价及独立 CPU 证据核验均通过。

## Resources / 资源

- Trainer runtime: 332.24 seconds; training entry through save/hash: 356.62 seconds.
- CUDA peak allocated/reserved: 6.8929 / 7.8555 GiB.
- Minimum observed step-end available RAM: 0.0959 GiB.
- Reload evaluation: 7.08 seconds, 2 questions per domain across 4 domains.

以上为本次实际测量值。首次因可用内存不足而未启动训练；用户释放资源后重新检查通过，之后完成试跑。

## Evidence and boundary / 证据与范围

The complete run, including its full checkpoint, is archived locally under `study/runs/simnpo-s0-cuda-probe-v1/`. The original isolated-checkout copy is retained. Signed metadata preserves the original output path; the archive does not rewrite experiment provenance. The archive checkpoint SHA256 is `8ee182394eb89f62b369ffa7e26b89f0c3927882164e38f66d66c7d86f20ae65`.

完整实验及检查点已保存到上述本地目录，独立副本中的原始记录保留。签名配置维持原输出路径，不改写实验来源。

This is a resource/numerical/reload probe, not a formal forgetting-quality result. No 10-update SimNPO run or complete 657-question evaluation has been executed. No FQ/MU is claimed from eight questions. PR #3 has not been merged or commented on by this validation task.

这是资源、数值及重载试跑，不是正式遗忘效果结果。尚未执行 SimNPO 十次更新与完整 657 题评价，八题检查不提供正式 FQ/MU。本次验证没有合并 PR 或向组员发送消息。

To independently verify the archived run from the main local project directory:

```powershell
& '.\.venv-cuda\Scripts\python.exe' -X utf8 verify_cuda_run.py simnpo-s0-cuda-probe-v1
```

The next experiment, if authorized, is a fresh run ID for 10 updates followed by the full 657-question evaluation, keeping the same pinned model/data, sampling, and evaluation protocol.

后续正式实验需使用新的 run_id，完成十次更新和完整 657 题评价，继续保持固定模型、数据、采样及评价协议。


Follow-up on 2026-10-29: the first formal SimNPO run is now complete and verified; see [formal results](../simnpo-formal-20261029/RESULTS.md). This probe report describes the earlier validation stage.

2026-10-29 后续：首组 SimNPO 正式实验现已完成并核验，见上方正式结果链接。本试跑报告记录此前的验证阶段。
