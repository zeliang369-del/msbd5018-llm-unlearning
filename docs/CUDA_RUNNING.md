# Dell CUDA workflow

This is an isolated CUDA GA/GD workflow. **The actual full-parameter 1B probe, full CUDA baseline, and both ten-update GA/GD seed-0 controls with 657-question evaluations have completed and passed CPU evidence verification.** The Dell now matches the Mac's completed baseline/GA/GD stage; numerical equality across devices is not claimed. The frozen Mac code and results remain unchanged.

## Current machine and validation

Inspected 2026-10-21 (Asia/Shanghai): Windows 11 build 26200, RTX 4060 Laptop GPU, 8188 MiB VRAM, driver 596.08, 15.69 GiB system RAM, about 289 GiB free disk. Available RAM fluctuated between about 1.8 and 3.4 GiB. WSL reported that it is not installed. Python's normal WindowsApps command was unusable; a project-local `.venv-cuda` was created using the available Python 3.12.14 runtime.

At initial setup, remote branches were checked: `main` = `e7ab555699f7a75a1c74a67c6704acbda4e89e9b`; `docs/four-member-online-workflow` = `15077d0b89073edacd8b7d9b525dd2cce6192f8a`. Original local experiments used `codex/dell-cuda-workflow`, based on the latter. The CUDA publication is prepared separately from `main`, without incorporating the pending team-workflow draft.

Passed: the existing `verify_results.py`, four output/configuration guard tests, Python syntax compilation, command-line help screens, dependency consistency, small-model CPU/CUDA numerical checks, the real 1B resource probe, and full baseline calibration.

## Actual validation evidence

- `study/environment/dell-inventory.json`: PyTorch **2.9.1+cu128**, CUDA runtime **12.8**, `cuda_available=true`, BF16 supported. Transformers 4.45.1 and Accelerate 0.34.2 are installed. `installed-dependencies.txt` records the resolved environment; `pip check` passed.
- `study/environment/cuda-adapter-numerics.json`: original GD versus sequential CUDA BF16 gradients agree within approximately **0.19–0.30%** across three alpha/gamma settings; CPU FP32 relative error is approximately 1e-7. CPU and CUDA streamed AdamW FP32-master maximum error versus standard AdamW was **1.164e-10**. Zero-rate invariance, nonzero-rate changes, Windows mmap alignment, same-seed sampling, changed-seed sampling, and **10 updates / 320 visits / eight passes** all passed on real tiny Llama models.
- `study/runs/gd-a1-s0-cuda-probe-v1/`: actual full **1,235,814,400-parameter** training, GD alpha=gamma=1, seed 0, **three updates / 96 visits**. Every parameter was checked unchanged after the zero-rate warmup. The monitored q-projection slice changed in 514 positions after the third update. Losses and gradient norms were finite. Sampling matched the first 96 pairs of the frozen Mac run exactly.
- Probe training wall time: **259.1492 seconds**. Peak allocated CUDA memory: **6.1565 GiB**; peak reserved: **6.6230 GiB**. Step-end RSS reached **7.4912 GiB**, and available system RAM fell to **0.2236 GiB**. These RAM readings show substantial pressure; GPU fit does not establish comfortable system RAM headroom.
- The probe saved a normal full checkpoint, hashed it, reloaded it in a new process, and evaluated **two questions in each of four domains**. All eight prompt/target texts matched the official reference. This small check is not a formal FQ/MU result. `probe_passed.json` records completion.
- `study/runs/baseline-s0-cuda-v1/evaluation/`: all **40/400/100/117 = 657** questions completed, with complete raw outputs and original aggregate metrics. Baseline **FQ=0.0143015480**, **MU=0.5991077363**. All input/target texts matched official records. Elapsed evaluation time: **173.9380 seconds**, allocated GPU peak **4.5568 GiB**.
- The saved Mac baseline evaluation took **1285.1575 seconds** under the matched question/decoding protocol. The roughly 7.4x elapsed-time difference is a single historical observation, with uncontrolled background/environment effects; it is not a pure hardware benchmark or a training-speed claim. See `DELL_VS_MAC.md`.
- `study/runs/gd-a1-s0-cuda-v1/`: GD alpha=gamma=1, seed 0, **ten optimizer updates / 320 visits / eight passes**, with all 320 sampled pairs identical to the frozen Mac GD run. Training wall time was **894.7177 seconds**; the entrypoint's total through checkpoint saving was **914.8590 seconds**. Allocated/reserved GPU peaks were **6.1565 / 6.6230 GiB**, step-end RSS peaked at **7.9494 GiB**, and available system RAM reached a minimum of **0.1277 GiB**. Full checkpoint saving/reloading and the eight-question reload check passed.
- GD complete evaluation: **657 questions**, **FQ=0.5786001417**, **MU=0.3904962248**, elapsed **196.2030 seconds**. Checkpoint SHA256: `cfc933da860051ee2f51e09fd8fc3508e3d7eaaffc3e787d79100c6b765cd766`.
- `study/runs/ga-s0-cuda-v1/`: GA seed 0, **ten optimizer updates / 320 visits / eight passes**, all 320 sampled pairs identical to the frozen Mac GA run. Training wall time was **901.8951 seconds**; entrypoint time through checkpoint saving was **935.0320 seconds**. All parameters passed the zero-rate invariance check, gradients/losses were finite, and the monitored q-projection slice changed in 1,103 positions. Checkpoint saving and fresh-process eight-question reload evaluation passed. Allocated/reserved GPU peaks were **6.1565 / 6.6230 GiB**, step-end RSS peaked at **8.1382 GiB**, and minimum available RAM was **0.1191 GiB**.
- GA complete evaluation: **657 questions**, **FQ=0.4045874057**, **MU=0.4109146497**, elapsed **217.2650 seconds**. Checkpoint SHA256: `d8fc3fc1680055f600b2524e299dbb26132718d543957f55de837ec48ac3be76`.
- `verify_cuda_run.py` independently recomputed FQ and all nine MU components from the saved per-question records and checked counts, input texts, source snapshots, checkpoint hashes, updates, seeds, traces, and finite gradients. CPU verification passed for baseline, probe, GA, and GD controls; reports are `cpu_verification.json` in each run directory. This verification does not run a model.

These CUDA results are separate controls. The Mac GA FQ/MU were 0.4046/0.4019 and GD were 0.7659/0.3887; CUDA numerical results are not identical despite matched sampling. These are not additional seeds, and no cause of the device-dependent trajectory differences has been isolated. In this CUDA seed-0 comparison, GD has higher FQ (0.5786 versus 0.4046), while GA has higher MU (0.4109 versus 0.3905). This is a single-seed comparison, not a general ranking. The old Mac finding remains unchanged.

Earlier download failures and the pre-installation `dell-cuda-preflight-20261021` failure are preserved. They no longer describe the current installed environment. The system proxy caused very slow package downloads; `NO_PROXY=*` was applied only to installer child processes. Windows proxy settings were not changed. The model was fetched through the public `hf-mirror.com` mirror at the original pinned revision, verified against the original weight SHA256, and its tokenizer/generation files matched frozen hashes. See `artifacts/model/cuda-download-provenance.json`.

## Install and inspect (PowerShell, repository root)

```powershell
& '.\.venv-cuda\Scripts\python.exe' -X utf8 download_pinned_wheel.py
& '.\.venv-cuda\Scripts\python.exe' -X utf8 install_cuda_environment.py
# Or use the ordinary pip installation path below with a working connection.
& '.\.venv-cuda\Scripts\python.exe' -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128 --cache-dir .cache/pip
& '.\.venv-cuda\Scripts\python.exe' -m pip install -r requirements-cuda.txt --cache-dir .cache/pip
& '.\.venv-cuda\Scripts\python.exe' -X utf8 cuda_common.py
& '.\.venv-cuda\Scripts\python.exe' -X utf8 verify_results.py
& '.\.venv-cuda\Scripts\python.exe' -X utf8 -m unittest test_cuda_workflow -v
& '.\.venv-cuda\Scripts\python.exe' -X utf8 test_cuda_numerics.py
```

CUDA 12.8 / PyTorch 2.9.1 is listed for Windows on the [official PyTorch previous-versions page](https://pytorch.org/get-started/previous-versions/). The driver advertises CUDA 13.2; this is not proof that a PyTorch runtime works. Keep Transformers 4.45.1 and Accelerate 0.34.2. `requirements-light.lock` is unsuitable for this environment because it contains newer framework versions.

Use the existing pinned downloader after supplying a reachable proxy:

```powershell
# Replace this example with an actually reachable proxy, if one is needed.
& '.\.venv-cuda\Scripts\python.exe' -X utf8 fetch_light_model.py --proxy 'http://127.0.0.1:7892'
# For a working direct connection instead:
& '.\.venv-cuda\Scripts\python.exe' -X utf8 fetch_light_model.py --proxy ''
# The tested mirror path preserves the original revision and weight hash:
& '.\.venv-cuda\Scripts\python.exe' -X utf8 fetch_cuda_model.py --endpoint https://hf-mirror.com
```

Do not change the model revision or use generic Llama weights. The downloader verifies the pinned weight SHA256. `archive_checkpoint.py restore SOURCE DESTINATION` restores released Mac checkpoints against that exact base and verifies the complete resulting file; it does not provide the base itself. New CUDA controls start from the public TOFU-trained full checkpoint, rather than a trained Mac GA/GD checkpoint.

## Execute in this order

The saved `v1` runs are complete. Use the CPU review commands below to inspect them. The final metadata guard also binds the model YAML/dependency recipe and automatically saves exact source snapshots; that guard was strengthened after the baseline/GD `v1` experiments and before GA `v1`, without changing loss, optimizer arithmetic, sampling, precision, or budget. Fresh examples therefore use `v2` IDs. A cached GPU entrypoint rejects an older code signature rather than treating it as a fresh experiment.

Close unnecessary applications first. The loader requires at least 3 GiB available system RAM and 25 GiB free disk for training. This is a conservative preflight reserve, not a guarantee of success. The real probe must measure the remaining requirements on the 8 GiB GPU.

```powershell
# Three real optimizer updates; the scheduler still has the formal ten-update horizon.
& '.\.venv-cuda\Scripts\python.exe' -X utf8 train_cuda.py --method GradDiff --seed 0 --alpha 1 --gamma 1 --run-id gd-a1-s0-cuda-probe-v2 --probe

# Only after the probe passes: baseline calibration, full 657 questions.
& '.\.venv-cuda\Scripts\python.exe' -X utf8 evaluate_cuda.py --method baseline --seed 0 --run-id baseline-s0-cuda-v2

# Then seed-0 CUDA control: ten updates, followed automatically by a small reload check.
& '.\.venv-cuda\Scripts\python.exe' -X utf8 train_cuda.py --method GradDiff --seed 0 --alpha 1 --gamma 1 --run-id gd-a1-s0-cuda-v2
& '.\.venv-cuda\Scripts\python.exe' -X utf8 evaluate_cuda.py --method GradDiff --seed 0 --alpha 1 --gamma 1 --run-id gd-a1-s0-cuda-v2
```

Review the completed evidence without GPU inference:

```powershell
& '.\.venv-cuda\Scripts\python.exe' -X utf8 verify_results.py
& '.\.venv-cuda\Scripts\python.exe' -X utf8 verify_cuda_run.py gd-a1-s0-cuda-probe-v1
& '.\.venv-cuda\Scripts\python.exe' -X utf8 verify_cuda_run.py baseline-s0-cuda-v1
& '.\.venv-cuda\Scripts\python.exe' -X utf8 verify_cuda_run.py gd-a1-s0-cuda-v1
& '.\.venv-cuda\Scripts\python.exe' -X utf8 verify_cuda_run.py ga-s0-cuda-v1
```

Run one command at a time. Both CUDA seed-0 controls and the six phase-two extensions are now complete; the extension queue has exited. CUDA seed-0 controls are not additional seeds, and must not be pooled with Mac seed-0 results as seed variation.

## Records and invariants

`--method`, `--seed`, GD `--alpha`/`--gamma`, `--run-id`, and `--output-dir` are supported. Output defaults to `study/runs/<run-id>/`; a custom output must remain beneath `study/runs/` and end with the same run-id. Configurations, source hashes, and code commit bind each directory; different settings or code cannot overwrite it. Failed training is preserved and is not resumable. Evaluation can resume matching partial outputs. A process lock serializes the new CUDA entrypoints; do not launch legacy GPU scripts alongside them.

The workflow imports the fixed original losses, data preprocessing, and evaluation functions. GD uses the existing sequential backward adapter. All 1,235,814,400 parameters remain trainable, with BF16 model weights and dense FP32 AdamW master weights/moments on CPU/disk. A Windows-specific mmap adapter changes flush alignment, not optimizer precision or formulas. No automatic batch-size or precision fallback is provided.

Training sets the seed in the model/runtime and Trainer/data settings; GD coefficients reach the actual loss. It keeps learning rate 1e-5, microbatch 4, accumulation 8, token-mean loss reduction, 192-token ignored padding, weight decay .01, Adam betas .9/.999, epsilon 1e-8, clipping 1, one warmup step, and the linear ten-update schedule. Actual 1B counts passed: probe 3/96, formal 10/320 (eight full data passes).

Per-step evidence includes the learning rate used, finite gradient norms, a parameter-change sentinel, CUDA memory peaks, RAM/RSS, and free disk. The first zero-rate update must preserve every parameter. Nonzero-rate updates must change the sentinel before completion. Training saves the sampling trace, Trainer settings/logs, and checkpoint SHA256. A new process reloads that saved checkpoint and evaluates two questions per domain. A probe only receives `probe_passed.json` after that succeeds.

Small evaluations are stored under `reload_check/`, never labeled as full FQ/MU results. Formal evaluation is under `evaluation/`, requires 40/400/100/117 questions, retains the fixed date and greedy 200-token protocol, and saves full raw records plus the original aggregate metrics. Baseline evaluation also measures text agreement and NLL/ROUGE differences against the official imported full-model records. Hardware/precision differences remain explicit.

Baseline, both GA/GD seed-0 CUDA controls, and all six planned weight/seed extensions are complete and verified. New methods and robustness inference remain pending. The original Mac finding that GD MU was lower than GA MU remains preserved, and the same ordering appears in the CUDA seed-0 comparison; across CUDA seeds 1 and 2, GD MU exceeds GA. See [phase-two results](PHASE2_RESULTS.md) for the complete seed/weight comparison, means, sample standard deviations, charts, and handoff.

Completed GA control commands (already executed; do not retrain into the same directory):

```powershell
& '.\.venv-cuda\Scripts\python.exe' -X utf8 train_cuda.py --method GradAscent --seed 0 --run-id ga-s0-cuda-v1
& '.\.venv-cuda\Scripts\python.exe' -X utf8 evaluate_cuda.py --method GradAscent --seed 0 --run-id ga-s0-cuda-v1
```

## Changed files

- CUDA execution and identity: `train_cuda.py`, `evaluate_cuda.py`, `cuda_runtime.py`, `cuda_common.py`.
- Reproducible setup: `requirements-cuda.txt`, `download_pinned_wheel.py`, `fetch_cuda_model.py`, `install_cuda_environment.py`.
- Evidence and checks: `capture_cuda_source.py`, `verify_cuda_run.py`, `test_cuda_workflow.py`, `test_cuda_numerics.py`, `study/environment/`, and the four completed `study/runs/` directories.
- Documentation/packaging: this file, `docs/DELL_VS_MAC.md`, and `.gitignore` rules excluding model weights, scratch state, and temporary partial files.

The publication contains the isolated CUDA workflow, verified evidence, and current README. Proposals and role documents were not rewritten, and no team notifications were sent. Historical runs record their actual base Git commit plus exact dirty-source hashes and snapshots; the newer publication commit does not replace that provenance. The frozen Mac scripts, protocol, and result files remain intact.

## Completed phase-two extensions

`run_phase2_cuda.py` executed the six agreed extensions serially and stopped accepting each run until its training, fresh-process reload check, 657-question evaluation, and independent CPU verification passed. `study/phase2/execution.json` records the matrix and completion. The cross-run audit also passed: all shared execution mathematics and protocols match, same-seed traces match across methods and weights, and seeds 0/1/2 have different traces. The audit covers nine full CUDA evaluations (baseline plus eight trained models).

Use `audit_phase2_cuda.py` in the CUDA environment for CPU-only review. `report_phase2_cuda.py` renders the verified records with the separate `requirements-report.lock` environment; no plotting package was added to the CUDA environment. Completed run directories must not be used for retraining. New phase-three work remains separate.

Results and next team actions: [PHASE2_RESULTS.md](PHASE2_RESULTS.md). Execution and acceptance details: [PHASE2_STATUS.md](PHASE2_STATUS.md). Zeliang authorized GitHub publication after reviewing the phase-two results; team notification remains with Zeliang.
