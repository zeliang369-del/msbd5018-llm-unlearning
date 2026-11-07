# CUDA training and evaluation

**English** | [简体中文](CUDA_RUNNING.zh-CN.md)

Use this runbook for **new** experiments with the current entrypoints. To inspect the published results without training, use the [reproducibility guide](REPRODUCIBILITY.md). All commands below are PowerShell commands run from the repository root. They launch real downloads, training, or inference where indicated.

## Environment and fixed protocol

The recorded training platform is Windows 11, Python 3.12.14, PyTorch 2.9.1+cu128, Transformers 4.45.1, Accelerate 0.34.2, an RTX 4060 Laptop GPU with 8 GB VRAM, and approximately 16 GB system RAM. This is an observed configuration, not a minimum-resource guarantee. Training experienced substantial RAM pressure. The training preflight requires at least 3 GiB available RAM and 25 GiB free disk; additional completed checkpoints consume additional storage.

The experiment uses the pinned TOFU-trained Llama-3.2-1B-Instruct checkpoint in `artifacts/model/`, all 1,235,814,400 trainable parameters, and the original forget01/retain99 data. Formal training uses 10 optimizer updates, 320 forget-sample visits, microbatch 4, accumulation 8, learning rate 1e-5, and fixed 192-token padding. Full evaluation covers 40/400/100/117 questions across forget/retain/real-author/world-fact domains, totaling 657. Probes stop after three updates on the same ten-update scheduler horizon and are not formal results.

## 1. Install and inspect

Python 3.12 and Windows `curl.exe` must be available. Create the isolated CUDA environment, then use its interpreter throughout:

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

Keep this shell open for the remaining commands. The helper stops a command sequence on failure. Do not install `requirements-light.lock` or plotting dependencies into this training environment. [`requirements-cuda.txt`](../requirements-cuda.txt) fixes the framework versions; the [figure recipe](REPRODUCIBILITY.md#2-rebuild-the-current-research-figures) uses a separate environment.

## 2. Obtain the fixed initial checkpoint

Review the [model license and attribution](../THIRD_PARTY.md), then download the pinned model:

```powershell
Invoke-Cuda fetch_cuda_model.py --endpoint https://huggingface.co
```

The downloader verifies the pinned model-weight SHA256 and records download provenance. Its supported mirror endpoint is `https://hf-mirror.com`; use it only if needed and retain the same revision and hash. The model is `open-unlearning/tofu_Llama-3.2-1B-Instruct_full` at revision `88e31200b97e4c0c04ae0d2f0b591f427046d192`. Generic Llama weights are not an equivalent starting point. The retain-only Oracle evaluation is an imported reference, not a locally trained model.

## 3. Choose new run IDs and prepare NPO

Never reuse a completed or failed training ID for a changed source/configuration. The timestamp below creates distinct example IDs; regenerate it for a new attempt. NPO needs the verified full 40-sample reference-NLL cache from the **initial model**, not an unlearned checkpoint.

```powershell
$runTag = Get-Date -Format 'yyyyMMdd-HHmmss'
$referenceCache = ".cache/npo-reference-$runTag.json"
Invoke-Cuda prepare_npo_reference.py --checkpoint artifacts/model --output $referenceCache
```

Cache preparation runs model inference. Do not add `--limit` for formal training: a limited cache is a probe artifact and is rejected. Cache metadata binds the initial checkpoint, tokenizer, sample identity, 192-token protocol, and full-cache hash. The cache generator does not overwrite an existing output.

## 4. Probe, train, evaluate, and verify

Run one block at a time, only proceeding after its checks pass. The entrypoints serialize supported CUDA work with a lock; do not run legacy GPU scripts alongside them. Training saves a full checkpoint and automatically launches a new-process reload check with two questions per domain. A probe is accepted only after this check succeeds.

First check the actual GD resource path and calibrate the fixed baseline:

```powershell
Invoke-Cuda train_cuda.py --method GradDiff --seed 0 --alpha 1 --gamma 1 --run-id "gd-probe-$runTag" --probe
Invoke-Cuda verify_cuda_run.py "gd-probe-$runTag"
Invoke-Cuda evaluate_cuda.py --method baseline --seed 0 --run-id "baseline-$runTag"
Invoke-Cuda verify_cuda_run.py "baseline-$runTag"
```

The following blocks reproduce the default coefficients for one seed. The accepted CLI names are `GradAscent` and `GradDiff`, not the display abbreviations GA and GD.

```powershell
# GA: original unweighted objective.
Invoke-Cuda train_cuda.py --method GradAscent --seed 0 --run-id "ga-probe-$runTag" --probe
Invoke-Cuda verify_cuda_run.py "ga-probe-$runTag"
Invoke-Cuda train_cuda.py --method GradAscent --seed 0 --run-id "ga-$runTag"
Invoke-Cuda evaluate_cuda.py --method GradAscent --seed 0 --run-id "ga-$runTag"
Invoke-Cuda verify_cuda_run.py "ga-$runTag"

# GD: retain weight alpha=1, gamma=1.
Invoke-Cuda train_cuda.py --method GradDiff --seed 0 --alpha 1 --gamma 1 --run-id "gd-$runTag"
Invoke-Cuda evaluate_cuda.py --method GradDiff --seed 0 --alpha 1 --gamma 1 --run-id "gd-$runTag"
Invoke-Cuda verify_cuda_run.py "gd-$runTag"
```

```powershell
# SimNPO: pass gamma explicitly; the shared default is 1.
Invoke-Cuda train_cuda.py --method SimNPO --seed 0 --alpha 1 --gamma 0.125 --beta 4.5 --delta 0 --run-id "simnpo-probe-$runTag" --probe
Invoke-Cuda verify_cuda_run.py "simnpo-probe-$runTag"
Invoke-Cuda train_cuda.py --method SimNPO --seed 0 --alpha 1 --gamma 0.125 --beta 4.5 --delta 0 --run-id "simnpo-$runTag"
Invoke-Cuda evaluate_cuda.py --method SimNPO --seed 0 --alpha 1 --gamma 0.125 --beta 4.5 --delta 0 --run-id "simnpo-$runTag"
Invoke-Cuda verify_cuda_run.py "simnpo-$runTag"

# NPO: the identical full reference cache is required for training and evaluation.
Invoke-Cuda train_cuda.py --method NPO --seed 0 --alpha 1 --gamma 1 --beta 0.1 --reference-cache $referenceCache --run-id "npo-probe-$runTag" --probe
Invoke-Cuda verify_cuda_run.py "npo-probe-$runTag"
Invoke-Cuda train_cuda.py --method NPO --seed 0 --alpha 1 --gamma 1 --beta 0.1 --reference-cache $referenceCache --run-id "npo-$runTag"
Invoke-Cuda evaluate_cuda.py --method NPO --seed 0 --alpha 1 --gamma 1 --beta 0.1 --reference-cache $referenceCache --run-id "npo-$runTag"
Invoke-Cuda verify_cuda_run.py "npo-$runTag"
```

Complete the 657-question evaluation without `--limit` before treating a trained setting as a formal result. For the full published design, use seeds 0/1/2; vary `alpha` over 0.25/0.5/1/2/4 for GD, SimNPO, and NPO while holding their other coefficients fixed. GA has no retain-weight sweep. Allocate a distinct run ID to each configuration and keep training/evaluation arguments identical.

## 5. Generate the four-family prompt responses

After a formal checkpoint and its complete evaluation pass verification:

```powershell
Invoke-Cuda evaluate_robustness_cuda.py --model-run "simnpo-$runTag" --run-id "robust-simnpo-$runTag"
Invoke-Cuda verify_robustness_cuda.py "robust-simnpo-$runTag"
```

Repeat with the other verified model-run IDs and unique prompt-run IDs. Each full prompt run covers 40 forget and 40 retain questions in four families, yielding 320 responses. `--limit 5` instead creates a separate 40-response pilot; never include it as a complete 320-response run. Pair Direct and alternative prompts from the same prompt run. ROUGE overlap is a proxy, not a factual-recovery label.

## Evidence and failure handling

Outputs default to `study/runs/<run-id>/`. A custom output directory must remain under `study/runs/` and end with its run ID. Configuration, Git/source identities, cache hashes, and saved source snapshots bind each run; an incompatible existing directory is rejected. Preserve partial failures. Training is not resumable after failure; investigate and use a new ID. Matching partial evaluation records may resume.

`verify_cuda_run.py` checks saved evidence without model inference but requires the matching local model/checkpoint files for their hashes. Full CUDA checkpoints, optimizer scratch files, environments, and generated reference caches are not distributed in Git. Use the three [saved-delivery checks](REPRODUCIBILITY.md#1-verify-the-published-evidence-without-a-gpu) when those weights are unavailable. A passing numerical or file-integrity check does not establish independent human annotation or a universal ranking of unlearning methods.
