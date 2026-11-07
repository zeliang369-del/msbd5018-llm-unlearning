# Reproducibility guide

**English** | [简体中文](REPRODUCIBILITY.zh-CN.md)

This guide separates checking the published evidence, rebuilding the figures, and running new experiments. All commands below run from the repository root. The saved results describe a fixed Llama-3.2-1B-Instruct / TOFU `forget01` study; they are not a claim of hardware-independent numerical identity.

## 1. Verify the published evidence without a GPU

Use a complete checkout and Python 3.12, the version used for the recorded CUDA environment. The three saved-record checks use the Python standard library; model downloads and training dependencies are unnecessary for this route.

```sh
python3 verify_results.py
python3 verify_phase3_delivery.py
python3 verify_phase3_grid_delivery.py --require-visual
```

| Check | What it verifies |
|---|---|
| `verify_results.py` | Frozen reproduction records, data, metrics, provenance, and prepared prompts |
| `verify_phase3_delivery.py` | The first phase-three delivery, saved source/report identities, and the original 1,600-response annotation packages |
| `verify_phase3_grid_delivery.py --require-visual` | The complete 49-setting grid, 15,680 prompt outputs, summary arithmetic, annotation-package integrity, and the recorded figure-review binding |

The latter two commands regenerate audit JSON files in the historical study directories. Use a disposable checkout for review, and inspect any resulting changes before committing them. A passing check validates the saved evidence within its stated scope; it does not perform new model inference or independently re-review every figure.

The audit field `human_labels_filled=0` refers to the original blank annotation handoff packages. Those packages remain frozen. The later C workbook and evaluation tables are separate artifacts in [`evaluation/supplementary/`](evaluation/supplementary/); the blank-package count does **not** describe their completion status. Conversely, passing package checks does not certify independent human labeling or adjudication. See the [evaluation report](evaluation/report/evaluation_metrics_and_results.md).

## 2. Rebuild the current research figures

Keep plotting dependencies separate from the CUDA training environment: the two environments pin different NumPy versions. For example, on macOS/Linux:

```sh
python3 -m venv /tmp/msbd5018-figures-venv
/tmp/msbd5018-figures-venv/bin/python -m pip install -r scripts/figures/requirements.txt
/tmp/msbd5018-figures-venv/bin/python scripts/plot_evaluation.py --output-dir /tmp/msbd5018-figures
```

These commands write regenerated figures outside the repository so the published image bytes remain available for comparison. On Windows, use the corresponding virtual-environment `Scripts/python.exe` path and a temporary output directory.

The plotting recipe pins Matplotlib 3.10.1 and NumPy 2.2.6. It reads the existing evaluation CSV tables and the frozen prompt summary, and exports **six main figures and two supplementary figures**, each as PNG, PDF, and SVG. Rebuilding plots does not regenerate model answers or annotations. See the [figure guide](evaluation/README.md) for inputs and statistical conventions.

## 3. Reproduce training and evaluation

The current [CUDA runbook](CUDA_RUNNING.md) gives the setup, probe, fresh-run, evaluation, and verification commands. Original experiment configurations and exact source snapshots remain in the study records so the published evidence can be traced to the code that actually produced it.

The recorded formal CUDA environment was Windows 11, Python 3.12.14, an RTX 4060 Laptop GPU with 8 GB VRAM, and approximately 16 GB system RAM. Core versions were:

| Component | Recorded version or recipe |
|---|---|
| PyTorch | 2.9.1+cu128; CUDA runtime 12.8 |
| Transformers | 4.45.1 |
| Accelerate | 0.34.2 |
| Training dependencies | [`requirements-cuda.txt`](../requirements-cuda.txt) |
| Resolved environment | [`study/environment/`](../study/environment/) |

Follow the runbook to create the environment, obtain the pinned model, inspect resources, and run the probe before formal training. Do not substitute `requirements-light.lock` for the CUDA recipe. The measured configuration experienced substantial system-RAM pressure; successful execution on that machine is not a guarantee that any 8 GB GPU system can run the same workload.

The formal budget is **10 optimizer updates / 320 sample visits** with all **1,235,814,400 parameters** trainable; no LoRA substitution is used. Each complete evaluation contains 40 forget, 400 retain, 100 real-author, and 117 world-fact questions (657 total). The grid comprises three GA seeds and five retain weights × three seeds for each of GD, SimNPO, and NPO, plus one fixed baseline evaluation.

Before launching a new run:

1. Use a fresh run ID whenever code or configuration changes. Saved directories bind configuration and source identity; never overwrite completed evidence. Custom output directories must remain under `study/runs/` and end with the run ID.
2. Run GPU tasks serially and satisfy the resource preflight. Preserve failed runs for diagnosis. Failed training is not resumed as if complete; use a new ID after investigation. Matching partial evaluations may resume.
3. For NPO, prepare and verify the reference-NLL cache from the **fixed initial checkpoint**, covering all 40 forget samples. Limited probe caches are invalid for formal training. The full cache hash is part of run identity.
4. Pass the declared method coefficients explicitly: GD/NPO `gamma=1`; SimNPO `gamma=0.125`, `beta=4.5`, `delta=0`; NPO `beta=0.1`. The shared gamma default is 1, so it does not reproduce the SimNPO setting without an explicit override.
5. Require checkpoint saving, fresh-process reload, complete evaluation, and the independent run checks before counting a run as complete. Keep pilot outputs separate from full evaluations.

The runbook supplies the actual training, cache preparation, evaluation, and prompt-generation commands. Keep newly generated results separate from the published evidence.

## 4. Match artifacts to their source

| Artifact | Location and interpretation |
|---|---|
| Complete grid | [`study/phase3/grid-20261029/`](../study/phase3/grid-20261029/): configuration-level metrics, summaries, paired prompt proxies, and manifests |
| Individual experiments | [`study/runs/`](../study/runs/): recorded configuration, outputs, and provenance; retain the exact source snapshots |
| Annotation and evaluation materials | [`evaluation/supplementary/`](evaluation/supplementary/): later workbook, derived tables, coding rules, and verification log |
| Original annotation handoffs | Frozen ZIPs under `study/phase3/`; blank assignments are not completed reviews |
| Current report and figures | [`evaluation/report/`](evaluation/report/): interpretation and figures based on the existing evidence |

CUDA full model checkpoints, runtime environments, optimizer scratch files, and generated reference caches are excluded from Git. Saved-record checks can run without weights; direct checkpoint-hash/reload verification requires the matching local full checkpoints. An imported retain-only Oracle supplies comparison records and was not trained locally.

## 5. Sources and terms

The [source and license notices](../THIRD_PARTY.md) list the exact upstream revisions. The initial model is `open-unlearning/tofu_Llama-3.2-1B-Instruct_full` at revision `88e31200b97e4c0c04ae0d2f0b591f427046d192`; do not replace it with a generic Llama checkpoint when claiming reproduction of this study.

**Built with Llama.** Model terms remain governed by the [Llama 3.2 Community License](../LICENSES/LLAMA_3_2_LICENSE.txt) and [notice](../LICENSES/Notice.txt), separately from the [OpenUnlearning MIT license](../LICENSES/OpenUnlearning_MIT.txt) and the pinned [TOFU dataset card](../LICENSES/TOFU_DATASET_CARD.md). Preserve source revisions, hashes, licenses, and attribution when redistributing derived materials.
