# Sources and notices

## Framework and data

- OpenUnlearning snapshot: `open-unlearning-original/`, commit `2e16546d1b8033346a55b301166f48861b879062`. Original MIT copyright/license remain there and in `LICENSES/OpenUnlearning_MIT.txt`. The source archive and manifest in `reproduction_full/` support the unchanged-core audit.
- TOFU data: `locuslab/TOFU`, revision `324592d84ae4f482ac7249b9285c2ecdb53e3a68`. Its pinned card declares MIT and is retained in `LICENSES/TOFU_DATASET_CARD.md`. Hashes are in `artifacts/tofu/manifest.json`.
- Official evaluation records: `open-unlearning/eval`, revision `f31717fe522c2a6725ca0024980f51169a084ca1`. They are imported references, not local retraining. See `artifacts/tofu/reference-metadata.json`.

## Model

**Built with Llama.** Base: `open-unlearning/tofu_Llama-3.2-1B-Instruct_full`, revision `88e31200b97e4c0c04ae0d2f0b591f427046d192`, derived from Llama 3.2. Base weights are downloaded separately, not committed. Derived checkpoint packages include the Llama 3.2 license and attribution notice; model terms are separate from the framework's MIT license.

Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.

See `LICENSES/LLAMA_3_2_LICENSE.txt`, `LICENSES/Notice.txt`, and the model publisher's metadata. License source: Meta's `llama-models/models/llama3_2/LICENSE`. The framework license does not replace underlying model terms.

## Research references

These sources support the [research background and scope](docs/RESEARCH_REPORT.md#1-research-questions-and-context) ([中文](docs/RESEARCH_REPORT.zh-CN.md#1-研究问题与背景)) and Part 1 of the group presentation. Numbering follows the presentation's reference list.

1. Bourtoule et al. (2021). *Machine Unlearning*. IEEE Symposium on Security and Privacy. [arXiv:1912.03817](https://arxiv.org/abs/1912.03817). Introduces SISA and selective retraining through a training-time organization.
2. Maini et al. (2024). *TOFU: A Task of Fictitious Unlearning for LLMs*. [arXiv:2401.06121](https://arxiv.org/abs/2401.06121). Provides the fictitious-author benchmark and forgetting–utility evaluation context.
3. Zhang et al. (2024). *Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning*. COLM. [arXiv:2404.05868](https://arxiv.org/abs/2404.05868). Defines the reference-relative negative-preference objective.
4. Fan et al. (2025). *Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning*. NeurIPS. [arXiv:2410.07163](https://arxiv.org/abs/2410.07163). Develops a reference-free, length-normalized objective; first posted as a preprint in 2024.
5. Li et al. (2024). *The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning*. ICML. [Proceedings](https://proceedings.mlr.press/v235/li24bc.html) · [arXiv:2403.03218](https://arxiv.org/abs/2403.03218). Introduces WMDP and the RMU representation-based method.
6. Lynch et al. (2024). *Eight Methods to Evaluate Robust Unlearning in LLMs*. [arXiv:2402.16835](https://arxiv.org/abs/2402.16835). Motivates evaluation beyond responses to the original wording.
7. Dorna et al. (2025). *OpenUnlearning: Accelerating LLM Unlearning via Unified Benchmarking of Methods and Metrics*. NeurIPS Datasets and Benchmarks. [arXiv:2506.12618](https://arxiv.org/abs/2506.12618). Provides a unified framework for unlearning methods and evaluation metrics.

GA, GD, NPO, and SimNPO are evaluated in the completed experiment grid using TOFU and the pinned OpenUnlearning source. SISA and WMDP/RMU are related-work context only. Our prompt study does not reproduce the full eight-test suite of Lynch et al. or SimNPO's relearning attacks. Literature findings are not additional experimental results from this repository. These references do not alter the software, dataset, or model notices above.
