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

- Maini et al. (2024), *TOFU: A Task of Fictitious Unlearning for LLMs*. https://arxiv.org/abs/2401.06121
- Dorna et al. (2025), *OpenUnlearning: Accelerating LLM Unlearning via Unified Benchmarking of Methods and Metrics*. https://arxiv.org/abs/2506.12618
- Zhang et al. (2024), *Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning*. https://arxiv.org/abs/2404.05868
- Fan et al., *Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning*. https://arxiv.org/abs/2410.07163
- Li et al. (2024), *The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning*. https://arxiv.org/abs/2403.03218

The last three references motivate planned extensions, not completed local experiments.
