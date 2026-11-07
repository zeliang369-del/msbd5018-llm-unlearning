# Contributing

**English** | [简体中文](CONTRIBUTING.zh-CN.md)

Read the [research report](docs/RESEARCH_REPORT.md), [reproduction guide](docs/REPRODUCIBILITY.md), and [contribution record](docs/TEAM_PROGRESS.md). The repository describes delivered research; proposals and earlier task schedules are not the current specification.

1. Create a topic branch from `main`. Describe the problem, change, verification and limitations in the pull request.
2. Preserve frozen experiment outputs, source snapshots, hashes and original annotation provenance. New settings and code require fresh run IDs; do not overwrite a completed or failed run.
3. Run `python3 verify_results.py`, `python3 verify_phase3_delivery.py`, and `python3 verify_phase3_grid_delivery.py --require-visual` in a disposable checkout. These audit saved evidence and may rewrite audit JSON; inspect changes before committing.
4. Keep English and Chinese principal documents consistent, with reciprocal language links. Preserve metric values, exclusion rules and figure scope across translations.
5. Teammate review is optional under the repository's existing policy; required automated checks must pass. Use Git revert to undo a merged change without rewriting shared history.

Record model/data/source identities, seed, coefficients, sampling, update budget, resource use, failures, checkpoint reload and evaluation scope. Do not describe imported Oracle records as local retraining, cached outputs as new experiments, or AI-assisted labels as independently adjudicated human labels. New factual annotations must preserve case identifiers, reviewer provenance, reasons, exclusions and disagreements.

Do not commit model weights, environments, caches or credentials. Follow the distinct model, data and framework terms in [sources and notices](THIRD_PARTY.md). Preserve authorship independently of upload accounts.
