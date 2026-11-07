# Evaluation metrics and results

This directory holds the metric verification, experimental write-up, and figures prepared for the course report, plus the recomputed tables used to check those results.

- [`report/`](report/evaluation_metrics_and_results.md) — evaluation metrics, experimental results, six main figures, and two supplementary heatmaps covering the full utility grid and prompt-overlap proxies
- [`supplementary/`](supplementary/) — recomputed FQ/MU tables, fact-annotation recovery counts, retain-set spot-check summaries, and the verification log

Figure labels are in English for paper reuse; the report and figure captions are in Chinese. Each figure is available as a 300 dpi PNG and as vector PDF and SVG files in [`report/figures/`](report/figures/).

To rebuild the figures from the existing supplementary CSV files and the frozen `study/phase3/grid-20261029/prompt-three-seed-summary.csv`, run from the repository root:

```sh
python -m pip install -r scripts/figures/requirements.txt
python scripts/plot_evaluation.py --output-dir docs/evaluation/report/figures
```

The script recomputes plotted means and sample standard deviations from the per-run metrics and uses the recorded annotation counts for factual rates. FQ summaries are descriptive KS p-value summaries. Annotation rates pool the recorded counts; seed points show the observed variation, and repeated questions across settings are not treated as independent observations for confidence intervals.

Supplementary Figure S2 shows ROUGE-L recall proxies for the 17 setting groups and four prompt families, with separate forget and retain panels. These are three-seed means (one fixed evaluation for Baseline), with a shared 0–1 color scale; text overlap is not a factual recovery rate.
