# Evaluation materials and figures

**English** | [简体中文](README.zh-CN.md)

C (top-linan) authored the evaluation materials and refined the final figures. D (xing-pangdingyu) prepared the initial figures and submitted the evaluation package. These links point to the latest delivery.

- [Full evaluation chapter: five tables and all captions](report/evaluation_metrics_and_results.md)
- [Annotation scope, provenance, denominators and exclusions](../DATA_AND_ANNOTATIONS.md)
- [Label coding rules](supplementary/annotation_coding.md)
- [Verification log](supplementary/verification_log.md)
- [Source workbook and derived statistical tables](supplementary/)
- [Six main and two supplementary figures](report/figures/)

Figure labels are English; the report and captions are available in both languages. Each figure has 300 dpi PNG, vector PDF and SVG exports. Rebuild in a separate plotting environment and output directory:

```sh
python -m pip install -r scripts/figures/requirements.txt
python scripts/plot_evaluation.py --output-dir /tmp/msbd5018-figures
```

Run from the repository root. FQ summaries are descriptive KS p-value summaries, and error bars show sample standard deviations. Annotation rates pool yes/(yes+no), reporting unclear separately. Repeated questions are not treated as independent samples for confidence intervals. Supplementary Figure S2 shows text-overlap proxies, not factual recovery. Figure 3 shows signed changes from Baseline; S1 and S2 show absolute 0–1 scores.
