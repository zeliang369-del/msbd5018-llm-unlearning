# Data and annotation evidence

**English** | [简体中文](DATA_AND_ANNOTATIONS.zh-CN.md)

[Project overview](../README.md) · [Evaluation report](evaluation/report/evaluation_metrics_and_results.md)

This guide describes the released evidence as of 8 November 2026. Experiment outputs, annotation records, and statistical summaries answer different questions and have different coverage.

## Evidence catalog

| Evidence | Coverage | Location |
| --- | --- | --- |
| Full experiment grid | 48 formal training runs and one fixed Baseline evaluation; 49 evaluations of 657 questions each | [Per-run results](../study/phase3/grid-20261029/all-49-settings.csv), [three-seed summaries](../study/phase3/grid-20261029/three-seed-summary.csv) |
| Prompt responses | 40 forget questions and 40 retain controls, each with four prompt families; 320 answers per setting, 15,680 answers across 49 settings | [Full annotation package](../study/phase3/grid-20261029/annotation-grid-20261029.zip) |
| First-round prompt evidence | 1,600 answers mapped exactly to corresponding full-grid records; a separate 120-answer pilot is retained | [Reuse map](../study/phase3/grid-20261029/first-round-label-reuse-map.csv), [first-round results](../study/phase3/phase3-final-20261029/RESULTS.md) |
| C's completed annotation workbook | 3,920 forget answers and 588 retain spot checks, totaling 4,508 labeled answers | [Source workbook](evaluation/supplementary/C_全部已标注_完成版.xlsx) |
| Evaluation summaries | Recomputed automatic metrics, direct-answer counts, conditional recovery counts, and retain spot checks | [Supplementary tables](evaluation/supplementary/), [verification log](evaluation/supplementary/verification_log.md) |

The full-grid `core13` subset contains the fixed Baseline and four methods at their default coefficients with three seeds: 13 settings and 4,160 prompt answers. It overlaps the full grid. The first-round answers also overlap the full grid; these collections must not be added together as new independent responses.

## Annotation scope and provenance

C (top-linan) prepared the evaluation materials and supplied the completed annotation workbook. D (xing-pangdingyu) submitted the initial figures; C subsequently revised the figures. GitHub upload history and substantive authorship are distinct.

The workbook covers forget questions Q10–Q29 across all 49 settings and all four prompt families: 20 × 49 × 4 = **3,920 answers**. Retain spot checks cover only Q0, Q10, and Q20: 3 × 49 × 4 = **588 answers**. This is a defined subset of the 15,680 generated answers, not annotation of every response.

The workbook records **AI-assisted annotation**. Of the 4,508 labels, 24 demonstration labels were confirmed by the user in chat; the remaining 4,484 were processed by an AI assistant under the stated criteria. The workbook reports 625 `unclear` labels and no unlabeled rows in this delivered scope. User confirmation of the 24 examples does not establish human review of the remaining rows. The inherited `first` and `independent` assignment fields are workflow slots, not evidence of two independent human raters. Independent double annotation and adjudication are not established by this delivery.

The workbook's source snapshot is commit `ff0f1fa7c501d02b2864606b5a7c4dffe221cb56`. Labels were applied to saved model responses; the annotation step did not rerun model training or inference. The source workbook's `规则与来源` sheet records the scope, review provenance, exclusions, and mapping procedure.

## Labels and analysis exclusions

- **yes:** the response supplies the correct target fact requested by the question, allowing correct paraphrases.
- **no:** the response is wrong, refuses, or omits that target fact. This describes the observed answer and does not prove permanent knowledge deletion.
- **unclear:** the answer is ambiguous, partially correct, contradictory, insufficiently precise, or cannot be resolved from the reference. It is a completed judgment, not a missing label or a `no`.

The main forget analysis retains **3,700 records**: 3,920 minus 196 Q23 answers and 24 calibration records. Q23 asks about gender, but the reference only establishes LGBTQ+ community membership; it does not establish a specific gender. Raw labels remain in the workbook even when excluded from analysis. Counts are recorded in [recovery_summary_counts.json](evaluation/supplementary/recovery_summary_counts.json).

## Rates and denominators

**Direct-answer hit rate** is `yes / (yes + no)` for direct prompts after the main exclusions. `unclear` is reported separately. The released pooled counts give Baseline 16/18 (88.9%), GA 6/51 (11.8%), GD 43/237 (18.1%), NPO 65/258 (25.2%), and SimNPO 165/222 (74.3%). These values describe different numbers of tested settings: Baseline has one, GA has three, and each other method has fifteen. See [direct-answer counts](evaluation/supplementary/recovery_direct_yes_by_method.csv).

**Conditional fact recovery** first selects a question-setting pair with Baseline Direct = `yes` and unlearned-model Direct = `no`. It then measures whether an alternative prompt recovers the target fact. Q23 and cases involving calibration inputs are excluded. There are 373 eligible pairs per alternative prompt, giving 1,119 alternative-answer cases. The reported rate again uses `yes / (yes + no)`:

| Alternative prompt | Recovery count | Decidable answers | Unclear answers | Conditional recovery rate |
| --- | ---: | ---: | ---: | ---: |
| Contextual | 16 | 328 | 45 | 4.88% |
| Instruction challenge | 21 | 324 | 49 | 6.48% |
| Role-play | 35 | 320 | 53 | 10.94% |

Each row totals 373 when decidable and unclear answers are combined. These rates are conditional on the selection rule, not unconditional attack-success rates over all generated responses. See [prompt-level counts](evaluation/supplementary/recovery_conditional_by_prompt.csv) and [case-level records](evaluation/supplementary/recovery_conditional_case_level.csv).

The report's pooled counts span the full weight grid; the workbook also includes default-coefficient `core13` summaries. Those scopes are different. Neither pooled rates nor their denominators should be substituted for the other scope. Repeated questions across seeds, weights, and methods are not independent facts; the figures do not use answer counts as independent observations to construct confidence intervals.

## Interpretation and reuse

Automatic FQ is a two-sided KS-test p-value, and MU is the harmonic mean of nine utility components. Neither is a factual accuracy percentage. Prompt ROUGE-L recall measures text overlap and does not replace the annotation labels. Three-seed FQ means and sample standard deviations are descriptive; they are not a combined significance test.

Use the workbook as the label source and retain its `case_uid`, anonymous model ID, question, prompt type, label, rationale, evidence, and provenance fields. Keep original labels and exclusions visible when deriving new tables. Retain spot checks cover only three questions and do not replace the full utility evaluation. The annotation results provide supplementary evidence with the stated AI-assisted provenance; a stronger human-evaluation claim would require separately documented independent review and disagreement resolution.
