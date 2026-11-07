# Evaluation Metrics and Experimental Results

[English](evaluation_metrics_and_results.md) | [简体中文](evaluation_metrics_and_results.zh-CN.md)

Evaluation materials were prepared by C (`top-linan`). D (`xing-pangdingyu`) contributed the initial figures and submitted the materials through PR #9; C subsequently revised and finalized the figures in PR #11. Repository submission history and material authorship are distinguished in the [contribution record](../../TEAM_PROGRESS.md).

## Evaluation metrics

This study evaluates machine unlearning with Llama-3.2-1B on the TOFU benchmark using the OpenUnlearning evaluation framework. Evaluation includes automatic metrics and factual annotations of generated responses. The annotation analysis covers forget-set questions Q10–Q29, with spot checks on retain-set questions Q0, Q10, and Q20.

**Forget Quality (FQ)** is the two-sided Kolmogorov–Smirnov test \(p\)-value obtained by comparing the unlearned model's forget-set score distribution with the corresponding distribution of an Oracle model trained only on retain data. FQ measures distinguishability under this particular test: a larger \(p\)-value indicates weaker evidence against the null hypothesis that the distributions are the same. For each hyperparameter setting, we report the mean and sample standard deviation across three random seeds to describe variation between runs. These summaries do not constitute a combined significance test.

**Model Utility (MU)** is the harmonic mean of nine utility components. The components cover three task domains—Retain, Real Authors, and World Facts—with answer probability, ROUGE-L recall, and truth ratio measured in each domain. MU summarizes preservation of capabilities outside the targeted forget knowledge.

## Experimental setup

We compare gradient ascent (GradAscent, GA), gradient difference (GradDiff, GD), NPO, and SimNPO against a baseline model without unlearning. Baseline is evaluated once. GA is trained once for each of three random seeds. GD, NPO, and SimNPO are each trained with three random seeds at every retain-loss weight \(\alpha \in \{0.25, 0.5, 1, 2, 4\}\). This yields 48 formal training runs and one baseline evaluation: 49 configurations, each evaluated on all 657 questions.

The weight sweep varies only the retain-loss coefficient \(\alpha\); other method coefficients remain fixed as declared in advance. GD and NPO use \(\gamma = 1\); SimNPO uses \(\gamma = 0.125\); NPO uses \(\beta = 0.1\); and SimNPO uses \(\beta = 4.5\), \(\delta = 0\). Objective scales differ between methods, and the comparison does not match methods at an equal achieved level of forgetting. Cross-method results must therefore be interpreted within this experimental grid.

## Automatic evaluation results

Table 1 reports the three-seed means and sample standard deviations of FQ and MU at each \(\alpha\). The unchanged Baseline has FQ = 0.014 and MU = 0.599. SimNPO maintains relatively high MU across all \(\alpha\) values (0.590–0.592), close to Baseline, but consistently low FQ (0.006–0.046). NPO obtains high FQ at most weights: for \(\alpha \le 1\), its three-seed mean is 0.919. Its MU ranges from 0.556 to 0.577, below Baseline and SimNPO but above GD at low \(\alpha\) and GA (MU = 0.367). GD's MU increases with \(\alpha\), from 0.253 at \(\alpha = 0.25\) to 0.566 at \(\alpha = 4\); at \(\alpha \ge 2\), it exceeds GA's MU. Meanwhile, the sample standard deviation of FQ exceeds 0.1 for several GD weights and for NPO at higher weights, indicating sensitivity to the random seed in those configurations.

**Table 1.** FQ and MU at different retain-loss weights (three-seed mean ± sample standard deviation; Baseline is evaluated once).

| Method | \(\alpha\) | \(n\) | FQ | MU |
|---|---:|---:|---:|---:|
| GradAscent | 1 | 3 | 0.463 ± 0.100 | 0.367 ± 0.044 |
| GradDiff | 0.25 | 3 | 0.432 ± 0.289 | 0.253 ± 0.026 |
| GradDiff | 0.5 | 3 | 0.303 ± 0.239 | 0.256 ± 0.007 |
| GradDiff | 1 | 3 | 0.641 ± 0.108 | 0.379 ± 0.011 |
| GradDiff | 2 | 3 | 0.303 ± 0.239 | 0.520 ± 0.008 |
| GradDiff | 4 | 3 | 0.083 ± 0.025 | 0.566 ± 0.001 |
| NPO | 0.25 | 3 | 0.919 ± 0.000 | 0.556 ± 0.001 |
| NPO | 0.5 | 3 | 0.919 ± 0.000 | 0.562 ± 0.001 |
| NPO | 1 | 3 | 0.919 ± 0.000 | 0.567 ± 0.001 |
| NPO | 2 | 3 | 0.892 ± 0.114 | 0.573 ± 0.003 |
| NPO | 4 | 3 | 0.754 ± 0.170 | 0.577 ± 0.003 |
| SimNPO | 0.25 | 3 | 0.046 ± 0.015 | 0.591 ± 0.003 |
| SimNPO | 0.5 | 3 | 0.029 ± 0.000 | 0.590 ± 0.001 |
| SimNPO | 1 | 3 | 0.006 ± 0.002 | 0.592 ± 0.003 |
| SimNPO | 2 | 3 | 0.007 ± 0.000 | 0.591 ± 0.001 |
| SimNPO | 4 | 3 | 0.007 ± 0.000 | 0.592 ± 0.002 |
| Baseline | 1 | 1 | 0.014 | 0.599 |

The value 1 in the GA and Baseline rows follows the source table's configuration notation. GA has no retain-loss term, and Baseline is fixed; neither participates in the \(\alpha\) sweep.

Figure 1 shows all 49 configurations in the MU–FQ plane and the three-seed means for each method and weight. NPO lies mainly in the high-FQ region; SimNPO clusters at high MU and low FQ; GD moves toward higher utility as \(\alpha\) increases; and GA has intermediate FQ and lower MU. Figure 2 shows how FQ and MU change with \(\alpha\). Figure 3 highlights changes in the nine utility components relative to Baseline at the default settings, while Figure 4 shows their trends over the full weight grid. GD at low \(\alpha\) shows pronounced decreases in components such as retain-set probability and ROUGE, whereas SimNPO remains broadly close to Baseline across components.

![Figure 1](figures/fig1_fq_vs_mu_scatter.png)

**Figure 1. Forget quality and model utility.** Small translucent points represent individual runs; larger markers show three-seed means for each method and weight. Filled markers emphasize default settings, and open markers show other weights. The panels on the right enlarge the dense NPO and SimNPO regions. Numeric labels denote retain-loss weight α; GA has no retain-loss weight. Baseline is one fixed evaluation. FQ is a KS-test p-value.

![Figure 2](figures/fig2_fq_and_mu_vs_alpha.png)

**Figure 2. Sensitivity to retain-loss weight.** Curves show three-seed means, error bars show sample standard deviations, and small points show individual seeds. Horizontal reference lines indicate the three-seed GA mean and the fixed Baseline; neither is included in the retain-loss weight sweep.

![Figure 3](figures/fig3_nine_utility_heatmap.png)

**Figure 3. Utility component changes relative to Baseline.** Each cell is the method's three-seed mean at the default setting minus the fixed Baseline component score. Signed differences are preserved, with a symmetric color scale centered at zero. GA uses its original objective; GD, NPO, and SimNPO use \(\alpha=1\). Absolute scores for all nine components across all 17 setting groups are shown in [Supplementary Figure S1](figures/figS1_full_utility_heatmap.png).

![Figure 4](figures/fig4_nine_utility_vs_alpha.png)

**Figure 4. Nine utility components across retain-loss weights.** Curves, error bars, and small points indicate three-seed means, sample standard deviations, and individual-seed results, respectively. Horizontal reference lines indicate the fixed Baseline. All panels use the same 0–1 vertical scale to support comparisons of the magnitude of component changes.

[Supplementary Figure S2](figures/figS2_prompt_overlap_heatmap.png) shows mean ROUGE-L recall across three seeds for all 17 setting groups and four prompt families, separately for forget questions and retain controls; Baseline is one fixed evaluation. Both panels share a 0–1 color scale. This metric measures text overlap and must not be interpreted as a factual recovery rate.

Supplementary Figures S1 and S2 use the same red–white–blue palette as Figure 3. Absolute scores in S1 and S2 use a fixed 0–1 scale: red indicates lower scores, white corresponds to 0.5, and blue indicates higher scores. Figure 3 centers the difference scale at zero, with red indicating decreases and blue indicating increases.

## Factual annotation results

Generated responses to forget-set questions Q10–Q29 are labeled yes, no, or unclear. The direct-prompt hit rate among determinate labels is defined as yes divided by yes + no. As shown in Table 2, Baseline has a determinate hit rate of 0.889. SimNPO has a rate of 0.743, substantially higher than the other unlearning methods; NPO, GD, and GA have rates of 0.252, 0.181, and 0.118, respectively.

Table 2 and Figures 5 and 6 use the primary analysis scope, excluding question 23 and 24 calibration records. Unclear labels are reported separately and excluded from determinate denominators.

**Table 2.** Direct-prompt judgments and determinate hit rates for forget-set questions Q10–Q29.

| Method | yes | no | unclear | Determinate hit rate |
|---|---:|---:|---:|---:|
| GradAscent | 6 | 45 | 6 | 0.118 |
| GradDiff | 43 | 194 | 44 | 0.181 |
| NPO | 65 | 193 | 25 | 0.252 |
| SimNPO | 165 | 57 | 63 | 0.743 |
| Baseline | 16 | 2 | 1 | 0.889 |

To examine whether a different prompt form can elicit a target fact again, we define **conditional recovery rate** as follows. Select questions for which Baseline receives a yes label under direct prompting and the unlearned model receives a no label under direct prompting. Then calculate the proportion labeled yes under an alternative prompt. The alternative families are contextual, role-play, and instruction challenge. The denominator is the number of determinate alternative-prompt responses (yes + no), with unclear reported separately. Question 23 is excluded because its reference answer does not specify gender.

As shown in Table 3, the pooled conditional recovery rates are 0.049 (16/328) for contextual prompts, 0.065 (21/324) for instruction challenge, and 0.109 (35/320) for role-play. Table 4 breaks these results down by method and prompt family. Role-play has the highest pooled rate. No determinate recovery is observed for GA under contextual or instruction-challenge prompts. Some cells contain few determinate observations, so these proportions are descriptive. Figure 5 shows conditional recovery rates by alternative prompt and method; Figure 6 shows the direct-prompt determinate hit rate as a function of retain-loss weight.

![Figure 5](figures/fig5_conditional_fact_recovery.png)

**Figure 5. Conditional recovery of target facts.** Each panel corresponds to one alternative prompt family. Point positions show yes/(yes+no); labels report recovered responses, determinate responses, and unclear counts (u). GA pools three seed settings, while each other method pools 15 settings spanning five weights and three seeds. Consequently, aggregation scope and conditional denominators may differ between methods. Labels are primarily AI-assisted. Questions recur across configurations; these proportions are descriptive, and response counts are not treated as independent factual samples for confidence intervals.

![Figure 6](figures/fig6_direct_hit_rate_vs_alpha.png)

**Figure 6. Direct-prompt target-fact hit rates.** The three panels show GD, NPO, and SimNPO. Curves are yes/(yes+no) computed by pooling counts across the three seeds at each weight. Small points show individual-seed rates, and labels show pooled numerators and denominators. Horizontal reference lines show the corresponding Baseline and GA rates. The curves are not arithmetic means of seed-level rates. Labels are primarily AI-assisted, and unclear labels are excluded from determinate denominators.

**Table 3.** Conditional recovery rates pooled by alternative prompt family.

| Alternative prompt | Recovered as yes | Determinate responses | Conditional recovery rate | unclear |
|---|---:|---:|---:|---:|
| Contextual | 16 | 328 | 0.049 | 45 |
| Instruction challenge | 21 | 324 | 0.065 | 49 |
| Role-play | 35 | 320 | 0.109 | 53 |

**Table 4.** Conditional recovery rates by method and alternative prompt family.

| Method | Alternative prompt | Recovered as yes | Determinate responses | Conditional recovery rate | unclear |
|---|---|---:|---:|---:|---:|
| GradAscent | Contextual | 0 | 30 | 0.000 | 7 |
| GradAscent | Instruction challenge | 0 | 36 | 0.000 | 1 |
| GradAscent | Role-play | 2 | 34 | 0.059 | 3 |
| GradDiff | Contextual | 6 | 136 | 0.044 | 16 |
| GradDiff | Instruction challenge | 6 | 132 | 0.045 | 20 |
| GradDiff | Role-play | 17 | 131 | 0.130 | 21 |
| NPO | Contextual | 8 | 134 | 0.060 | 19 |
| NPO | Instruction challenge | 14 | 130 | 0.108 | 23 |
| NPO | Role-play | 13 | 128 | 0.102 | 25 |
| SimNPO | Contextual | 2 | 28 | 0.071 | 3 |
| SimNPO | Instruction challenge | 1 | 26 | 0.038 | 5 |
| SimNPO | Role-play | 3 | 27 | 0.111 | 4 |

Table 5 reports direct-prompt determinate hit rates for the retain-set spot checks (Q0, Q10, and Q20). These checks cover only three retain questions and provide supplementary observations; they do not replace the broader utility evaluation summarized by MU.

**Table 5.** Direct-prompt judgments for retain-set spot checks (Q0, Q10, and Q20).

| Method | yes | no | unclear | Determinate hit rate |
|---|---:|---:|---:|---:|
| GradAscent | 5 | 4 | 0 | 0.556 |
| GradDiff | 18 | 24 | 3 | 0.429 |
| NPO | 30 | 15 | 0 | 0.667 |
| SimNPO | 35 | 10 | 0 | 0.778 |
| Baseline | 2 | 1 | 0 | 0.667 |

These annotations are primarily AI-assisted and underwent pilot-label checks. They should be treated as supplementary evidence alongside the automatic metrics, rather than as evidence of completed independent double annotation and adjudication.

## Discussion

The automatic evaluations and factual annotations support four observations within the scope of this study. First, SimNPO preserves utility closest to the unchanged Baseline but has low FQ, indicating that its forget-set score distribution remains distinguishable from the Oracle's under the specified test. Second, NPO achieves high FQ in this experiment's distributional test while limiting the reduction in utility. Third, GD's utility is sensitive to retain-loss weight: increasing \(\alpha\) can raise MU above GA, whereas very small \(\alpha\) substantially impairs utility. FQ varies across seeds for some configurations, so both means and standard deviations should be reported. Fourth, among cases where the unlearned model fails to state a target fact under direct prompting that Baseline states correctly, an alternative prompt can sometimes elicit the fact again. This provides evidence of residual information at the level of prompt variation, complementing the automatic metrics.
