# Metric verification log

**English** | [简体中文](verification_log.zh-CN.md)

This is the translated verification record delivered with the evaluation materials. Its original heading was “D 指标核验记录” (“D metric verification record”). That historical heading identifies the delivered record; substantive authorship follows the project's [contribution statement](../../TEAM_PROGRESS.md), rather than being inferred from the heading or uploading account. The figures below preserve the recorded checks and are not a claim of a new verification run.

- Rows in `all-49-settings.csv`: **49** (expected: 49).
- Method × alpha × seed coverage is complete:

  | Method | Alpha | Number of settings |
  | --- | ---: | ---: |
  | GradAscent | 1.0 | 3 |
  | GradDiff | 0.25 | 3 |
  | GradDiff | 0.5 | 3 |
  | GradDiff | 1.0 | 3 |
  | GradDiff | 2.0 | 3 |
  | GradDiff | 4.0 | 3 |
  | NPO | 0.25 | 3 |
  | NPO | 0.5 | 3 |
  | NPO | 1.0 | 3 |
  | NPO | 2.0 | 3 |
  | NPO | 4.0 | 3 |
  | SimNPO | 0.25 | 3 |
  | SimNPO | 0.5 | 3 |
  | SimNPO | 1.0 | 3 |
  | SimNPO | 2.0 | 3 |
  | SimNPO | 4.0 | 3 |
  | baseline | 1.0 | 1 |

- MU versus the harmonic mean of its nine components: **passed**, maximum absolute difference **2.220e-16**.
- Differences from `three-seed-summary.csv`: **0**, within a **1e-9** tolerance.
- Differences from the three-seed table in `RESULTS.md`: **0**. The reported values match recomputation from the CSV after rounding to the displayed precision.

## Metric definitions

- **FQ:** the **p-value** from a two-sided KS test comparing the forget-set score distribution with that of the retain-only Oracle. It is neither forgetting accuracy nor a forgetting percentage. A higher p-value provides less evidence against the null hypothesis that the distributions are the same; it does not establish distributional equivalence. A mean FQ across three seeds is not a combined significance test.
- **MU:** the **harmonic mean** of nine non-forget utility components: Retain, Real Authors, and World Facts, each evaluated by answer probability, ROUGE, and truth ratio. It is not accuracy.
