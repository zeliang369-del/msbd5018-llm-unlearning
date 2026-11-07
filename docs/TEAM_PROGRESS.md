# Contributions and delivery provenance

**English** | [简体中文](TEAM_PROGRESS.zh-CN.md) · [Project home](../README.md)

Updated 8 November 2026. All four members contribute as peers. Research authorship and the account that uploads a file are recorded separately.

| Member | GitHub account | Substantive contribution | Evidence |
|---|---|---|---|
| A | zeliang369-del | Initial reproduction, proposal, CUDA infrastructure and repairs, experiment execution, verification, grid completion and repository integration | Initial history; PRs #2, #5–#8; frozen run records |
| B | caojie1126-loryhbox (commit name LORYHBOX) | SimNPO/NPO loss implementations, training adapters, stable sample identity, reference-NLL cache and method tests | PRs #3 and #4; original commits retained in integration ancestry |
| C | top-linan | Evaluation materials, completed annotation workbook and analysis, revision of the final research figures, reproducible PNG/PDF/SVG exports | Evaluation report and workbook; PRs #10 and #11; clarified team authorship |
| D | xing-pangdingyu | Initial figure preparation and submission of the evaluation package | PR #9; figures subsequently revised by C |

## Authorship and revision history

The evaluation package entered GitHub through D's PR #9. This upload does not make D the author of all its contents: **C authored the evaluation materials**. D prepared the initial figures, and C subsequently revised and finalized their presentation in PR #11. The repository uses the latest figures and retains the earlier versions in Git history. The historical verification log's original “D” heading is artifact metadata, not an override of this authorship clarification.

B's original method commits remain in the graph; A's later integration, cache/data-path fixes, checkpoint-persistence repair and GPU execution are distinct contributions. Commit counts and merge operations are not used as measures of individual research effort.

## Completed delivery

- Four unlearning methods plus a fixed baseline; 48 formal training runs, 49 complete evaluations, and 15,680 full-grid prompt answers.
- A completed 4,508-label workbook in C's defined scope, with explicit AI-assisted provenance, exclusions and reasons.
- Evaluation text, statistical tables, six main figures, two supplementary figures, plotting sources and vector exports.
- Pinned sources, raw outputs, source/checkpoint hashes and saved-record verification tools.

The [annotation guide](DATA_AND_ANNOTATIONS.md) distinguishes delivered labels from independent double-human annotation and adjudication, which are not established by the current evidence. This page records delivered work; obsolete role-claiming tables and execution schedules have been removed from the current documentation.

## Pull-request record

[CUDA results #2](https://github.com/zeliang369-del/msbd5018-llm-unlearning/pull/2) · [SimNPO #3](https://github.com/zeliang369-del/msbd5018-llm-unlearning/pull/3) · [NPO #4](https://github.com/zeliang369-del/msbd5018-llm-unlearning/pull/4) · [Integration #5](https://github.com/zeliang369-del/msbd5018-llm-unlearning/pull/5) · [Coverage #6](https://github.com/zeliang369-del/msbd5018-llm-unlearning/pull/6) · [Checkpoint repair #7](https://github.com/zeliang369-del/msbd5018-llm-unlearning/pull/7) · [Complete grid #8](https://github.com/zeliang369-del/msbd5018-llm-unlearning/pull/8) · [Evaluation upload #9](https://github.com/zeliang369-del/msbd5018-llm-unlearning/pull/9) · [Workbook #10](https://github.com/zeliang369-del/msbd5018-llm-unlearning/pull/10) · [Final figures #11](https://github.com/zeliang369-del/msbd5018-llm-unlearning/pull/11)
