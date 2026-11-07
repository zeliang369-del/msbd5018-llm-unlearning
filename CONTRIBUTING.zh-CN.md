# 协作与维护

[English](CONTRIBUTING.md) | **简体中文**

先阅读[研究报告](docs/RESEARCH_REPORT.zh-CN.md)、[复现指南](docs/REPRODUCIBILITY.zh-CN.md)和[贡献记录](docs/TEAM_PROGRESS.zh-CN.md)。仓库描述已经交付的研究，早期提案与任务排期不再作为当前规范。

1. 从 `main` 创建主题分支，在 PR 中说明问题、改动、验证与局限。
2. 保留冻结实验输出、源码快照、哈希与原始标注来源。新设置或源码须使用新运行编号，不覆盖已完成或失败的运行。
3. 在一次性检出副本中运行 `python3 verify_results.py`、`python3 verify_phase3_delivery.py` 和 `python3 verify_phase3_grid_delivery.py --require-visual`。这些命令审计已有证据，可能重写审计 JSON；提交前检查差异。
4. 主要文档的中英文保持一致并互相链接；翻译不得改变指标数值、排除规则与图表范围。
5. 按现有仓库约定，组员复核为可选，必要自动检查须通过。撤销合并时使用 Git revert，不重写共享历史。

记录模型／数据／源码身份、种子、系数、采样、更新预算、资源、失败、检查点重载与评价范围。不要把导入 Oracle 称为本地重训、缓存结果称为新实验，或 AI 辅助标签称为独立人工裁决。新增事实标签须保留案例身份、审阅来源、理由、排除项与分歧。

模型权重、运行环境、缓存与凭证不进入 Git。遵循[来源说明](THIRD_PARTY.md)中模型、数据、框架各自的条款。作者归属与上传账号分开记录。
