# 成员贡献与交付来源

[English](TEAM_PROGRESS.md) | **简体中文** · [项目首页](../README.zh-CN.md)

更新于 2026 年 11 月 8 日。四位成员平等协作；实际科研作者与上传文件的账号分开记录。

| 成员 | GitHub 账号 | 实际贡献 | 依据 |
|---|---|---|---|
| A | zeliang369-del | 初始复现、提案、CUDA 基础设施与修复、实验执行、验证、完整矩阵与仓库整合 | 初始记录、PR #2、#5–#8、冻结运行记录 |
| B | caojie1126-loryhbox（提交名 LORYHBOX） | SimNPO/NPO 损失实现、训练适配、稳定样本身份、参照 NLL 缓存和方法测试 | PR #3、#4；原始提交保留在合并历史中 |
| C | top-linan | 评估材料、完成的标注工作簿与分析、最终科研图表修订、可复现 PNG/PDF/SVG 导出 | 评估报告和工作簿、PR #10、#11、已澄清的组内作者归属 |
| D | xing-pangdingyu | 初版图表制作与评估包提交 | PR #9；图表随后由 C 修订 |

## 作者与修订关系

评估包通过 D 的 PR #9 进入 GitHub，这并不意味着全部内容均由 D 撰写：**评估材料由 C 完成**。D 制作初版图表，C 随后通过 PR #11 修订并完善最终呈现。仓库使用最新版图表，早期版本可在 Git 历史中追溯。历史核验日志原有的“D”标题属于文件元数据，不能覆盖这里澄清的实际作者归属。

B 的原始方法提交保留在历史中；A 后续的整合、缓存／数据路径修复、检查点保存修复和 GPU 执行为独立贡献。提交数和合并次数不作为个人科研工作量的衡量标准。

## 已完成交付

- 四种遗忘方法和固定基线；48 次正式训练、49 组完整评测、15,680 条完整矩阵提示回答。
- C 在明确范围内完成的 4,508 条标签工作簿，说明 AI 辅助来源、排除规则和理由。
- 评估文字、统计表、六张主图、两张补充图、绘图源码和矢量导出。
- 固定源码、原始输出、源码／检查点哈希与已存记录核验工具。

[标注指南](DATA_AND_ANNOTATIONS.zh-CN.md)区分已交付标签与双人独立人工标注和裁决；现有证据尚不支持后者已经完成。本页只记录实际交付，旧认领任务表和执行排期已从当前文档清理。

## PR 记录

[CUDA #2](https://github.com/zeliang369-del/msbd5018-llm-unlearning/pull/2) · [SimNPO #3](https://github.com/zeliang369-del/msbd5018-llm-unlearning/pull/3) · [NPO #4](https://github.com/zeliang369-del/msbd5018-llm-unlearning/pull/4) · [整合 #5](https://github.com/zeliang369-del/msbd5018-llm-unlearning/pull/5) · [覆盖说明 #6](https://github.com/zeliang369-del/msbd5018-llm-unlearning/pull/6) · [保存修复 #7](https://github.com/zeliang369-del/msbd5018-llm-unlearning/pull/7) · [完整矩阵 #8](https://github.com/zeliang369-del/msbd5018-llm-unlearning/pull/8) · [评估包提交 #9](https://github.com/zeliang369-del/msbd5018-llm-unlearning/pull/9) · [工作簿 #10](https://github.com/zeliang369-del/msbd5018-llm-unlearning/pull/10) · [最终图表 #11](https://github.com/zeliang369-del/msbd5018-llm-unlearning/pull/11)
