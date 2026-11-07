# D 指标核验记录
- all-49-settings.csv 行数: 49（期望 49）
- 方法×alpha×seed 组合齐全: {('GradAscent', 1.0): 3, ('GradDiff', 0.25): 3, ('GradDiff', 0.5): 3, ('GradDiff', 1.0): 3, ('GradDiff', 2.0): 3, ('GradDiff', 4.0): 3, ('NPO', 0.25): 3, ('NPO', 0.5): 3, ('NPO', 1.0): 3, ('NPO', 2.0): 3, ('NPO', 4.0): 3, ('SimNPO', 0.25): 3, ('SimNPO', 0.5): 3, ('SimNPO', 1.0): 3, ('SimNPO', 2.0): 3, ('SimNPO', 4.0): 3, ('baseline', 1.0): 1}
- MU 与九项调和平均复核: 通过（最大绝对差 2.220e-16）
- 与 three-seed-summary.csv 比对差异条数: 0
  - 无差异（在 1e-9 容差内）
- 与 RESULTS.md 三种子表比对差异条数: 0
  - 无差异（RESULTS.md 表内数字与从 CSV 重算一致，舍入到表内位数）

## 指标定义（报告用）
- **FQ**：遗忘集分数分布相对「只练保留集的 Oracle」做双侧 KS 检验得到的 **p 值**。不是遗忘准确率，也不是遗忘百分比。FQ 较高 ≈ 更难用该检验把遗忘后模型与 Oracle 区分开（常解读为分布层面变化证据较弱）。三种子 FQ 均值不是合并显著性检验。
- **MU**：九项**非遗忘**能力分量的**调和平均**（保留集 / 真实作者 / 世界事实 × Prob / ROUGE / Truth Ratio）。不是准确率。
