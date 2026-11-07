# 评估材料与图表

[English](README.md) | **简体中文**

C（top-linan）负责评估材料与最终图表完善；D（xing-pangdingyu）提供初版图表并提交了评估包。以下链接均指向最新版。

- [完整评估章节：五张表与全部图注](report/evaluation_metrics_and_results.zh-CN.md)
- [标注范围、来源、分母与排除项](../DATA_AND_ANNOTATIONS.zh-CN.md)
- [标签判定规则](supplementary/annotation_coding.zh-CN.md)
- [核验日志](supplementary/verification_log.zh-CN.md)
- [原始工作簿与派生统计表](supplementary/)
- [六张主图与两张补充图](report/figures/)

图表标签为英文；报告和图注提供中英文版本。每张图包含 300 dpi PNG、矢量 PDF 与 SVG。重建时使用独立绘图环境，避免改变训练环境或覆盖已发布图片：

```sh
python -m pip install -r scripts/figures/requirements.txt
python scripts/plot_evaluation.py --output-dir /tmp/msbd5018-figures
```

命令在仓库根目录运行。FQ 为描述性 KS p 值汇总，误差棒为样本标准差；标注比例使用合并的 yes/(yes+no)，unclear 单列。重复问题不视为独立样本，图中未据此构造置信区间。补充图 S2 为文本重合代理，不是事实恢复率。图 3 为相对基线的有符号差值；S1、S2 为 0–1 绝对分数。
