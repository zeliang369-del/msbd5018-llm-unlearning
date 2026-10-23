"""Render verified phase-two results for local review and later GitHub sharing."""
from cuda_common import ROOT, read_json, write_json, file_hash
from run_phase2_cuda import DIRECTORY


def render():
    audit = read_json(DIRECTORY / "audit.json")
    assert audit["status"] == "passed" and audit["extensions_verified"] == 6
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180, "figure.dpi": 120})
    runs = audit["runs"]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for column, (metric, title) in enumerate((("forget_quality", "FQ: KS-test p-value"), ("model_utility", "Model utility (MU)"))):
        axis = axes[0, column]
        weights = [.5, 1., 2.]
        ids = ["gd-a0p5-s0-cuda-v1", "gd-a1-s0-cuda-v1", "gd-a2-s0-cuda-v1"]
        values = [runs[name]["metrics"][metric] for name in ids]
        axis.plot(weights, values, "o--", color="#D55E00", linewidth=1.2)
        for x, y in zip(weights, values):
            axis.annotate(f"{y:.4f}", (x, y), xytext=(0, 8), textcoords="offset points", ha="center")
        axis.set(title="GD retain-weight comparison: seed 0", xlabel="Retain weight alpha (gamma = 1)", ylabel=title,
                 xticks=weights, ylim=(0, 1.06))
        axis.grid(axis="y", alpha=.2)
        axis = axes[1, column]
        for seed, color in enumerate(("#0072B2", "#009E73", "#CC79A7")):
            values = [runs[f"ga-s{seed}-cuda-v1"]["metrics"][metric], runs[f"gd-a1-s{seed}-cuda-v1"]["metrics"][metric]]
            axis.plot([0, 1], values, "o-", color=color, alpha=.8, linewidth=1.1, label=f"Seed {seed}")
        for x, method in enumerate(("GradAscent", "GradDiff")):
            summary = next(row for row in audit["seed_summary"] if row["method"] == method and row["metric"] == metric)
            axis.errorbar(x + .08, summary["mean"], yerr=summary["sample_sd"], fmt="s", color="black", capsize=4, markersize=4)
        axis.set(title="Matched-seed GA/GD comparison (n = 3)", ylabel=title, xticks=[0, 1],
                 xticklabels=["GA", "GD (alpha = gamma = 1)"], xlim=(-.25, 1.35), ylim=(0, 1.06))
        axis.grid(axis="y", alpha=.2)
        handles, labels = axis.get_legend_handles_labels()
        axis.legend(handles + [Line2D([], [], color="black", marker="s", linestyle="none")],
                    labels + ["Mean +/- sample SD"], loc="upper right", fontsize=8)
    figure.suptitle("TOFU forget01 | full-parameter Llama-3.2-1B | CUDA", fontsize=14)
    figure.savefig(DIRECTORY / "phase2-results.png")
    figure.savefig(DIRECTORY / "phase2-results.pdf")
    plt.close(figure)
    lines = ["# Phase-two verified results / 第二阶段核验结果", "",
             "All six planned extensions and three CUDA controls passed independent CPU verification. Each complete evaluation has 657 questions; each training run has 10 updates and 320 sample visits.", "",
             "计划中的六组扩展和三组 CUDA 对照均通过独立 CPU 核验。每组完整评估覆盖 657 题，每组训练完成 10 次更新和 320 次样本访问。", "",
             "| Run / 实验 | FQ (KS p-value) | MU | Training seconds / 训练秒数 |", "|---|---:|---:|---:|"]
    for name, row in runs.items():
        training = f"{row['training_seconds']:.2f}" if row["training_seconds"] is not None else "—"
        lines.append(f"| {name} | {row['metrics']['forget_quality']:.6f} | {row['metrics']['model_utility']:.6f} | {training} |")
    lines += ["", "| Method / 方法 | Metric / 指标 | n | Mean / 均值 | Sample SD / 样本标准差 |",
              "|---|---|---:|---:|---:|"]
    for row in audit["seed_summary"]:
        if row["metric"] in ("forget_quality", "model_utility"):
            lines.append(f"| {row['method']} | {row['metric']} | 3 | {row['mean']:.6f} | {row['sample_sd']:.6f} |")
    ga_mu = next(row for row in audit["seed_summary"] if row["method"] == "GradAscent" and row["metric"] == "model_utility")
    gd_mu = next(row for row in audit["seed_summary"] if row["method"] == "GradDiff" and row["metric"] == "model_utility")
    lines += ["", f"Across these three seeds, GA MU is {ga_mu['mean']:.4f} +/- {ga_mu['sample_sd']:.4f} (sample SD), and GD MU is {gd_mu['mean']:.4f} +/- {gd_mu['sample_sd']:.4f}. GA has higher MU at seed 0; GD has higher MU at seeds 1 and 2. The observed average favors GD slightly, with less variation, but the ordering depends on seed and three runs do not establish a general advantage.", "",
              f"这三个种子中，GA 的 MU 为 {ga_mu['mean']:.4f} ± {ga_mu['sample_sd']:.4f}，GD 为 {gd_mu['mean']:.4f} ± {gd_mu['sample_sd']:.4f}（样本标准差）。seed 0 下 GA 的 MU 更高，seed 1、2 下 GD 更高。观测均值略偏向 GD，且 GD 波动较小，但排序依赖种子，三次实验不足以证明普遍优势。", "",
              "At seed 0, increasing GD retain weight from 0.5 to 1 to 2 gives MU 0.2564, 0.3905, and 0.5235. FQ is 0.1650, 0.5786, and 0.1650 respectively. These observed points show that the metrics need not move together; the two equal FQ p-values do not imply identical checkpoints or answer distributions.", "",
              "在 seed 0 下，GD 保留权重从 0.5 增至 1、2 时，MU 分别为 0.2564、0.3905、0.5235，FQ 分别为 0.1650、0.5786、0.1650。这些观测点说明两项指标未必同步变化；两个相同的 FQ p 值不代表检查点或回答分布相同。", "",
              "The seed summaries include seeds 0, 1, and 2 on the Dell only. Weight comparisons fix seed 0 and gamma=1. Same-seed sampling matches across methods and weights, while the three seeds have distinct traces. All checkpoints are distinct and passed fresh-process reload checks.", "",
              "种子汇总仅使用戴尔上的 seed 0、1、2。权重比较固定 seed 0 和 gamma=1。同种子的方法和权重比较使用一致采样，三个种子的轨迹互不相同。各检查点均不同，并通过新进程重载检查。", "",
              "FQ is a KS-test p-value, not accuracy or a percentage forgotten. Its three-seed mean is descriptive, not a combined hypothesis test. Three seeds and three observed weight points do not establish a general ranking or a continuous trade-off curve. Mac runs remain separate. Prompt robustness and new methods are phase-three work.", "",
              "FQ 是 KS 检验的 p 值，不是准确率或遗忘百分比。三个种子的均值仅作描述，不是合并后的假设检验。三个种子和三个已观测权重点不足以建立普遍方法排名或连续权衡曲线。Mac 结果单独保留。提示鲁棒性与新方法属于第三阶段工作。", "",
              "![Verified phase-two results / 第二阶段核验结果](../study/phase2/phase2-results.png)", "",
              "Evidence: `study/phase2/audit.json`, `results.csv`, `seed-summary.csv`, `paired-seed-differences.csv`, and each run's configuration, exact source snapshot, complete raw evaluation, and CPU verification report. Figures are available as PNG and PDF.", "",
              "证据包括 `study/phase2/audit.json`、`results.csv`、`seed-summary.csv`、`paired-seed-differences.csv`，以及每组配置、精确源码快照、完整原始评价和 CPU 核验报告。图表提供 PNG 和 PDF。", "",
              "For the next team handoff: B reviews the method/loss settings and supplies SimNPO code; C freezes prompt tests and annotation rules; D reviews all seed/weight results and the nine utility components; A schedules resource checks and inference after those inputs are ready. Zeliang authorized GitHub publication; team notifications remain with Zeliang.", "",
              "下一步组内交接：B 复核方法与损失配置并提供 SimNPO 代码；C 冻结提示测试和标注规则；D 复核全部种子／权重结果及九项效用分量；相关输入就绪后，由 A 安排资源检查与推理。Zeliang 已授权发布到 GitHub，组员通知仍由 Zeliang 进行。", ""]
    (ROOT / "docs/PHASE2_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(DIRECTORY / "report-manifest.json", {"status": "rendered_from_verified_results",
               "audit_sha256": file_hash(DIRECTORY / "audit.json"),
               "renderer_sha256": file_hash(ROOT / "report_phase2_cuda.py"),
               "matplotlib_version": matplotlib.__version__,
               "requirements_report_sha256": file_hash(ROOT / "requirements-report.lock"),
               "files": {str(path.relative_to(ROOT)).replace("\\", "/"): file_hash(path) for path in
                         [ROOT / "docs/PHASE2_RESULTS.md", DIRECTORY / "phase2-results.png", DIRECTORY / "phase2-results.pdf"]}})
    print("Rendered verified phase-two report and figures")


if __name__ == "__main__":
    render()
