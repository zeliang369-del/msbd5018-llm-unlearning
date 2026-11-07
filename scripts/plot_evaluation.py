"""Rebuild publication figures from the committed evaluation tables (no inference)."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/evaluation/supplementary"
METHODS = ["GradAscent", "GradDiff", "NPO", "SimNPO"]
SWEEP = METHODS[1:]
ALPHAS = [0.25, 0.5, 1., 2., 4.]
LABEL = dict(zip(METHODS + ["baseline"], ["GA", "GD", "NPO", "SimNPO", "Baseline"]))
COLOR = dict(zip(METHODS + ["baseline"], ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#616975"]))
MARKER = dict(zip(METHODS + ["baseline"], ["D", "o", "s", "^", "X"]))
COMPONENTS = ["retain_Q_A_Prob", "retain_Q_A_ROUGE", "retain_Truth_Ratio",
              "ra_Q_A_Prob_normalised", "ra_Q_A_ROUGE", "ra_Truth_Ratio",
              "wf_Q_A_Prob_normalised", "wf_Q_A_ROUGE", "wf_Truth_Ratio"]
DOMAINS = ["Retain", "Real authors", "World facts"]
METRICS = ["Probability", "ROUGE-L", "Truth ratio"]
INK, MUTED, GRID = "#1F2933", "#586574", "#E6EAF0"
SCORE_CMAP = LinearSegmentedColormap.from_list("red_white_blue", ["#B84132", "#FAFAF8", "#2166AC"])


def read_csv(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(row, key):
    return float(row[key])


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "mathtext.fontset": "dejavusans",
        "axes.labelsize": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": "#89939F", "axes.linewidth": .65,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "xtick.major.width": .6, "ytick.major.width": .6,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "lines.linewidth": 1.6, "lines.markersize": 4.5, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "msbd5018-research-figures-v1", "axes.unicode_minus": True,
    })


def panel(ax, title, grid="y"):
    ax.set_title(title, loc="left", pad=9)
    ax.set_axisbelow(True)
    ax.grid(axis=grid, color=GRID, linewidth=.6)
    ax.tick_params(length=3, pad=3)


def alpha_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ALPHAS, ["0.25", "0.5", "1", "2", "4"])
    ax.set_xlim(.21, 4.8)
    ax.set_xlabel(r"Retain weight $\alpha$")
    ax.minorticks_off()


def title(fig, heading):
    fig.text(.08, .965, heading, fontsize=12, weight="bold", va="top")


def handles(methods, refs=False):
    return [Line2D([], [], color=COLOR[m], marker=None if refs and m in ["baseline", "GradAscent"] else MARKER[m], linestyle="--" if refs and m in ["baseline", "GradAscent"] else "-",
                   markersize=4, label=LABEL[m]) for m in methods]


def save(fig, out, name):
    for ext in ["png", "pdf", "svg"]:
        metadata = {"CreationDate": None, "ModDate": None} if ext == "pdf" else ({"Date": None} if ext == "svg" else {})
        path = out / f"{name}.{ext}"
        fig.savefig(path, dpi=300, metadata=metadata)
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)


def cell_text_color(im, value):
    """Use the higher-contrast text color against the actual cell background."""
    def luminance(rgb):
        rgb = np.asarray(rgb)
        linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
        return np.dot(linear, [.2126, .7152, .0722])

    background = luminance(im.cmap(im.norm(value))[:3])
    ink = luminance(matplotlib.colors.to_rgb(INK))
    dark_contrast = (background + .05) / (ink + .05)
    light_contrast = 1.05 / (background + .05)
    return INK if dark_contrast >= light_contrast else "white"


def load_data():
    rows = read_csv("table_all49_fq_mu_components.csv")
    groups = defaultdict(list)
    for r in rows:
        groups[r["method"], number(r, "alpha")].append(r)
    assert len(rows) == 49 and len(groups) == 17
    assert len({(r["method"], r["alpha"], r["seed"]) for r in rows}) == 49
    for (m, _), rs in groups.items():
        assert len(rs) == (1 if m == "baseline" else 3)
    for r in read_csv("table_fq_mu_mean_sd.csv"):
        for k in ["fq", "mu"]:
            values = [number(t, k) for t in groups[r["method"], number(r, "alpha")]]
            assert np.isclose(np.mean(values), number(r, k + "_mean"), rtol=0, atol=1e-10)
            if len(values) > 1:
                assert np.isclose(np.std(values, ddof=1), number(r, k + "_sample_sd"), rtol=0, atol=1e-10)
    direct = read_csv("recovery_direct_yes_by_method_alpha.csv")
    direct_seed = read_csv("recovery_direct_yes_by_setting.csv")
    for r in direct:
        selected = [t for t in direct_seed if t["method"] == r["method"] and number(t, "alpha") == number(r, "alpha")]
        for k in ["yes", "no", "unclear"]:
            assert sum(number(t, k) for t in selected) == number(r, k)
        assert np.isclose(number(r, "yes_rate_decidable"), number(r, "yes") / (number(r, "yes") + number(r, "no")))
    conditional = read_csv("recovery_conditional_by_method_prompt.csv")
    cases = read_csv("recovery_conditional_case_level.csv")
    for r in conditional:
        selected = [t for t in cases if t["method"] == r["method"] and t["替代提示"] == r["替代提示"]]
        for label, key in [("yes", "recover_yes"), ("no", "alt_no"), ("unclear", "alt_unclear")]:
            assert sum(t["判定"] == label for t in selected) == number(r, key)
        assert number(r, "decidable_denom") == number(r, "recover_yes") + number(r, "alt_no")
        assert np.isclose(number(r, "conditional_recover_rate"), number(r, "recover_yes") / number(r, "decidable_denom"))
    return rows, groups, direct, direct_seed, conditional


def avg(groups, method, alpha, key):
    return np.mean([number(r, key) for r in groups[method, alpha]])


def scatter_panel(ax, groups, methods, raw_size=13, mean_size=30):
    for m in methods:
        for (gm, a), rs in groups.items():
            if gm != m:
                continue
            if m != "baseline":
                ax.scatter([number(r, "mu") for r in rs], [number(r, "fq") for r in rs],
                           s=raw_size, marker=MARKER[m], color=COLOR[m], alpha=.34, linewidths=0, zorder=3)
            ax.scatter(avg(groups, m, a, "mu"), avg(groups, m, a, "fq"), s=mean_size,
                       marker=MARKER[m], facecolors=COLOR[m] if a == 1 else "white",
                       edgecolors=COLOR[m], linewidths=1.1, zorder=5)


def fig1(out, groups):
    fig = plt.figure(figsize=(7.6, 4.3))
    title(fig, "Forget Quality and Model Utility")
    ax = fig.add_axes([.08, .18, .53, .63])
    right_top = fig.add_axes([.73, .54, .245, .27])
    right_bottom = fig.add_axes([.73, .18, .245, .24])
    panel(ax, "a  All configurations", "both")
    scatter_panel(ax, groups, METHODS + ["baseline"])
    ax.axhline(.05, color="#A7ADB5", ls=(0, (3, 3)), lw=.8)
    ax.text(.205, .065, r"$p = 0.05$", color=MUTED, fontsize=7)
    ax.set(xlim=(.20, .62), ylim=(-.025, 1.05), xlabel="Model utility (MU)", ylabel=r"Forget quality (KS $p$-value)")
    for a, offset in zip(ALPHAS, [(0, 8), (0, -14), (0, 8), (-6, 8), (-8, 8)]):
        ax.annotate(f"{a:g}", (avg(groups, "GradDiff", a, "mu"), avg(groups, "GradDiff", a, "fq")),
                    xytext=offset, textcoords="offset points", fontsize=7, color=COLOR["GradDiff"])
    for z, method, label in [(right_top, "NPO", "b  NPO detail"), (right_bottom, "SimNPO", "c  SimNPO detail")]:
        panel(z, label, "both")
        scatter_panel(z, groups, [method], 12, 25)
        z.tick_params(labelsize=7)
        if method == "NPO":
            z.set(xlim=(.552, .584), ylim=(.54, 1.04))
            offsets = [(-9, 8), (-5, -14), (3, 8), (8, 0), (-8, 7)]
        else:
            z.set(xlim=(.5878, .5964), ylim=(-.003, .061), xlabel="MU")
            z.set_yticks([0, .02, .04, .06])
            offsets = [(-20, 7), (-20, 7), (17, -1), (-27, 10), (17, 15)]
        for a, offset in zip(ALPHAS, offsets):
            z.annotate(f"{a:g}", (avg(groups, method, a, "mu"), avg(groups, method, a, "fq")),
                       xytext=offset, textcoords="offset points", fontsize=6.8, color=COLOR[method],
                       arrowprops=dict(arrowstyle="-", color=COLOR[method], lw=.45, alpha=.7))
    fig.legend(handles=handles(METHODS + ["baseline"]), loc="upper left", bbox_to_anchor=(.075, .915), ncol=5, handlelength=1.5, columnspacing=1.6)
    save(fig, out, "fig1_fq_vs_mu_scatter")


def fig2(out, groups):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5))
    fig.subplots_adjust(left=.08, right=.975, bottom=.17, top=.76, wspace=.28)
    title(fig, "Retain-Weight Sensitivity Analysis")
    for ax, key, label in zip(axes, ["fq", "mu"], [r"a  Forget quality (KS $p$-value)", "b  Model utility"]):
        panel(ax, label)
        alpha_axis(ax)
        for m in SWEEP:
            values = np.array([[number(r, key) for r in groups[m, a]] for a in ALPHAS])
            for i, a in enumerate(ALPHAS):
                ax.scatter(a * np.array([.96, 1., 1.04]), values[i], s=9, color=COLOR[m], alpha=.35, linewidths=0, zorder=3)
            ax.errorbar(ALPHAS, values.mean(axis=1), yerr=values.std(axis=1, ddof=1),
                        color=COLOR[m], marker=MARKER[m], capsize=2.5, elinewidth=.85, zorder=4)
        for m in ["GradAscent", "baseline"]:
            ax.axhline(avg(groups, m, 1., key), color=COLOR[m], ls="--", lw=.95)
        if key == "fq":
            ax.set_ylim(-.025, 1.065)
        else:
            ax.set_ylim(.205, .625)
    fig.legend(handles=handles(SWEEP + ["GradAscent", "baseline"], refs=True), ncol=5,
               loc="upper left", bbox_to_anchor=(.075, .895), handlelength=1.8, columnspacing=1.5)
    save(fig, out, "fig2_fq_and_mu_vs_alpha")


def fig3(out, groups):
    base = np.array([avg(groups, "baseline", 1., k) for k in COMPONENTS])
    values = np.array([[avg(groups, m, 1., k) for k in COMPONENTS] for m in METHODS]) - base
    limit = np.ceil(np.abs(values).max() * 20) / 20
    fig = plt.figure(figsize=(7.6, 3.1))
    title(fig, "Utility Component Changes Relative to Baseline")
    base_ax = fig.add_axes([.12, .74, .735, .09])
    base_ax.imshow(np.zeros((1, 9)), cmap=LinearSegmentedColormap.from_list("neutral", ["#EDF0F4", "#EDF0F4"]), aspect="auto")
    base_ax.set_yticks([0], ["Baseline"])
    base_ax.set_xticks([])
    base_ax.tick_params(axis="y", length=0)
    for x, val in enumerate(base):
        base_ax.text(x, 0, f"{val:.3f}", ha="center", va="center", fontsize=7.5)
    for i, domain in enumerate(DOMAINS):
        base_ax.text(1 + 3 * i, -.83, domain, ha="center", va="bottom", fontsize=8.5, weight="bold")
    ax = fig.add_axes([.12, .15, .735, .54])
    im = ax.imshow(values, cmap=SCORE_CMAP, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_yticks(range(4), [LABEL[m] for m in METHODS])
    ax.set_xticks(range(9), ["Prob.", "ROUGE-L", "TR"] * 3, fontsize=8)
    ax.tick_params(length=0, pad=6)
    for tick, m in zip(ax.get_yticklabels(), METHODS):
        tick.set_color(COLOR[m]); tick.set_weight("bold")
    for y in range(4):
        for x in range(9):
            val = values[y, x]
            txt = "0.00" if abs(val) < .005 else f"{val:+.2f}"
            ax.text(x, y, txt, ha="center", va="center", fontsize=8,
                    color="white" if abs(val) > limit * .6 else INK)
    for a in [ax, base_ax]:
        for spine in a.spines.values(): spine.set_visible(False)
        for x in [2.5, 5.5]: a.axvline(x, color="white", lw=4)
    cax = fig.add_axes([.89, .15, .018, .54])
    cb = fig.colorbar(im, cax=cax, ticks=[-.75, -.5, -.25, 0, .25, .5, .75])
    cb.set_label(r"$\Delta$ score vs Baseline", fontsize=8)
    cb.outline.set_visible(False)
    save(fig, out, "fig3_nine_utility_heatmap")


def fig4(out, groups):
    fig, axes = plt.subplots(3, 3, figsize=(7.6, 6.2))
    fig.subplots_adjust(left=.085, right=.975, bottom=.09, top=.82, hspace=.46, wspace=.35)
    title(fig, "Utility Component Sensitivity to Retain Weight")
    for j, ax in enumerate(axes.flat):
        k = COMPONENTS[j]
        panel(ax, f"{chr(97+j)}  {DOMAINS[j//3]} · {['Prob.', 'ROUGE-L', 'TR'][j%3]}")
        alpha_axis(ax)
        for m in SWEEP:
            data = np.array([[number(r, k) for r in groups[m, a]] for a in ALPHAS])
            for i, a in enumerate(ALPHAS):
                ax.scatter(a * np.array([.96, 1., 1.04]), data[i], s=5, color=COLOR[m], alpha=.3, linewidths=0)
            ax.errorbar(ALPHAS, data.mean(axis=1), yerr=data.std(axis=1, ddof=1),
                        color=COLOR[m], marker=MARKER[m], capsize=1.5, elinewidth=.65,
                        markersize=3, linewidth=1.1)
        ax.axhline(avg(groups, "baseline", 1., k), color=COLOR["baseline"], ls="--", lw=.85)
        # Shared 0–1 score scale prevents tiny truth-ratio changes looking large.
        ax.set_ylim(0, 1)
        ax.set_yticks([0, .5, 1])
        if j < 6: ax.set_xlabel("")
        if j % 3 == 0: ax.set_ylabel("Component score")
        ax.tick_params(labelsize=7)
        ax.title.set_fontsize(8.2)
    fig.legend(handles=handles(SWEEP + ["baseline"], refs=True), ncol=4,
               loc="upper left", bbox_to_anchor=(.08, .915), handlelength=2, columnspacing=2)
    save(fig, out, "fig4_nine_utility_vs_alpha")


def fig5(out, rows):
    prompts = [("上下文", "Contextual"), ("角色扮演", "Role-play"), ("指令挑战", "Instruction challenge")]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.2), sharey=True)
    fig.subplots_adjust(left=.12, right=.98, bottom=.17, top=.81, wspace=.20)
    title(fig, "Conditional Target-Fact Recovery Rates")
    for j, (ax, (prompt, label)) in enumerate(zip(axes, prompts)):
        panel(ax, f"{chr(97+j)}  {label}", "x")
        for i, m in enumerate(METHODS):
            r = next(r for r in rows if r["method"] == m and r["替代提示"] == prompt)
            rate = number(r, "conditional_recover_rate")
            ax.hlines(i, 0, rate, color=COLOR[m], alpha=.22, lw=2)
            ax.scatter(rate, i, color=COLOR[m], marker=MARKER[m], s=35, zorder=4, clip_on=False)
            detail = f"{int(number(r, 'recover_yes'))}/{int(number(r, 'decidable_denom'))}   u={int(number(r, 'alt_unclear'))}"
            ax.annotate(detail, (rate, i), xytext=(5, 8), textcoords="offset points", fontsize=7.1, color=INK)
        ax.set_xlim(-.007, .205)
        ax.set_xticks([0, .05, .1, .15, .2])
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_xlabel("Conditional recovery rate", fontsize=8)
        ax.set_yticks(range(4), [LABEL[m] for m in METHODS])
        ax.set_ylim(3.55, -.65)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    save(fig, out, "fig5_conditional_fact_recovery")


def fig6(out, rows, seed_rows):
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 4.0), sharey=True)
    fig.subplots_adjust(left=.08, right=.975, bottom=.16, top=.75, wspace=.15)
    title(fig, "Target-Fact Hit Rates Under Direct Prompts")
    refs = {m: next(r for r in rows if r["method"] == m) for m in ["baseline", "GradAscent"]}
    for ax, m, letter in zip(axes, SWEEP, "abc"):
        panel(ax, f"{letter}  {LABEL[m]}")
        alpha_axis(ax)
        ys = []
        for a in ALPHAS:
            r = next(r for r in rows if r["method"] == m and number(r, "alpha") == a)
            selected = sorted([t for t in seed_rows if t["method"] == m and number(t, "alpha") == a], key=lambda t: int(t["seed"]))
            ax.scatter(a * np.array([.95, 1., 1.05]), [number(t, "yes_rate_decidable") for t in selected],
                       s=14, color=COLOR[m], alpha=.30, linewidths=0, zorder=3)
            rate = number(r, "yes_rate_decidable"); ys.append(rate)
            n = int(number(r, "yes") + number(r, "no"))
            # Alternate labels at nearby low rates to keep reference lines legible.
            offset = (0, 22 if rate < .10 else (9 if rate < .75 else -15))
            ax.annotate(f"{int(number(r, 'yes'))}/{n}", (a, rate), textcoords="offset points", xytext=offset,
                        ha="center", fontsize=7, color=COLOR[m], zorder=6)
        ax.plot(ALPHAS, ys, color=COLOR[m], marker=MARKER[m], zorder=4)
        for ref, ls in [("baseline", "--"), ("GradAscent", ":")]:
            ax.axhline(number(refs[ref], "yes_rate_decidable"), color=COLOR[ref], linestyle=ls, lw=1)
        ax.set_ylim(-.045, 1.08)
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[0].set_ylabel("Direct fact-hit rate")
    ref_handles = [Line2D([], [], color=COLOR[m], ls=ls, lw=1, label=f"{LABEL[m]}: {int(number(refs[m], 'yes'))}/{int(number(refs[m], 'yes') + number(refs[m], 'no'))}") for m, ls in [("baseline", "--"), ("GradAscent", ":")]]
    fig.legend(handles=ref_handles, ncol=2, loc="upper left", bbox_to_anchor=(.075, .875), handlelength=2.6)
    save(fig, out, "fig6_direct_hit_rate_vs_alpha")


def supplement(out, groups):
    order = [("baseline", 1.), ("GradAscent", 1.)] + [(m, a) for m in SWEEP for a in ALPHAS]
    values = np.array([[avg(groups, m, a, k) for k in COMPONENTS] for m, a in order])
    fig = plt.figure(figsize=(7.6, 7.3))
    title(fig, "Utility Component Scores Across Experimental Settings")
    ax = fig.add_axes([.18, .08, .66, .78])
    im = ax.imshow(values, vmin=0, vmax=1, cmap=SCORE_CMAP, aspect="auto")
    labels = [LABEL[m] if m in ["baseline", "GradAscent"] else f"{LABEL[m]}  α={a:g}" for m, a in order]
    ax.set_yticks(range(17), labels, fontsize=8)
    ax.set_xticks(range(9), ["Prob.", "ROUGE-L", "TR"] * 3, fontsize=8)
    ax.tick_params(length=0, pad=6)
    for i, domain in enumerate(DOMAINS):
        ax.text(i * 3 + 1, -1.05, domain, ha="center", fontsize=8.5, weight="bold")
    for y in range(17):
        for x in range(9):
            v = values[y, x]
            ax.text(x, y, f"{v:.2f}", ha="center", va="center", fontsize=7.2, color=cell_text_color(im, v))
    for y in [.5, 1.5, 6.5, 11.5]: ax.axhline(y, color="white", lw=2.5)
    for x in [2.5, 5.5]: ax.axvline(x, color="white", lw=3)
    for spine in ax.spines.values(): spine.set_visible(False)
    cb = fig.colorbar(im, cax=fig.add_axes([.88, .08, .02, .78]))
    cb.set_label("Component score", fontsize=8)
    cb.outline.set_visible(False)
    save(fig, out, "figS1_full_utility_heatmap")


def prompt_supplement(out):
    path = ROOT / "study/phase3/grid-20261029/prompt-three-seed-summary.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    families = ["direct", "contextual", "role_play", "instruction_challenge"]
    order = [("baseline", 1.), ("GradAscent", 1.)] + [(m, a) for m in SWEEP for a in ALPHAS]
    lookup = {(r["method"], number(r, "alpha"), r["domain"], r["family"]): r for r in rows}
    assert len(rows) == len(lookup) == 17 * 2 * 4
    assert all(int(r["n"]) == (1 if r["method"] == "baseline" else 3) for r in rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 7.0), sharey=True)
    fig.subplots_adjust(left=.18, right=.86, bottom=.09, top=.86, wspace=.13)
    title(fig, "ROUGE-L Recall Across Prompt Families")
    labels = [LABEL[m] if m in ["baseline", "GradAscent"] else f"{LABEL[m]}  α={a:g}" for m, a in order]
    for ax, domain, heading in zip(axes, ["forget", "retain"], ["a  Forget set", "b  Retain set"]):
        values = np.array([[number(lookup[m, a, domain, family], "rouge_proxy_mean") for family in families] for m, a in order])
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        im = ax.imshow(values, vmin=0, vmax=1, cmap=SCORE_CMAP, aspect="auto")
        ax.set_title(heading, loc="left", pad=10)
        ax.set_yticks(range(17), labels, fontsize=8)
        ax.set_xticks(range(4), ["Direct", "Context", "Role-play", "Instruction\nchallenge"], fontsize=7.5)
        ax.tick_params(length=0, pad=6)
        for y in range(17):
            for x in range(4):
                v = values[y, x]
                ax.text(x, y, f"{v:.2f}", ha="center", va="center", fontsize=7.4, color=cell_text_color(im, v))
        for y in [.5, 1.5, 6.5, 11.5]: ax.axhline(y, color="white", lw=2.5)
        for spine in ax.spines.values(): spine.set_visible(False)
    cb = fig.colorbar(im, cax=fig.add_axes([.90, .09, .018, .77]))
    cb.set_label("ROUGE-L recall proxy", fontsize=8)
    cb.outline.set_visible(False)
    save(fig, out, "figS2_prompt_overlap_heatmap")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/evaluation/report/figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()
    rows, groups, direct, direct_seed, conditional = load_data()
    fig1(args.output_dir, groups)
    fig2(args.output_dir, groups)
    fig3(args.output_dir, groups)
    fig4(args.output_dir, groups)
    fig5(args.output_dir, conditional)
    fig6(args.output_dir, direct, direct_seed)
    supplement(args.output_dir, groups)
    prompt_supplement(args.output_dir)
    print(f"Validated {len(rows)} runs, 17 summaries, direct-label counts, and conditional recovery counts.")
    print(f"Exported 8 figures × PNG / PDF / SVG to {args.output_dir}")


if __name__ == "__main__":
    main()
