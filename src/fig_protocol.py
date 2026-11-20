"""Figure 1: macro-F1 along the evaluation protocol, by language and model.

Values: results/res_rev.json (SVM), results/res_v2.json and res_v3.json (XLM-R), res_v3.json (zero-shot LLM). Majority-class macro-F1 of each test set is drawn as a dashed tick.
    python src/fig_protocol.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, RED, GREY, TEAL = "#0F4D92", "#E9A6A1", "#6E6E6E", "#42949E"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 1.2, "legend.frameon": False,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})

PANELS = {
    "Russian": dict(
        stages=["Standard\nsplit", "Same\npublisher", "+ claim-only\ntext"],
        n=["579", "334", "334"],
        svm=[0.955, 0.825, 0.694], xlmr=[0.967, 0.886, 0.650], llm=[0.809, 0.658, 0.632],
        xlmr_seeds=[None, [0.900, 0.888, 0.870], [0.799, 0.696, 0.455]],
        majority=[0.341, 0.455, 0.455]),
    "Kazakh": dict(
        stages=["Standard\nsplit", "Same\npublisher", "+ claim-only\ntext", "+ unambiguous\nverdicts"],
        n=["502", "387", "387", "255"],
        svm=[0.824, 0.811, 0.738, 0.814], xlmr=[0.830, 0.813, 0.765, 0.831], llm=[0.620, 0.587, 0.641, 0.669],
        xlmr_seeds=[None, None, None, None],
        majority=[0.384, 0.338, 0.338, 0.349]),
}


def main(out="figures/fig1_protocol"):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), gridspec_kw={"width_ratios": [3, 4]}, sharey=True)
    w = 0.26
    for ax, (lang, p) in zip(axes, PANELS.items()):
        x = np.arange(len(p["stages"]))
        for off, vals, col, name in [(-w, p["svm"], RED, "SVM (word + char n-grams)"),
                                     (0, p["xlmr"], BLUE, "XLM-R base (fine-tuned)"),
                                     (w, p["llm"], TEAL, "Qwen2.5-7B (zero-shot)")]:
            for k, (xi, v) in enumerate(zip(x + off, vals)):
                ax.bar(xi, v, w, color=col, edgecolor="black", linewidth=0.6, label=name)
                seeds = p["xlmr_seeds"][k] if off == 0 else None
                top = max([v] + (seeds or [])) + 0.012
                ax.text(xi, top, f"{v:.3f}".lstrip("0"), ha="center", va="bottom", fontsize=6.5)
        for xi, seeds in zip(x, p["xlmr_seeds"]):
            if seeds:
                ax.scatter([xi] * len(seeds), seeds, s=9, color="white", edgecolor="black", linewidth=0.6, zorder=3,
                           label="XLM-R single seeds")
        for xi, m in zip(x, p["majority"]):
            ax.plot([xi - 1.5 * w, xi + 1.5 * w], [m, m], ls="--", color=GREY, lw=1.1, label="Majority class")
        ax.set_xticks(x, [f"{s}\n(n = {n})" for s, n in zip(p["stages"], p["n"])], fontsize=8.5)
        ax.set_title(lang, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[0].set_ylabel("Test macro-F1")
    h, l = axes[0].get_legend_handles_labels()
    uniq = dict(zip(l, h))
    fig.legend(uniq.values(), uniq.keys(), loc="upper center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
