"""Figure 1: macro-F1 along the evaluation protocol, by language and model.

SVM: results/res_rev.json. XLM-R: class-weighted loss, mean ± SD over 5 seeds (10 for the Russian claim-only row and
QazRuFake-Core), results/res_v4.json. LLM: Qwen2.5-14B-Instruct, prompt in the language of the text, threshold chosen
on dev, results/res_v4.json. Majority-class macro-F1 of each test set is drawn as a dashed line.
    python src/fig_protocol.py
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, RED, GREY, TEAL = "#0F4D92", "#E9A6A1", "#555555", "#42949E"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 1.2, "legend.frameon": False,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})
V4 = json.load(open("results/res_v4.json"))
X = lambda k: V4["xlmr_cw|" + k]
LL = lambda k: V4["llm14b|" + k]["f1_thrdev"]
XR = [X("ru|text|ru"), X("rufc|text|ru_pub"), X("rufc|claim|ru_pub")]
XK = [X("kk|text|kk"), X("kkfc|text|kk_pub"), X("kkfc|claim|kk_pub"), X("kkunamb|claim|kk_core")]
PANELS = {
    "Russian": dict(stages=["Standard\nsplit", "Same\npublisher", "+ claim-only\ntext"], n=["579", "334", "334"],
        svm=[0.955, 0.825, 0.694], xlmr=XR, llm=[LL("text|ru"), LL("text|ru_pub"), LL("claim|ru_pub")],
        majority=[0.341, 0.455, 0.455]),
    "Kazakh": dict(stages=["Standard\nsplit", "Same\npublisher", "+ claim-only\ntext", "+ unambiguous\nverdicts"],
        n=["502", "387", "387", "255"], svm=[0.824, 0.811, 0.738, 0.814], xlmr=XK,
        llm=[LL("text|kk"), LL("text|kk_pub"), LL("claim|kk_pub"), LL("claim|kk_core")],
        majority=[0.384, 0.338, 0.338, 0.349]),
}

def main(out="figures/fig1_protocol"):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), gridspec_kw={"width_ratios": [3, 4]}, sharey=True)
    w = 0.26
    for ax, (lang, p) in zip(axes, PANELS.items()):
        x = np.arange(len(p["stages"]))
        ax.yaxis.grid(True, color="#e5e5e5", lw=0.8); ax.set_axisbelow(True)
        for k, xi in enumerate(x):
            xm = p["xlmr"][k]; m, sd = xm["f1_mean"], xm["f1_sd"]
            for off, v, col, name in [(-w, p["svm"][k], RED, "SVM (word + char n-grams)"),
                                      (0, m, BLUE, "XLM-R base (fine-tuned, class-weighted, mean ± SD)"),
                                      (w, p["llm"][k], TEAL, "Qwen2.5-14B (zero-shot)")]:
                ax.bar(xi + off, v, w, color=col, edgecolor="black", linewidth=0.6, label=name)
                top = v + 0.012
                if off == 0:
                    ax.errorbar(xi, m, yerr=sd, color="black", capsize=2.5, lw=1)
                    top = max(m + sd, max(xm["f1_per_seed"]) if xm["n_seeds"] >= 10 else 0) + 0.012
                ax.text(xi + off, top, f"{v:.3f}".lstrip("0"), ha="center", va="bottom", fontsize=6.5)
            if xm["n_seeds"] >= 10:
                ax.scatter([xi] * xm["n_seeds"], xm["f1_per_seed"], s=8, color="white", edgecolor="black", linewidth=0.5,
                           zorder=3, label="XLM-R single seeds (10-seed rows)")
            ax.plot([xi - 1.5 * w, xi + 1.5 * w], [p["majority"][k]] * 2, ls="--", color=GREY, lw=1.1, label="Majority class")
        ax.set_xticks(x, [f"{s}\n(n = {n})" for s, n in zip(p["stages"], p["n"])], fontsize=8.5)
        ax.set_title(lang, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[0].set_ylabel("Test macro-F1")
    h, l = axes[0].get_legend_handles_labels(); uniq = dict(zip(l, h))
    order = ["SVM (word + char n-grams)", "XLM-R base (fine-tuned, class-weighted, mean ± SD)", "Qwen2.5-14B (zero-shot)",
             "Majority class", "XLM-R single seeds (10-seed rows)"]
    fig.legend([uniq[k] for k in order], order, loc="upper center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, 1.0),
               columnspacing=1.2, handlelength=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(f"{out}.png", dpi=600, bbox_inches="tight"); fig.savefig(f"{out}.pdf", bbox_inches="tight")

if __name__ == "__main__":
    main()
