"""Methodology figure: what each row of the evaluation protocol changes.
    python src/fig_method.py   -> figures/fig_method.{png,pdf}
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
STYLE = {  # fill, edge
    "plain": ("#FFFFFF", "#7A7A7A"), "source": ("#DCE8F7", "#0F4D92"), "text": ("#FFF1C9", "#B08400"),
    "verdict": ("#E3F1E1", "#3C8D3C"), "meta": ("#FBE3D6", "#B85C38"), "note": ("#FFFFFF", "#9A9A9A"),
}
fig = plt.figure(figsize=(7.4, 6.6))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def box(x, y, w, h, title, body, kind="plain", dashed=False, lw=0.9):
    fc, ec = STYLE[kind]
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc=fc, ec=ec, lw=lw, ls="--" if dashed else "-"))
    if title:
        ax.text(x, y + h / 2 - 0.012, title, ha="center", va="top", fontsize=8.2, fontweight="bold")
        ax.text(x, y + h / 2 - 0.037, body, ha="center", va="top", fontsize=7.4, linespacing=1.25)
    else:
        ax.text(x, y, body, ha="center", va="center", fontsize=7.4, linespacing=1.25,
                color="#6E6E6E" if dashed else "black")
    return (x, y, w, h)


def arrow(a, b, dashed=False, color="#6E6E6E", start="bottom", end="top"):
    (xa, ya, wa, ha), (xb, yb, wb, hb) = a, b
    p0 = (xb, ya - ha / 2) if start == "bottom" else (xa + wa / 2, ya)
    p1 = (xb, yb + hb / 2) if end == "top" else (xb - wb / 2, yb)
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, color=color, lw=0.9,
                                 ls="--" if dashed else "-", shrinkA=0, shrinkB=0))


W, H = 0.32, 0.085
corpus = box(0.5, 0.955, 0.42, 0.06, "QazRuFake: 7,213 texts", "Russian 3,866 · Kazakh 3,347")
split = box(0.43, 0.875, 0.66, 0.06, "Group-aware train / dev / test split", "6,036 article and near-duplicate groups · 70 / 15 / 15")
arrow(corpus, split)
XR, XK = 0.19, 0.555
ax.text(XR - W / 2, 0.815, "Russian", ha="left", fontsize=9, fontweight="bold")
ax.text(XK - W / 2, 0.815, "Kazakh", ha="left", fontsize=9, fontweight="bold")
Y = [0.745, 0.635, 0.525, 0.405]
r1 = box(XR, Y[0], W, H, "Row 1 · Standard split", "all sources, original text\ntest 579")
k1 = box(XK, Y[0], W, H, "Row 1 · Standard split", "all sources, original text\ntest 502")
r2 = box(XR, Y[1], W, H, "Row 3 · Same publisher", "fact-checking outlets only (Tengrinews.kz\nposts removed) · train 1,529 · test 334", "source")
k2 = box(XK, Y[1], W, H, "Row 3 · Same publisher", "Factcheck.kz only (kk_news posts\nremoved) · train 1,800 · test 387", "source")
r3 = box(XR, Y[2], W, H, "Row 4 · Claim-only text", "channel, framing and rebuttal removed\nby rule · test 334", "text")
k3 = box(XK, Y[2], W, H, "Row 4 · Claim-only text", "channel, framing and rebuttal removed\nby rule · test 387", "text")
r4 = box(XR, Y[3], W, 0.07, None, "no graded verdicts in the Russian\nexport, so no Core subset", "note", dashed=True)
k4 = box(XK, Y[3], W, 0.1, "QazRuFake-Core", "unambiguous verdicts: true vs false,\nfake or mostly false · 1,578 items\ntrain 1,094 · test 255", "verdict", lw=1.6)
for a, b in [(split, r1), (split, k1), (r1, r2), (r2, r3), (k1, k2), (k2, k3), (k3, k4)]:
    arrow(a, b)
arrow(r3, r4, dashed=True, color="#9A9A9A")
meta = box(0.855, Y[0], 0.25, 0.1, "Row 2 · Metadata probe", "no text: source name,\nsource type, publication\nyear or surface statistics", "meta")
ax.add_patch(FancyArrowPatch((0.76, 0.875), (0.855, Y[0] + 0.05), arrowstyle="-|>", mutation_scale=8, color="#6E6E6E",
                             lw=0.9, connectionstyle="angle,angleA=0,angleB=90"))
# legend
ly = 0.60
for i, (k, lab) in enumerate([("source", "restricts the sources"), ("text", "changes the input text"),
                              ("verdict", "selects items by verdict"), ("meta", "removes the text")]):
    fc, ec = STYLE[k]
    ax.add_patch(FancyBboxPatch((0.745, ly - i * 0.035 - 0.009), 0.022, 0.018, boxstyle="round,pad=0.002",
                                fc=fc, ec=ec, lw=0.9))
    ax.text(0.775, ly - i * 0.035, lab, va="center", fontsize=7.4)
# example panel
ex = FancyBboxPatch((0.03, 0.06), 0.94, 0.25, boxstyle="round,pad=0.004,rounding_size=0.012",
                    fc="#FFFBEF", ec=STYLE["text"][1], lw=0.9)
ax.add_patch(ex)
ax.text(0.05, 0.293, "Row 4 example (Russian, Aqparat Saqshysy, FAKE)", fontsize=8.2, fontweight="bold", va="top")
lines = [("В мессенджерах казахстанцам приходит сообщение от имени менеджера по подбору", "strike"),
         ("персонала TikTok, в котором утверждается, что", "strike"),
         ("за просмотр видео в социальной сети TikTok можно ежедневно получать", "keep"),
         ("от 20 до 50 тысяч тенге.", "keep")]
y = 0.258
fig.canvas.draw()
for s, kind in lines:
    t = ax.text(0.05, y, s, fontsize=7.8, va="center", color="#8A8A8A" if kind == "strike" else "black",
                fontweight="normal" if kind == "strike" else "bold")
    if kind == "strike":
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(ax.transData.inverted())
        ax.plot([bb.x0, bb.x1], [y, y], color="#8A8A8A", lw=0.8)
    y -= 0.027
ax.text(0.05, y + 0.004, "English: [Kazakhstanis receive a message in messengers from a supposed TikTok recruiter, claiming that]\n"
        "one can earn 20,000–50,000 tenge a day by watching videos on TikTok.", fontsize=7.3, style="italic",
        color="#555555", va="top", linespacing=1.3)
ax.text(0.05, 0.08, "Struck through: circulation context and framing removed by the extractor. Bold: the claim that the model sees.",
        fontsize=7.0, color="#555555", va="center")
for ext in ("png", "pdf"):
    fig.savefig(f"figures/fig_method.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
