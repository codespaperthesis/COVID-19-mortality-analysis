"""
Forest plot — Crude and adjusted odds ratios for in-hospital death
(English version).

This reproduces, with translated text only, the corrected Portuguese
chart pasted into the conversation (crude age-group OR recalculated
pairwise against the <18 reference; crude Estado civil/Escolaridade OR
kept from the original run; adjusted OR as given). Values are hardcoded
from that reference figure rather than re-estimated, since the exact
adjusted model specification used to produce them is not available in
this environment — the crude age-group values were independently
verified against New_pacientes703.xlsx (see 22_forest_plot_sociodemografico_pt.py).
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _register_arial():
    candidates = [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            fm.fontManager.addfont(c)
            name = fm.FontProperties(fname=c).get_name()
            plt.rcParams["font.family"] = name
            return name
    plt.rcParams["font.family"] = "DejaVu Sans"
    return "DejaVu Sans"


_register_arial()

GREEN = "#1D9E75"
ORANGE = "#E07B39"
GOLD = "#caa63a"
TEXT = "#1a1a1a"
MUTED = "#8a8a8a"
STRIPE = "#eef0f3"

LINHAS = [
    {"tipo": "header", "label": "Sex"},
    {"tipo": "ref", "label": "Male", "n": 367, "pct": 52},
    {"tipo": "data", "label": "Female", "n": 336, "pct": 48,
     "crude": (0.87, 0.64, 1.17, False), "adj": (0.84, 0.52, 1.34, False)},

    {"tipo": "header", "label": "Marital status"},
    {"tipo": "ref", "label": "Single", "n": 344, "pct": 49},
    {"tipo": "data", "label": "Not single†", "n": 334, "pct": 47,
     "crude": (4.60, 3.32, 6.38, True), "adj": (1.09, 0.61, 1.93, False)},
    {"tipo": "data", "label": "Not reported†", "n": 25, "pct": 4,
     "crude": (0.48, 0.19, 1.22, False), "adj": (0.68, 0.17, 2.81, False)},

    {"tipo": "header", "label": "Age group"},
    {"tipo": "ref", "label": "<18", "n": 208, "pct": 30},
    {"tipo": "data", "label": "18–44", "n": 106, "pct": 15,
     "crude": (16.03, 6.00, 42.82, True), "adj": (21.61, 5.92, 78.87, True)},
    {"tipo": "data", "label": "45–60", "n": 135, "pct": 19,
     "crude": (41.21, 15.95, 106.47, True), "adj": (32.20, 9.08, 114.15, True)},
    {"tipo": "data", "label": ">60", "n": 254, "pct": 36,
     "crude": (83.65, 33.16, 210.99, True), "adj": (75.28, 21.32, 265.79, True)},

    {"tipo": "header", "label": "Education"},
    {"tipo": "ref", "label": "Low education", "n": 532, "pct": 76},
    {"tipo": "data", "label": "Medium education†", "n": 106, "pct": 15,
     "crude": (1.08, 0.71, 1.65, False), "adj": (0.86, 0.45, 1.61, False)},
    {"tipo": "data", "label": "High education†", "n": 27, "pct": 4,
     "crude": (0.92, 0.41, 2.03, False), "adj": (0.59, 0.19, 1.82, False)},

    {"tipo": "header", "label": "Residence"},
    {"tipo": "inestimavel", "label": "São Paulo state", "n": 701, "pct": 100},
    {"tipo": "data", "label": "Ribeirão Preto municipality", "n": 280, "pct": 40,
     "crude": (0.99, 0.73, 1.36, False), "adj": (1.01, 0.62, 1.64, False)},
]

n_rows = len(LINHAS)
row_h = 0.355
fig_h = n_rows * row_h + 1.5
fig, axes = plt.subplots(
    1, 4, figsize=(15.4, fig_h), facecolor="white",
    gridspec_kw={"width_ratios": [3.1, 3.3, 2.5, 3.3], "wspace": 0.05},
)
ax_labels, ax_c, ax_gap, ax_a = axes
ax_gap.set_visible(False)

for ax in (ax_labels, ax_c, ax_a):
    ax.set_ylim(n_rows - 0.5, -0.5)

for i, linha in enumerate(LINHAS):
    if linha["tipo"] != "header" and i % 2 == 1:
        for ax in (ax_labels, ax_c, ax_a):
            ax.axhspan(i - 0.5, i + 0.5, color=STRIPE, zorder=0)

for ax in (ax_c, ax_a):
    ax.set_xscale("log")
    ax.set_xlim(0.1, 400)
    ax.axvline(1.0, color=GOLD, linewidth=1.3, linestyle="--", zorder=2)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=9.2)
    ax.set_xlabel("Odds ratio (log scale)", fontsize=9.6, color="#444")

ax_labels.set_xlim(0, 1)
for spine in ax_labels.spines.values():
    spine.set_visible(False)
ax_labels.set_xticks([])
ax_labels.set_yticks([])

ax_c.set_title("Crude OR (95% CI)", fontsize=12, fontweight="bold", color=TEXT, pad=9)
ax_a.set_title("Adjusted OR (95% CI)", fontsize=12, fontweight="bold", color=TEXT, pad=9)

for i, linha in enumerate(LINHAS):
    if linha["tipo"] == "header":
        ax_labels.text(0.0, i, linha["label"], fontsize=11.5,
                        fontweight="bold", color=TEXT, va="center", ha="left")
        continue
    ax_labels.text(0.045, i, linha["label"], fontsize=10.6, color=TEXT, va="center", ha="left")
    ax_labels.text(0.98, i, f"n = {linha['n']} ({linha['pct']}%)",
                    fontsize=8.8, style="italic", color=MUTED, va="center", ha="right")

    if linha["tipo"] == "ref":
        for ax in (ax_c, ax_a):
            ax.text(0.5, i, "ref.", transform=ax.get_yaxis_transform(),
                    ha="center", va="center", fontsize=9.8, color=MUTED, style="italic")
        continue
    if linha["tipo"] == "inestimavel":
        for ax in (ax_c, ax_a):
            ax.plot(1.0, i, marker="D", markersize=5.5, color="black", zorder=5)
            ax.text(0.5, i, "not estimable", transform=ax.get_yaxis_transform(),
                    ha="center", va="center", fontsize=9.5, color=MUTED, style="italic")
        continue

    for ax, key, bold in [(ax_c, "crude", True), (ax_a, "adj", False)]:
        OR, lo, hi, sig = linha[key]
        cor = GREEN if OR < 1 else ORANGE
        lo_p = max(lo, 0.105)
        hi_p = min(hi, 380)
        ax.plot([lo_p, hi_p], [i, i], color=cor, linewidth=2.0, zorder=3, solid_capstyle="round")
        ax.plot(OR, i, marker="D" if sig else "o", markersize=7.3 if sig else 6.8,
                color=cor, markerfacecolor=cor if sig else "white",
                markeredgecolor=cor, markeredgewidth=1.4, zorder=5)
        txt = f"{OR:.2f} ({lo:.2f}–{hi:.2f})" + ("***" if sig else "")
        ax.text(1.0, i, txt, transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=9.8, color=TEXT,
                fontweight="bold" if (bold and sig) else "normal", clip_on=False)

legend_elements = [
    mpatches.Patch(facecolor=GREEN, edgecolor=GREEN, label="OR < 1"),
    mpatches.Patch(facecolor=ORANGE, edgecolor=ORANGE, label="OR > 1"),
    Line2D([0], [0], marker="D", color="none", markerfacecolor=TEXT, markeredgecolor=TEXT,
           markersize=6, label="p < 0.05"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=TEXT,
           markeredgewidth=1.1, markersize=6.5, label="p ≥ 0.05"),
    Line2D([0], [0], color=GOLD, linewidth=1.3, linestyle="--", label="OR = 1"),
]
fig.legend(handles=legend_elements, fontsize=9.6, frameon=False, ncol=5,
           loc="upper center", bbox_to_anchor=(0.58, 1.0), labelcolor=TEXT,
           handletextpad=0.5, columnspacing=1.4)

fig.suptitle("Forest plot — Crude and adjusted odds ratios for in-hospital death",
             fontsize=15.5, fontweight="bold", color=TEXT, x=0.012, ha="left", y=1.12)
fig.text(0.012, 1.065,
          "Sociodemographic variables | N = 703 | 274 deaths | Crude age-group OR recalculated with <18 years as reference",
          ha="left", va="top", fontsize=9.8, color="#555")

footnote = (
    "Notes: Crude OR from simple logistic regression; adjusted OR from multiple logistic regression with mutual adjustment among the variables shown.\n"
    "Age group (crude): deaths/discharges <18 = 5/203; 18–44 = 30/76; 45–60 = 68/67; >60 = 171/83; 95% CI and Wald p.\n"
    "† Crude values kept from the original run: for variables with more than two categories, confirm in the dataset whether they were estimated against the stated reference category.\n"
    "Education: 38 missing. *p<0.05; **p<0.01; ***p<0.001."
)
fig.text(0.012, 0.025, footnote, ha="left", va="top", fontsize=7.6, color="#555")

fig.subplots_adjust(top=0.84, bottom=0.145, left=0.012, right=0.99)

for ext, dpi in [("png", 180)]:
    out_path = os.path.join(BASE_DIR, f"forest_sociodemographic_en.{ext}")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print("saved:", out_path)

out_tiff = os.path.join(BASE_DIR, "forest_sociodemographic_en_600dpi.tiff")
fig.savefig(out_tiff, dpi=600, bbox_inches="tight", facecolor="white", format="tiff",
            pil_kwargs={"compression": "tiff_lzw"})
print("saved:", out_tiff)

plt.close(fig)
