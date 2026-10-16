import os
import matplotlib as mpl
import matplotlib.font_manager as fm


# ── Register Arial (with local fallback, no network dependency) ───────────
def _register_arial():
    fonts = [f.name for f in fm.fontManager.ttflist]
    if "Arial" in fonts:
        return "Arial"
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        os.path.join(os.path.dirname(__file__), "Arial.ttf"),
    ]
    for path in candidates:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            return "Arial"
    try:
        import urllib.request
        dest = os.path.join(os.path.expanduser("~"), "Arial.ttf")
        if not os.path.exists(dest):
            url = "https://github.com/matomo-org/travis-scripts/raw/master/fonts/Arial.ttf"
            urllib.request.urlretrieve(url, dest)
        fm.fontManager.addfont(dest)
        return "Arial"
    except Exception:
        liberation = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
        if os.path.exists(liberation):
            fm.fontManager.addfont(liberation)
            return "Liberation Sans"
    return "Arial"


_FONTE = _register_arial()
mpl.rcParams["font.family"] = _FONTE

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────
# Forest plot — crude and adjusted odds ratio for in-hospital death,
# sociodemographic variables (English version of the Portuguese figure:
# "Forest plot — Razão de chances bruta e ajustada para óbito hospitalar").
# Values are the ones already reported in the Portuguese version (n = 703;
# 274 deaths); this script only re-renders them in English, unchanged.
# ─────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "forest_plot_sociodemographic_en.png")
OUTPUT_TIFF_300 = os.path.join(OUTPUT_DIR, "forest_plot_sociodemographic_en.tiff")
OUTPUT_TIFF_600 = os.path.join(OUTPUT_DIR, "forest_plot_sociodemographic_en_600dpi.tiff")

N_TOTAL = 703
N_DEATHS = 274

# ── Data (header rows and data rows, in display order) ─────────────────────
ROWS = [
    {"type": "header", "label": "Sex"},
    {"type": "ref", "label": "Male", "n_pct": "n = 367 (52%)"},
    {"type": "row", "label": "Female", "n_pct": "n = 336 (48%)",
     "or": 0.87, "lo": 0.64, "hi": 1.17, "p": 0.40,
     "ora": 0.84, "aloa": 0.52, "ahia": 1.34, "pa": 0.46},

    {"type": "header", "label": "Marital status"},
    {"type": "ref", "label": "Single", "n_pct": "n = 344 (49%)"},
    {"type": "row", "label": "Not single\u2020", "n_pct": "n = 334 (47%)",
     "or": 4.60, "lo": 3.32, "hi": 6.38, "p": 0.0001,
     "ora": 1.09, "aloa": 0.61, "ahia": 1.93, "pa": 0.77},
    {"type": "row", "label": "Not informed\u2020", "n_pct": "n = 25 (4%)",
     "or": 0.48, "lo": 0.19, "hi": 1.22, "p": 0.12,
     "ora": 0.68, "aloa": 0.17, "ahia": 2.81, "pa": 0.59},

    {"type": "header", "label": "Age group"},
    {"type": "ref", "label": "<18", "n_pct": "n = 208 (30%)"},
    {"type": "row", "label": "18\u201344", "n_pct": "n = 106 (15%)",
     "or": 16.03, "lo": 6.00, "hi": 42.82, "p": 0.0001,
     "ora": 21.61, "aloa": 5.92, "ahia": 78.87, "pa": 0.0001},
    {"type": "row", "label": "45\u201360", "n_pct": "n = 135 (19%)",
     "or": 41.21, "lo": 15.95, "hi": 106.47, "p": 0.0001,
     "ora": 32.20, "aloa": 9.08, "ahia": 114.15, "pa": 0.0001},
    {"type": "row", "label": ">60", "n_pct": "n = 254 (36%)",
     "or": 83.65, "lo": 33.16, "hi": 210.99, "p": 0.0001,
     "ora": 75.28, "aloa": 21.32, "ahia": 265.79, "pa": 0.0001},

    {"type": "header", "label": "Education"},
    {"type": "ref", "label": "Low education", "n_pct": "n = 532 (76%)"},
    {"type": "row", "label": "Medium education\u2020", "n_pct": "n = 106 (15%)",
     "or": 1.08, "lo": 0.71, "hi": 1.65, "p": 0.73,
     "ora": 0.86, "aloa": 0.45, "ahia": 1.61, "pa": 0.63},
    {"type": "row", "label": "High education\u2020", "n_pct": "n = 27 (4%)",
     "or": 0.92, "lo": 0.41, "hi": 2.03, "p": 0.83,
     "ora": 0.59, "aloa": 0.19, "ahia": 1.82, "pa": 0.36},

    {"type": "header", "label": "Residence"},
    {"type": "notest", "label": "State of S\u00e3o Paulo", "n_pct": "n = 701 (100%)"},
    {"type": "row", "label": "Ribeir\u00e3o Preto municipality", "n_pct": "n = 280 (40%)",
     "or": 0.99, "lo": 0.73, "hi": 1.36, "p": 0.95,
     "ora": 1.01, "aloa": 0.62, "ahia": 1.64, "pa": 0.97},
]

# ════════════════════════════════════════════════════════════
# PALETTE
# ════════════════════════════════════════════════════════════
BG = "#FFFFFF"
PANEL = "#F6F8FA"
BORDER = "#D0D7DE"
TEXT = "#1F2328"
SUBTEXT = "#57606A"
GOLD = "#B08800"
COR_PROT = "#1D9E75"
COR_RISCO = "#E07B39"
ZEBRA = "#E8EDF2"


def sig_stars(p):
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


n_rows = len(ROWS)
POL_CABECALHO = 3.3
POL_RODAPE = 2.3
altura_conteudo = n_rows * 0.56 + 1.0
fig_h = POL_CABECALHO + altura_conteudo + POL_RODAPE
top_frac = 1 - POL_CABECALHO / fig_h
bottom_frac = POL_RODAPE / fig_h

fig, axes = plt.subplots(
    1, 4, figsize=(19.5, fig_h), facecolor=BG,
    gridspec_kw={"width_ratios": [3.0, 3.3, 2.6, 3.3], "wspace": 0.06,
                 "top": top_frac, "bottom": bottom_frac},
)
ax_labels, ax_or, ax_gap, ax_ora = axes
ax_gap.set_visible(False)

# ── Label panel ──────────────────────────────────────────────────────────
ax_labels.set_facecolor(BG)
ax_labels.set_xlim(0, 1)
ax_labels.set_ylim(n_rows - 0.5, -0.5)
for spine in ax_labels.spines.values():
    spine.set_visible(False)
ax_labels.set_xticks([])
ax_labels.set_yticks([])

# Zebra stripes (skip header rows)
for ax in (ax_labels, ax_or, ax_ora):
    stripe = 0
    for i, row in enumerate(ROWS):
        if row["type"] != "header":
            bg_col = ZEBRA if stripe % 2 == 0 else PANEL
            ax.axhspan(i - 0.42, i + 0.42, color=bg_col, alpha=0.55, zorder=0)
            stripe += 1

for i, row in enumerate(ROWS):
    if row["type"] == "header":
        ax_labels.text(0.98, i, row["label"], color=TEXT, fontsize=16,
                        fontweight="bold", va="center", ha="right")
        ax_labels.text(0.02, i, "", va="center", ha="left")
    else:
        lbl_color = SUBTEXT if row["type"] in ("ref", "notest") else TEXT
        ax_labels.text(0.98, i, f"  {row['label']}", color=lbl_color,
                        fontsize=14, va="center", ha="right")
        ax_labels.text(0.98, i - 0.26, row["n_pct"], color=SUBTEXT,
                        fontsize=10.5, style="italic", va="center", ha="right")


def draw_panel(ax, col_or, col_lo, col_hi, col_p, title, xlim):
    ax.set_facecolor(PANEL)
    ax.set_xscale("log")
    ax.xaxis.grid(True, color=BORDER, linewidth=0.9, zorder=0, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(0.8)
    ax.axvline(1.0, color=GOLD, linewidth=2.2, linestyle="--", zorder=2, alpha=0.9)

    for i, row in enumerate(ROWS):
        if row["type"] == "header":
            continue
        if row["type"] == "ref":
            ax.text(0.5, i, "ref.", color=SUBTEXT, fontsize=13, style="italic",
                    va="center", ha="center", transform=ax.get_yaxis_transform())
            continue
        if row["type"] == "notest":
            ax.text(0.5, i, "not estimable", color=SUBTEXT, fontsize=12,
                    style="italic", va="center", ha="center",
                    transform=ax.get_yaxis_transform())
            continue

        OR_ = row[col_or]
        lo_ = row[col_lo]
        hi_ = row[col_hi]
        p_ = row[col_p]

        cor = COR_PROT if OR_ < 1 else COR_RISCO
        sig = p_ is not None and p_ < 0.05

        lo_plot = max(lo_, xlim[0] * 1.02)
        hi_plot = min(hi_, xlim[1] * 0.98)
        ax.plot([lo_plot, hi_plot], [i, i], color=cor, linewidth=3.4, zorder=3,
                 alpha=0.85, solid_capstyle="round")
        ax.plot(OR_, i, marker="D" if sig else "o", markersize=8 if sig else 6.5,
                color=cor, markerfacecolor=cor if sig else BG,
                markeredgecolor=cor, markeredgewidth=1.5, zorder=5)

        p_str = f"{sig_stars(p_)}"
        txt = f"{OR_:.2f} ({lo_:.2f}\u2013{hi_:.2f}){p_str}"
        ax.text(1.05, i, txt, transform=ax.get_yaxis_transform(),
                color=TEXT, fontsize=12.5, va="center", ha="left", clip_on=False)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([""] * n_rows)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=SUBTEXT, labelsize=10)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xlim(*xlim)
    ax.set_xlabel("Odds ratio (log scale)", fontsize=14, color=SUBTEXT, labelpad=6)
    ax.set_title(title, fontsize=16, fontweight="bold", color=TEXT, pad=8)


draw_panel(ax_or, "or", "lo", "hi", "p", "Crude OR (95% CI)", xlim=(0.1, 400))
draw_panel(ax_ora, "ora", "aloa", "ahia", "pa", "Adjusted OR (95% CI)", xlim=(0.1, 400))

# ── Legend row (top, matching the Portuguese figure) ───────────────────────
legend_elements = [
    mpatches.Patch(facecolor=COR_PROT, edgecolor=COR_PROT, label="OR < 1"),
    mpatches.Patch(facecolor=COR_RISCO, edgecolor=COR_RISCO, label="OR > 1"),
    Line2D([0], [0], marker="D", color="none", markerfacecolor=TEXT,
           markeredgecolor=TEXT, markersize=7, label="p < 0.05"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=BG,
           markeredgecolor=TEXT, markeredgewidth=1.6, markersize=8,
           label="p \u2265 0.05"),
    Line2D([0], [0], color=GOLD, linewidth=2.0, linestyle="--", label="OR = 1"),
]

y_titulo = 1 - 0.42 / fig_h
y_subtitulo = 1 - 1.00 / fig_h
y_legenda = 1 - 1.55 / fig_h
y_divisor = 1 - 1.90 / fig_h

fig.text(0.5, y_titulo,
          "Forest plot \u2014 Crude and adjusted odds ratio for in-hospital death",
          ha="center", va="top", fontsize=24, fontweight="bold", color=TEXT)
fig.text(0.5, y_subtitulo,
          f"Sociodemographic variables | N = {N_TOTAL} | {N_DEATHS} deaths | "
          "Crude OR for age group recalculated with <18 years as reference",
          ha="center", va="top", fontsize=13.5, color=SUBTEXT)
fig.legend(handles=legend_elements, loc="upper center",
           bbox_to_anchor=(0.5, y_legenda), ncol=5, frameon=False,
           fontsize=11, labelcolor=TEXT, columnspacing=1.4, handlelength=1.4)
fig.add_artist(plt.Line2D([0.03, 0.97], [y_divisor, y_divisor],
                           transform=fig.transFigure, color=BORDER, linewidth=1.4))

# ── Footnotes ────────────────────────────────────────────────────────────
notas = (
    "Notes: Crude OR from simple logistic regression; adjusted OR from multiple logistic regression "
    "with mutual adjustment among the variables shown.\n"
    "Age group (crude): deaths/discharges <18 = 5/203; 18\u201344 = 30/76; 45\u201360 = 68/67; "
    ">60 = 171/83; 95% CI and Wald p.\n"
    "\u2020 Crude values kept from the original version: for variables with more than two categories, "
    "confirm in the dataset whether they were estimated against the indicated reference category.\n"
    "Education: 38 missing. *p<0.05; **p<0.01; ***p<0.001."
)
fig.text(0.03, 0.012, notas, color=SUBTEXT, fontsize=10, style="italic",
          va="bottom", ha="left", linespacing=1.5)

# ── Export ──────────────────────────────────────────────────────────────
fig.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight", facecolor=BG)

_buf_600 = OUTPUT_PNG.replace(".png", "_600dpi_tmp.png")
fig.savefig(_buf_600, dpi=600, bbox_inches="tight", facecolor=BG)
Image.open(_buf_600).save(OUTPUT_TIFF_600, dpi=(600, 600), compression="tiff_lzw")
os.remove(_buf_600)

_buf_300 = OUTPUT_PNG.replace(".png", "_300dpi_tmp.png")
fig.savefig(_buf_300, dpi=300, bbox_inches="tight", facecolor=BG)
Image.open(_buf_300).save(OUTPUT_TIFF_300, dpi=(300, 300), compression="tiff_lzw")
os.remove(_buf_300)

print(f"Saved to:\n  {OUTPUT_PNG}\n  {OUTPUT_TIFF_300}\n  {OUTPUT_TIFF_600}")
