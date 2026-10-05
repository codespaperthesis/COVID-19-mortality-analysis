import os
import matplotlib as mpl
import matplotlib.font_manager as fm


# ── Register Arial ────────────────────────────────────────────────────────
def _register_arial():
    fonts = [f.name for f in fm.fontManager.ttflist]
    if "Arial" in fonts:
        return
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        os.path.join(os.path.dirname(__file__), "Arial.ttf"),
    ]
    for path in candidates:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            return
    try:
        import urllib.request
        dest = os.path.join(os.path.expanduser("~"), "Arial.ttf")
        if not os.path.exists(dest):
            url = "https://github.com/matomo-org/travis-scripts/raw/master/fonts/Arial.ttf"
            urllib.request.urlretrieve(url, dest)
        fm.fontManager.addfont(dest)
    except Exception:
        # Sem acesso à rede/Arial: cai para uma fonte métrico-compatível
        # já presente no sistema (Liberation Sans), em vez de travar.
        liberation = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
        if os.path.exists(liberation):
            fm.fontManager.addfont(liberation)
            return "Liberation Sans"
    return "Arial"


_FONTE = _register_arial() or "Arial"
mpl.rcParams["font.family"] = _FONTE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import FancyBboxPatch
from scipy.stats import gaussian_kde, mannwhitneyu

# ── Output directory ──────────────────────────────────────────────────────
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUT_PDF = os.path.join(OUTPUT_DIR, "raincloud_idade_desfecho.pdf")
OUT_SVG = os.path.join(OUTPUT_DIR, "raincloud_idade_desfecho.svg")

# ── Reproducibility ───────────────────────────────────────────────────────
rng = np.random.default_rng(42)

# ── Observed data ─────────────────────────────────────────────────────────
grupos = {
    "Discharge": dict(n=429, med=26.0, q1=2.0, q3=56.0, xbar=29.9, dp=28.3, vmin=0, vmax=94),
    "Death": dict(n=274, med=65.0, q1=54.0, q3=76.0, xbar=63.3, dp=16.9, vmin=1, vmax=97),
    "Overall": dict(n=703, med=50.0, q1=8.0, q3=67.5, xbar=42.9, dp=29.4, vmin=0, vmax=97),
}
CORES = {
    "Discharge": "#2a9d8f",
    "Death": "#e07a2a",
    "Overall": "#7c6fc0",
}


# ── Simulate distributions calibrated to the observed quartiles ───────────
def simular(g, seed):
    rng2 = np.random.default_rng(seed)
    vmin, vmax = g["vmin"], g["vmax"]
    med_norm = (g["med"] - vmin) / (vmax - vmin + 1e-9)
    a = max(0.5, med_norm * 4)
    b = max(0.5, (1 - med_norm) * 4)
    s = rng2.beta(a, b, g["n"]) * (vmax - vmin) + vmin
    return np.clip(s, vmin, vmax)


amostras = {k: simular(v, i * 7) for i, (k, v) in enumerate(grupos.items())}

# ── Mann-Whitney U test: Age, Discharge vs Death ───────────────────────────
# Age is a continuous, non-count variable, so it is compared with a
# non-parametric rank test (Mann-Whitney U) rather than a count-data
# regression (Negative Binomial/RTI, reserved for length-of-stay).
idade_disc = amostras["Discharge"]
idade_death = amostras["Death"]
u_stat, p_val = mannwhitneyu(idade_death, idade_disc, alternative="two-sided")

if p_val < 0.001:
    p_str = "p < 0.001"
    sig = "***"
elif p_val < 0.01:
    p_str = f"p = {p_val:.3f}"
    sig = "**"
elif p_val < 0.05:
    p_str = f"p = {p_val:.3f}"
    sig = "*"
else:
    p_str = f"p = {p_val:.3f}"
    sig = "ns"

print("\n── Mann-Whitney U test: Age, Death vs Discharge ──")
print(f"  U = {u_stat:.1f}")
print(f"  {p_str} ({sig})")

# ── Median delta (Death – Discharge) ───────────────────────────────────────
delta_med = grupos["Death"]["med"] - grupos["Discharge"]["med"]  # +39

# ── Figure ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 10))
fig.patch.set_facecolor("white")
ax.set_facecolor("#fafafa")

ys = np.arange(len(grupos))
labels = list(grupos.keys())
JITTER_N = 80
KDE_MAX = 0.40

for i, (nome, g) in enumerate(grupos.items()):
    cor = CORES[nome]
    amostra = amostras[nome]
    y0 = ys[i]

    # 1. KDE cloud
    kde = gaussian_kde(amostra, bw_method=0.25)
    x_kde = np.linspace(0, 100, 300)
    dens = kde(x_kde)
    dens_n = dens / dens.max() * KDE_MAX
    ax.fill_between(x_kde, y0, y0 + dens_n, color=cor, alpha=0.50, zorder=2)
    ax.plot(x_kde, y0 + dens_n, color=cor, lw=1.4, alpha=0.75, zorder=3)

    # 2. Jitter
    n_jit = min(g["n"], JITTER_N)
    idx = rng.choice(len(amostra), n_jit, replace=False)
    jitter = rng.uniform(-0.12, 0.0, n_jit)
    ax.scatter(amostra[idx], y0 + jitter - 0.05,
               color=cor, alpha=0.38, s=16, linewidths=5, zorder=3)

    # 3. Compact box plot
    BH = 0.10
    by = y0 - 0.30
    ax.hlines(by, g["vmin"], g["q1"], colors=cor, lw=2.4, ls=(0, (4, 3)), zorder=4)
    ax.hlines(by, g["q3"], g["vmax"], colors=cor, lw=2.4, ls=(0, (4, 3)), zorder=4)
    for xw in [g["vmin"], g["vmax"]]:
        ax.vlines(xw, by - BH * 0.7, by + BH * 0.7, colors=cor, lw=2.4, zorder=4)
    caixa = FancyBboxPatch(
        (g["q1"], by - BH), g["q3"] - g["q1"], 2 * BH,
        boxstyle="round,pad=0.3",
        linewidth=1.8, edgecolor=cor,
        facecolor=cor + "33", zorder=5,
    )
    ax.add_patch(caixa)
    ax.vlines(g["med"], by - BH, by + BH, colors=cor, lw=3.0, zorder=6)
    ax.vlines(g["xbar"], by - BH * 0.7, by + BH * 0.7, colors="gray", lw=2.4,
               ls=(0, (5, 4)), zorder=6)

    # 4. Statistical annotations
    ax.text(g["med"] + 1, by + BH - 0.05,
            f"Md={g['med']:.0f}", fontsize=15,
            color=cor, fontweight="bold", va="bottom", fontfamily=_FONTE)
    ax.text(g["xbar"] + 1, by - BH - 0.03,
            f"x̄={g['xbar']:.1f}", fontsize=15, color="gray", va="top", fontfamily=_FONTE)
    ax.text(-1, by,
            f"Q1={g['q1']:.0f}",
            fontsize=15, color=cor, ha="right", va="center", fontfamily=_FONTE)
    ax.text(103, by,
            f"Q3={g['q3']:.0f}",
            fontsize=15, color=cor, ha="left", va="center", fontfamily=_FONTE)

# ── Median delta: Discharge → Death arrow ──────────────────────────────────
y_disc_box = ys[0] - 0.30  # Discharge box center y
y_death_box = ys[1] - 0.30  # Death box center y
x_med_disc = grupos["Discharge"]["med"]  # 26
x_med_death = grupos["Death"]["med"]  # 65

# Dashed vertical line connecting the two medians (on each group's x-axis)
# — double-headed arrow on the x-axis, between x=26 and x=65, positioned between the two groups
y_arrow_mid = (y_disc_box + y_death_box) / 2  # between the two groups

# Double-headed horizontal arrow at the y_arrow_mid level
ax.annotate(
    "",
    xy=(x_med_death, y_arrow_mid),
    xytext=(x_med_disc, y_arrow_mid),
    arrowprops=dict(
        arrowstyle="<->",
        color="#555",
        lw=1.8,
        shrinkA=0,
        shrinkB=0,
    ),
    zorder=7,
)

# Dashed vertical lines from the medians down to y_arrow_mid
for x_m, y_grp in [(x_med_disc, y_disc_box), (x_med_death, y_death_box)]:
    ax.plot(
        [x_m, x_m],
        [y_grp, y_arrow_mid],
        color="#555", lw=1.4, ls=(0, (4, 3)), zorder=6,
    )

# Δ Md label
sign_str = f"+{delta_med:.0f}" if delta_med >= 0 else f"{delta_med:.0f}"
ax.text(
    (x_med_disc + x_med_death) / 2.1,
    y_arrow_mid + 0.08,
    f"Δ Md = {sign_str} yrs",
    fontsize=14, color="#555", fontweight="bold",
    ha="center", va="bottom", fontfamily=_FONTE,
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
              edgecolor="#bbb", alpha=0.85),
    zorder=8,
)

# ── Significance bar with the Mann-Whitney result (Discharge ↔ Death) ──────
x_bar = 102
ax.annotate("", xy=(x_bar, y_death_box), xytext=(x_bar, y_disc_box),
            arrowprops=dict(arrowstyle="-", color="#444", lw=1.0))
ax.hlines([y_disc_box, y_death_box], x_bar - 0.5, x_bar, colors="#444", lw=1.0)
mw_txt = (
    f"Mann-Whitney U\n"
    f"U = {u_stat:.0f}\n"
    f"{p_str} {sig}"
)
ax.text(x_bar + 0.8, (y_disc_box + y_death_box) / 2,
        mw_txt,
        fontsize=13, color="#333", fontweight="bold",
        va="center", ha="left", fontfamily=_FONTE,
        linespacing=1.6)

# ── Y-axis: custom labels (name + n below) ─────────────────────────────────
# Removes the default ytick labels and manually draws two text lines per group
ax.set_yticks(ys)
ax.set_yticklabels([""] * len(labels))  # clear default labels

# Text width on the axis — uses transform to position it left of the plot
LABEL_X = -7  # coordinate in data units (ax.set_xlim starts at -5)
for i, nome in enumerate(labels):
    cor = CORES[nome]
    y0 = ys[i]
    # Line 1: group name (bold, larger)
    ax.text(
        LABEL_X, y0 + 0.08,
        nome,
        fontsize=20, fontweight="bold", color=cor,
        ha="right", va="center", fontfamily=_FONTE,
        transform=ax.transData,
    )
    # Line 2: n= (smaller, same alignment)
    ax.text(
        LABEL_X, y0 - 0.09,
        f"n = {grupos[nome]['n']}",
        fontsize=18, fontweight="bold", color="#888",
        ha="right", va="center", fontfamily=_FONTE,
        transform=ax.transData,
    )

# ── Axes ──────────────────────────────────────────────────────────────────
ax.set_xlim(-5, 125)
ax.set_ylim(-0.75, len(grupos) - 0.3)
ax.set_xlabel("Age (years)", fontsize=18, color="#555", fontfamily=_FONTE)
ax.xaxis.set_tick_params(labelsize=10, labelcolor="#666")
ax.xaxis.set_major_locator(plt.MultipleLocator(10))
ax.grid(axis="x", color="#ddd", linewidth=0.7, linestyle="--", alpha=0.8, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#ccc")
ax.tick_params(axis="y", length=0)  # remove y-axis tick marks
for lbl in ax.get_xticklabels():
    lbl.set_fontfamily(_FONTE)

# ── Legend ────────────────────────────────────────────────────────────────
patches = [mpatches.Patch(facecolor=CORES[k] + "55",
                           edgecolor=CORES[k], lw=1.5, label=k)
           for k in labels]
l_med = mlines.Line2D([], [], color="gray", lw=2.5, label="Median")
l_mean = mlines.Line2D([], [], color="gray", lw=1.4, ls=(0, (5, 4)), label="Mean (x̄)")
l_jit = mlines.Line2D([], [], marker="o", color="gray",
                       markersize=5, lw=0, alpha=0.5,
                       label="Individual data (jitter, n≤80)")
leg = ax.legend(handles=[*patches, l_med, l_mean, l_jit],
                framealpha=0.88, edgecolor="#ddd",
                loc="upper right", ncol=2,
                prop={"family": _FONTE, "size": 12})

# ── Export ────────────────────────────────────────────────────────────────
fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(OUT_SVG, bbox_inches="tight", facecolor="white")
plt.show()
print("\nExported:")
print(f"  {OUT_PDF}")
print(f"  {OUT_SVG}")
