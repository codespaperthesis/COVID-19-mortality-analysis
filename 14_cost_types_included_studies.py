"""
Type of cost (direct, indirect, intangible) analysed in each study included in the systematic review.
Source: cost_types_included_studies.csv — one row per included study, classified from the
"Tipo de Custos (Direto/Indireto/Intangível)" column of Check_list_systematic_review_-_Price.xlsx
(original text kept in column tipo_custos_planilha). Edit the CSV to revise a classification.
  Sim     = cost type analysed
  Parcial = intangible only as a health outcome (QALY/DALY, EQ-5D) or mentioned without being valued
  Não     = not analysed
Generates Cost_types_PT.(png|tiff) and Cost_types_EN.(png|tiff).
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "cost_types_included_studies.csv")
DPI = 600
TITLE_CHARS = 62

TEXT, SUBTEXT, GRID, ABSENT = "#1F2328", "#57606A", "#D0D7DE", "#C9CCD1"
COMBO_BAR = "#6E7781"
TYPES = ["direto", "indireto", "intangivel"]
COLORS = {"direto": "#2a78d6", "indireto": "#eb6834", "intangivel": "#1baf7a"}

LANG = {
    "PT": dict(
        types={"direto": "Direto", "indireto": "Indireto", "intangivel": "Intangível"},
        country_col="pais",
        a="A  Tipo de custo analisado em cada estudo",
        b="B  Estudos por tipo de custo",
        c="C  Combinações de tipos de custo",
        xlabel="Nº de estudos",
        analysed="Analisado", partial="Parcial (QALY/DALY como desfecho ou citado sem valoração)",
        absent="Não analisado", partial_short="parcial",
        combos={"D": "Somente direto", "I": "Somente indireto", "DI": "Direto + indireto",
                "IT": "Indireto + intangível", "DIT": "Direto + indireto + intangível",
                "T": "Somente intangível", "DT": "Direto + intangível"},
        note="n = {n} estudos incluídos. Um estudo pode analisar mais de um tipo de custo.",
    ),
    "EN": dict(
        types={"direto": "Direct", "indireto": "Indirect", "intangivel": "Intangible"},
        country_col="country",
        a="A  Type of cost analysed in each study",
        b="B  Studies by type of cost",
        c="C  Combinations of cost types",
        xlabel="No. of studies",
        analysed="Analysed", partial="Partial (QALY/DALY as outcome or mentioned without valuation)",
        absent="Not analysed", partial_short="partial",
        combos={"D": "Direct only", "I": "Indirect only", "DI": "Direct + indirect",
                "IT": "Indirect + intangible", "DIT": "Direct + indirect + intangible",
                "T": "Intangible only", "DT": "Direct + intangible"},
        note="n = {n} included studies. A study may analyse more than one type of cost.",
    ),
}
COMBO_ORDER = ["DIT", "DI", "DT", "D", "IT", "I", "T"]
PT_TO_EN = {"Estados Unidos": "United States", "Reino Unido": "United Kingdom", "Alemanha": "Germany",
            "Austrália": "Australia", "França": "France", "Rússia": "Russia", "Bélgica": "Belgium",
            "Brasil": "Brazil", "Espanha": "Spain", "Israel": "Israel", "Itália": "Italy", "Japão": "Japan",
            "México": "Mexico", "Malásia": "Malaysia", "Taiwan": "Taiwan"}

# ── Data ────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
for t in TYPES:
    bad = set(df[t]) - {"Sim", "Parcial", "Não"}
    if bad:
        raise ValueError(f"Unexpected values in column {t}: {bad}")
df["country"] = df["pais"].apply(lambda s: " / ".join(PT_TO_EN[c.strip()] for c in s.split("/")))
df["combo"] = df.apply(lambda r: "".join(k for k, t in zip("DIT", TYPES) if r[t] == "Sim"), axis=1)
df = df.sort_values(["combo", "study_id"], key=lambda s: s.map(COMBO_ORDER.index) if s.name == "combo" else s)
df = df.reset_index(drop=True)


def short(title):
    return title if len(title) <= TITLE_CHARS else title[:TITLE_CHARS - 1].rstrip() + "…"


def style_bar_axis(ax, L, xmax):
    ax.set_xlim(0, xmax * 1.18)
    ax.set_xlabel(L["xlabel"], fontsize=8, color=SUBTEXT)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="y", labelsize=8.5, colors=TEXT, length=0)
    ax.tick_params(axis="x", labelsize=7.5, colors=SUBTEXT)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def draw(lang):
    L = LANG[lang]
    n = len(df)
    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], height_ratios=[1, 1.25], wspace=0.12, hspace=0.32)

    # A — study x cost-type matrix
    ax = fig.add_subplot(gs[:, 0])
    for j, t in enumerate(TYPES):
        for i, v in enumerate(df[t]):
            if v == "Sim":
                ax.scatter(j, i, s=95, color=COLORS[t], edgecolor="white", linewidth=1.2, zorder=3)
            elif v == "Parcial":
                ax.scatter(j, i, s=80, facecolor="white", edgecolor=COLORS[t], linewidth=1.6, zorder=3)
            else:
                ax.scatter(j, i, s=10, color=ABSENT, zorder=2)
    # separators between combination groups, with the group name on the right
    bounds = df.groupby("combo", sort=False).apply(lambda g: (g.index.min(), g.index.max()))
    for combo, (lo, hi) in bounds.items():
        if lo > 0:
            ax.axhline(lo - 0.5, color=GRID, lw=0.8)
        ax.text(2.55, (lo + hi) / 2, f"{L['combos'][combo]}\n(n = {hi - lo + 1})", fontsize=7.5,
                color=SUBTEXT, va="center", ha="left")
    labels = [f"{r.study_id}. {short(r.titulo)} ({getattr(r, L['country_col'])})" for r in df.itertuples()]
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(range(len(TYPES)))
    ax.set_xticklabels([L["types"][t] for t in TYPES], fontsize=9, color=TEXT, weight="bold")
    ax.xaxis.tick_top()
    ax.set_xlim(-0.5, 4.3)
    ax.tick_params(length=0, colors=TEXT)
    for s in ax.spines.values():
        s.set_visible(False)
    for j in range(len(TYPES)):
        ax.axvline(j, color=GRID, lw=0.5, zorder=1)
    ax.set_title(L["a"], fontsize=10.5, color=TEXT, weight="bold", loc="left", pad=24)

    # B — number of studies per cost type (analysed + partial)
    bx = fig.add_subplot(gs[0, 1])
    y = range(len(TYPES))[::-1]
    full = [int((df[t] == "Sim").sum()) for t in TYPES]
    part = [int((df[t] == "Parcial").sum()) for t in TYPES]
    for yy, t, f, p in zip(y, TYPES, full, part):
        bx.barh(yy, f, color=COLORS[t], height=0.6)
        if p:
            bx.barh(yy, p, left=f + 0.25, color="white", edgecolor=COLORS[t], linewidth=1.4,
                    height=0.6, hatch="////")
        txt = f"{f} ({100 * f / n:.0f}%)" + (f" + {p} {L['partial_short']}" if p else "")
        bx.text(f + p + (0.6 if p else 0.4), yy, txt, va="center", fontsize=8.5, color=TEXT)
    bx.set_yticks(list(y))
    bx.set_yticklabels([L["types"][t] for t in TYPES])
    style_bar_axis(bx, L, n)
    bx.set_title(L["b"], fontsize=10.5, color=TEXT, weight="bold", loc="left")

    # C — combinations
    cx = fig.add_subplot(gs[1, 1])
    counts = df["combo"].value_counts().reindex([c for c in COMBO_ORDER if c in set(df["combo"])])
    yc = range(len(counts))[::-1]
    for yy, (combo, k) in zip(yc, counts.items()):
        cx.barh(yy, k, color=COMBO_BAR, height=0.6)
        cx.text(k + 0.4, yy, f"{k} ({100 * k / n:.0f}%)", va="center", fontsize=8.5, color=TEXT)
    cx.set_yticks(list(yc))
    cx.set_yticklabels([L["combos"][c] for c in counts.index])
    style_bar_axis(cx, L, counts.max())
    cx.set_title(L["c"], fontsize=10.5, color=TEXT, weight="bold", loc="left")

    handles = [Line2D([], [], ls="", marker="o", ms=8, mfc=COLORS["direto"], mec="white", label=L["analysed"]),
               Line2D([], [], ls="", marker="o", ms=7.5, mfc="white", mec=COLORS["intangivel"], mew=1.6,
                      label=L["partial"]),
               Line2D([], [], ls="", marker="o", ms=3, mfc=ABSENT, mec=ABSENT, label=L["absent"])]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.03, 0.035), ncol=3, frameon=False,
               fontsize=8.5, labelcolor=TEXT, handletextpad=0.4, columnspacing=1.6)
    fig.text(0.03, 0.02, L["note"].format(n=n), fontsize=8, color=SUBTEXT)

    for ext in ("png", "tiff"):
        out = os.path.join(BASE_DIR, f"Cost_types_{lang}.{ext}")
        kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tiff" else {}
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white", **kw)
        print("saved", out)
    plt.close(fig)


if __name__ == "__main__":
    print(df.groupby("combo", sort=False).size().to_string())
    for t in TYPES:
        print(t, df[t].value_counts().to_dict())
    for lang in LANG:
        draw(lang)
