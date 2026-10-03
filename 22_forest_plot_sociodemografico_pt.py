"""
Forest Plot — Razão de Chances Bruta e Ajustada, Variáveis Sociodemográficas
(versão em português).

OR bruto: regressão logística bivariada, Óbito ~ variável (cada categoria
vs. a referência do próprio grupo).
OR ajustado: a mesma regressão logística multivariável completa usada em
05b_modelo_ajustado_vacinado.py / 14_forest_plot_modelo_ajustado.py
(sociodemográficas + comorbidades + Vacinado + Dias_permanência), mostrando
aqui só as linhas sociodemográficas.

Nota: este script não é uma cópia de um script pré-existente no repositório
nem foi gerado a partir de um arquivo .docx externo (como
10_forest_plot_sociodemographic.py) — foi reconstruído diretamente de
New_pacientes703.xlsx. Os valores batem com precisão nos OR BRUTOS; os OR
AJUSTADOS têm a mesma ordem de grandeza e significância da versão original
em português que foi colada na conversa, mas podem diferir ligeiramente
pois o conjunto exato de covariáveis do modelo original não está disponível
neste ambiente.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
import statsmodels.api as sm
import statsmodels.formula.api as smf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "New_pacientes703.xlsx")


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

# ════════════════════════════════════════════════════════════
# 1. DADOS E MODELOS
# ════════════════════════════════════════════════════════════
df = pd.read_excel(DATA_PATH, sheet_name="Sheet1")
df["Sexo_fem"] = df["Sexo"]
df["EC_1"] = (df["Estado_Civil"] == 1).astype(int)
df["EC_2"] = (df["Estado_Civil"] == 2).astype(int)
bins = [-1, 17, 44, 60, 200]
faixa_labels = ["<18", "18-44", "45-60", ">60"]
df["faixa"] = pd.cut(df["Idade"], bins=bins, labels=faixa_labels)
df["F_1844"] = (df["faixa"] == "18-44").astype(int)
df["F_4560"] = (df["faixa"] == "45-60").astype(int)
df["F_60"] = (df["faixa"] == ">60").astype(int)
df["GI_1"] = (df["Grau de Instrução"] == 1).astype(float)
df["GI_2"] = (df["Grau de Instrução"] == 2).astype(float)
df.loc[df["Grau de Instrução"].isna(), ["GI_1", "GI_2"]] = np.nan

n_total = len(df)


def crude_or(var):
    sub = df[[var, "Óbito"]].dropna()
    m = smf.glm(f"Óbito ~ {var}", data=sub, family=sm.families.Binomial()).fit()
    b, p, ci = m.params[var], m.pvalues[var], m.conf_int().loc[var]
    return np.exp(b), np.exp(ci[0]), np.exp(ci[1]), p


formula_adj = (
    "Óbito ~ Sexo_fem + EC_1 + EC_2 + F_1844 + F_4560 + F_60 + GI_1 + GI_2 + Município + "
    "Vacinado + Prob_Card + CP + Diabetes + SRAG + Choques + Prob_neurol + Prob_Hemat + "
    "Cancer + Prob_Resp + Prob_Metab + Prob_TGI + Prob_Hep + Prob_Hid_Elet + Prob_AI_Infla + "
    "Febre + Outros + Traumatismo + COVID_CRÍTICA + Prob_Renal + LRA + Dias_permanência"
)
modelo_adj = smf.glm(formula_adj, data=df, family=sm.families.Binomial()).fit()
n_adj = int(modelo_adj.nobs)
or_adj = np.exp(modelo_adj.params)
ci_adj = modelo_adj.conf_int()


def adj_or(var):
    return or_adj[var], np.exp(ci_adj.loc[var, 0]), np.exp(ci_adj.loc[var, 1]), modelo_adj.pvalues[var]


def pct(n):
    return 100 * n / n_total


# ── Linhas do forest plot ──────────────────────────────────────────────────
# tipo: "header" | "ref" | "data" | "inestimavel"
LINHAS = [
    {"tipo": "header", "label": "Sexo"},
    {"tipo": "ref", "label": "Masculino", "n": int((df["Sexo"] == 0).sum())},
    {"tipo": "data", "label": "Feminino", "n": int((df["Sexo"] == 1).sum()),
     "var": "Sexo_fem"},

    {"tipo": "header", "label": "Estado civil"},
    {"tipo": "ref", "label": "Solteiro(a)", "n": int((df["Estado_Civil"] == 0).sum())},
    {"tipo": "data", "label": "Não solteiro(a)", "n": int((df["Estado_Civil"] == 1).sum()),
     "var": "EC_1"},
    {"tipo": "data", "label": "Não informado", "n": int((df["Estado_Civil"] == 2).sum()),
     "var": "EC_2"},

    {"tipo": "header", "label": "Faixa etária"},
    {"tipo": "ref", "label": "<18", "n": int((df["faixa"] == "<18").sum())},
    {"tipo": "data", "label": "18-44", "n": int((df["faixa"] == "18-44").sum()),
     "var": "F_1844"},
    {"tipo": "data", "label": "45-60", "n": int((df["faixa"] == "45-60").sum()),
     "var": "F_4560"},
    {"tipo": "data", "label": ">60", "n": int((df["faixa"] == ">60").sum()),
     "var": "F_60"},

    {"tipo": "header", "label": "Escolaridade"},
    {"tipo": "ref", "label": "Baixa escolaridade", "n": int((df["Grau de Instrução"] == 0).sum())},
    {"tipo": "data", "label": "Escolaridade média", "n": int((df["Grau de Instrução"] == 1).sum()),
     "var": "GI_1"},
    {"tipo": "data", "label": "Alta escolaridade", "n": int((df["Grau de Instrução"] == 2).sum()),
     "var": "GI_2"},

    {"tipo": "inestimavel", "label": "Estado / Estado de São Paulo", "n": int((df["UF"] == 0).sum())},
    {"tipo": "data", "label": "Município / Ribeirão Preto", "n": int((df["Município"] == 1).sum()),
     "var": "Município"},
]

for linha in LINHAS:
    if linha["tipo"] == "data":
        linha["crude"] = crude_or(linha["var"])
        linha["adj"] = adj_or(linha["var"])

# ════════════════════════════════════════════════════════════
# 2. FIGURA
# ════════════════════════════════════════════════════════════
BG = "#FFFFFF"
PANEL = "#F6F8FA"
BORDER = "#D0D7DE"
TEXT = "#1F2328"
SUBTEXT = "#57606A"
MUTED = "#8b949e"
GOLD = "#B08800"
COR_PROT = "#1D9E75"
COR_RISCO = "#E07B39"
COR_INEST = "#1a1a1a"


def sig_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


n_rows = len(LINHAS)
row_h = 0.72
fig_h = n_rows * row_h + 1.9
fig, axes = plt.subplots(
    1, 4, figsize=(20, fig_h), facecolor=BG,
    gridspec_kw={"width_ratios": [4.6, 4.6, 1.8, 4.6], "wspace": 0.04},
)
ax_labels, ax_c, ax_gap, ax_a = axes
ax_gap.set_visible(False)

for ax in (ax_labels, ax_c, ax_a):
    ax.set_ylim(n_rows - 0.5, -0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

# zebra stripe por bloco de header
stripe_on = False
for i, linha in enumerate(LINHAS):
    if linha["tipo"] == "header":
        stripe_on = not False  # headers always shaded
        cor_fundo = "#E8EDF2"
    else:
        cor_fundo = "#E8EDF2" if stripe_on else BG
    for ax in (ax_labels, ax_c, ax_a):
        ax.axhspan(i - 0.5, i + 0.5, color=cor_fundo, alpha=0.6 if linha["tipo"] != "header" else 0.9, zorder=0)
    if linha["tipo"] != "header":
        stripe_on = not stripe_on

ax_labels.set_xlim(0, 1)
for i, linha in enumerate(LINHAS):
    if linha["tipo"] == "header":
        ax_labels.text(0.0, i, linha["label"], color=TEXT, fontsize=17,
                       fontweight="bold", va="center", ha="left")
    else:
        fs = 12.5 if len(linha["label"]) > 20 else 15
        ax_labels.text(0.04, i, linha["label"], color=TEXT, fontsize=fs,
                       va="center", ha="left")
        n = linha["n"]
        ax_labels.text(0.98, i, f"n = {n} ({pct(n):.0f}%)", color=MUTED,
                       fontsize=12.5, style="italic", va="center", ha="right")

for ax, key, titulo, xlim, nrow in [
    (ax_c, "crude", "OR bruto (IC 95%)", (0.05, 150), n_total),
    (ax_a, "adj", "OR ajustado (IC 95%)", (0.05, 350), n_adj),
]:
    ax.set_facecolor(PANEL)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.xaxis.grid(False)
    ax.axvline(1.0, color=GOLD, linewidth=1.8, linestyle="--", zorder=2, alpha=0.9)
    ax.set_title(titulo, fontsize=17, fontweight="bold", color=TEXT, pad=10)
    ax.tick_params(axis="x", labelsize=12.5, colors=SUBTEXT)

    for i, linha in enumerate(LINHAS):
        if linha["tipo"] == "header":
            continue
        if linha["tipo"] == "ref":
            ax.text(0.5, i, "ref.", transform=ax.get_yaxis_transform(),
                    ha="center", va="center", fontsize=13, color=MUTED, style="italic")
            continue
        if linha["tipo"] == "inestimavel":
            ax.plot(1.0, i, marker="D", markersize=9, color=COR_INEST, zorder=5)
            ax.text(1.35, i, "—", transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=15, color=COR_INEST)
            continue

        OR, lo, hi, p = linha[key]
        cor = COR_PROT if OR < 1 else COR_RISCO
        sig = p < 0.05
        lo_plot = max(lo, xlim[0] * 1.05)
        hi_plot = min(hi, xlim[1] * 0.95)
        ax.plot([lo_plot, hi_plot], [i, i], color=cor, linewidth=3.0, zorder=3,
                alpha=0.85, solid_capstyle="round")
        ax.plot(OR, i, marker="D" if sig else "o", markersize=9.5 if sig else 8.5,
                color=cor, markerfacecolor=cor if sig else BG,
                markeredgecolor=cor, markeredgewidth=1.6, zorder=5)
        txt = f"{OR:.2f} ({lo:.2f}–{hi:.2f}){sig_stars(p)}"
        ax.text(1.0, i, txt, transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=13, color=TEXT, clip_on=False)

    ax.set_xlabel("Razão de Chances (escala log)", fontsize=13, color=SUBTEXT, labelpad=8)

# ── Legenda ───────────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(facecolor=COR_PROT, edgecolor=COR_PROT, label="Fator de proteção (OR < 1)"),
    mpatches.Patch(facecolor=COR_RISCO, edgecolor=COR_RISCO, label="Fator de risco (OR > 1)"),
    Line2D([0], [0], marker="D", color="none", markerfacecolor=TEXT, markeredgecolor=TEXT,
           markersize=6, label="Losango = p < 0,05"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=BG, markeredgecolor=TEXT,
           markeredgewidth=1.2, markersize=7, label="Círculo aberto = p ≥ 0,05"),
    Line2D([0], [0], color=GOLD, linewidth=1.4, linestyle="--", label="Linha de referência (OR = 1)"),
]
fig.legend(handles=legend_elements, fontsize=12.5, frameon=False, ncol=3,
           loc="upper center", bbox_to_anchor=(0.5, 1.0), labelcolor=TEXT,
           handletextpad=0.6, columnspacing=1.6)

fig.suptitle("Forest Plot — Razão de Chances Bruta e Ajustada",
             fontsize=23, fontweight="bold", color=TEXT, x=0.015, ha="left", y=1.11)
fig.text(0.015, 1.055, f"Variáveis sociodemográficas  |  n = {n_total}",
          ha="left", va="top", fontsize=15, color=SUBTEXT)

fig.subplots_adjust(top=0.88, bottom=0.075, left=0.01, right=0.99)

for ext, dpi in [("png", 180)]:
    out_path = os.path.join(BASE_DIR, f"forest_sociodemografico_pt.{ext}")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=BG)
    print("saved:", out_path)

out_tiff = os.path.join(BASE_DIR, "forest_sociodemografico_pt_600dpi.tiff")
fig.savefig(out_tiff, dpi=600, bbox_inches="tight", facecolor=BG, format="tiff", pil_kwargs={"compression": "tiff_lzw"})
print("saved:", out_tiff)

plt.close(fig)
