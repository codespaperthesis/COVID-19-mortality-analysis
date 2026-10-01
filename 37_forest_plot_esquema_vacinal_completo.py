import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import statsmodels.api as sm
import statsmodels.formula.api as smf
from PIL import Image

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────
# Forest plot: esquema vacinal completo vs. incompleto (referência),
# em relação ao risco de óbito e de alta hospitalar.
#
# Completo   = 2 doses de Pfizer/AstraZeneca/Coronavac (ou qualquer
#              vacina de 2 doses), OU 1 dose de Janssen (dose única).
# Incompleto = menos doses que o necessário para completar o esquema
#              acima — é o grupo de REFERÊNCIA (OR=1) desta análise.
# Não vacinados são excluídos da comparação.
#
# Fonte: 703pacientes.xlsx (doses em "Vacinas", fabricante em
# "Fabricante_", usado só para identificar o caso de dose única).
# ─────────────────────────────────────────────────────────────────────────
XLSX_PATH = "703pacientes.xlsx"
OUTPUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "forest_plot_esquema_vacinal_completo.png")
OUTPUT_TIFF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "forest_plot_esquema_vacinal_completo.tiff")

# Conjunto de ajuste reduzido: a amostra desta comparação é pequena
# (n=95, Completo=75 vs Incompleto=20; ~41 óbitos), insuficiente para os
# ~28 covariáveis usadas nos outros forest plots deste projeto — nesse
# caso a regressão sofre separação quase perfeita (OR/IC absurdos,
# p≈1). Mantém-se aqui só os confundidores clínicos mais relevantes.
ADJUST_VARS = ["Sexo", "Idade", "COVID_CRÍTICA", "CP"]


def classifica_esquema(vacinas, fabricante, vacinado):
    """Completo = >=2 doses, ou 1 dose exclusivamente de Janssen."""
    if vacinado == 0 or vacinas == 0:
        return "Não vacinado"
    janssen_unico = isinstance(fabricante, str) and fabricante.strip() == "Janssen"
    completo = (vacinas >= 2) or (vacinas == 1 and janssen_unico)
    return "Completo" if completo else "Incompleto"


# ════════════════════════════════════════════════════════════
# 1. CARREGAR DADOS E DEFINIR EXPOSIÇÃO (referência = esquema incompleto)
# ════════════════════════════════════════════════════════════
dados_todos = pd.read_excel(XLSX_PATH)
dados_todos["Esquema"] = [
    classifica_esquema(v, f, vc) for v, f, vc in
    zip(dados_todos["Vacinas"], dados_todos["Fabricante_"], dados_todos["Vacinado"])
]
dados_todos["Alta"] = 1 - dados_todos["Óbito"]

n_total = len(dados_todos)
n_nao_vac = int((dados_todos["Esquema"] == "Não vacinado").sum())

dados = dados_todos[dados_todos["Esquema"].isin(["Completo", "Incompleto"])].copy()
dados["Completo_bin"] = (dados["Esquema"] == "Completo").astype(int)

n_analisado = len(dados)
n_completo = int(dados["Completo_bin"].sum())
n_incompleto = n_analisado - n_completo
n_obitos = int(dados["Óbito"].sum())
n_altas = int(dados["Alta"].sum())


def fit_or(formula, var, data):
    modelo = smf.glm(formula=formula, data=data,
                      family=sm.families.Binomial()).fit()
    beta = modelo.params[var]
    p = modelo.pvalues[var]
    lo, hi = modelo.conf_int().loc[var]
    return np.exp(beta), np.exp(lo), np.exp(hi), p


linhas = []
for desfecho, label in [("Óbito", "Óbito hospitalar"), ("Alta", "Alta hospitalar")]:
    n_evento = int(dados[desfecho].sum())
    n_evento_completo = int(dados.loc[dados["Completo_bin"] == 1, desfecho].sum())

    OR, lo, hi, p = fit_or(f"{desfecho} ~ Completo_bin", "Completo_bin", dados)
    formula_adj = f"{desfecho} ~ Completo_bin + " + " + ".join(ADJUST_VARS)
    ORa, loa, hia, pa = fit_or(formula_adj, "Completo_bin", dados)

    linhas.append({
        "label": label, "n_geral": n_evento, "n_completo_evento": n_evento_completo,
        "OR": OR, "IC_inf": lo, "IC_sup": hi, "p_OR": p,
        "ORa": ORa, "ICa_inf": loa, "ICa_sup": hia, "p_ORa": pa,
        # para Óbito, favorável = OR<1 (menos óbito); para Alta, favorável
        # = OR>1 (mais alta) — evita colorir "Alta" como se fosse risco
        "favoravel_quando_maior": desfecho == "Alta",
    })

df_raw = pd.DataFrame(linhas)

print("=" * 70)
print(f"Tabela lida de: {XLSX_PATH}")
print(f"n total = {n_total} | não vacinados excluídos = {n_nao_vac} | "
      f"analisados = {n_analisado} (Completo={n_completo}, "
      f"Incompleto={n_incompleto})")
print(df_raw.to_string())
print("=" * 70)

# ════════════════════════════════════════════════════════════
# 2. FOREST PLOT
# ════════════════════════════════════════════════════════════
BG = "#FFFFFF"
PANEL = "#F6F8FA"
BORDER = "#D0D7DE"
TEXT = "#1F2328"
SUBTEXT = "#57606A"
GOLD = "#B08800"
COR_PROT = "#1D9E75"
COR_RISCO = "#E07B39"


def sig_stars(p):
    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


n_rows = len(df_raw)
fig_h = max(4.5, n_rows * 0.9 + 3.0)
fig, axes = plt.subplots(
    1, 4, figsize=(16, fig_h), facecolor=BG,
    gridspec_kw={"width_ratios": [2.5, 5, 2.5, 5], "wspace": 0.04},
)
ax_labels, ax_or, ax_gap, ax_ora = axes
ax_gap.set_visible(False)

ax_labels.set_facecolor(BG)
ax_labels.set_xlim(0, 1)
ax_labels.set_ylim(n_rows - 0.5, -0.5)
for spine in ax_labels.spines.values():
    spine.set_visible(False)
ax_labels.set_xticks([])
ax_labels.set_yticks([])

for ax in (ax_labels, ax_or, ax_ora):
    for i in range(n_rows):
        bg_col = "#E8EDF2" if i % 2 == 0 else PANEL
        ax.axhspan(i - 0.42, i + 0.42, color=bg_col, alpha=0.55, zorder=0)

for i, row in df_raw.iterrows():
    ax_labels.text(0.98, i, f"{row['label']} (n={row['n_geral']})",
                   color=TEXT, fontsize=16, va="center", ha="right")


def draw_panel(ax, col_or, col_lo, col_hi, col_p, title, xlim):
    ax.set_facecolor(PANEL)
    ax.set_xscale("log")
    ax.xaxis.grid(True, color=BORDER, linewidth=0.9, zorder=0, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(0.8)
    ax.axvline(1.0, color=GOLD, linewidth=2.5, linestyle="--", zorder=2, alpha=0.9)

    for i, row in df_raw.iterrows():
        OR_ = row[col_or]
        lo_ = row[col_lo]
        hi_ = row[col_hi]
        p_ = row[col_p]

        if (pd.isna(OR_) or pd.isna(lo_) or pd.isna(hi_)
                or not np.isfinite(OR_) or OR_ < 1e-6):
            ax.text(0.5, i, "—", color=SUBTEXT, fontsize=13,
                    va="center", ha="center",
                    transform=ax.get_yaxis_transform())
            continue

        favoravel = (OR_ > 1) if row["favoravel_quando_maior"] else (OR_ < 1)
        cor = COR_PROT if favoravel else COR_RISCO
        sig = (p_ is not None) and not np.isnan(p_) and (p_ < 0.05)

        lo_plot = max(lo_, xlim[0] * 1.02)
        hi_plot = min(hi_, xlim[1] * 0.98) if np.isfinite(hi_) else xlim[1] * 0.98
        ax.plot([lo_plot, hi_plot], [i, i],
                color=cor, linewidth=3.8, zorder=3, alpha=0.85,
                solid_capstyle="round")

        ax.plot(OR_, i, marker="D" if sig else "o", markersize=9 if sig else 7,
                color=cor, markerfacecolor=cor if sig else BG,
                markeredgecolor=cor, markeredgewidth=1.6, zorder=5)

        hi_txt = min(hi_, 9999) if np.isfinite(hi_) else float("inf")
        hi_str = f"{hi_txt:.2f}" if np.isfinite(hi_txt) else "∞"
        txt = f"{OR_:.2f} ({lo_:.2f}–{hi_str}){sig_stars(p_)}"
        ax.text(1.02, i, txt, transform=ax.get_yaxis_transform(),
                color=TEXT, fontsize=15, va="center", ha="left", clip_on=False)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([""] * n_rows)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=SUBTEXT, labelsize=14)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xlim(*xlim)
    ax.set_xlabel("Odds Ratio (escala log)", fontsize=20, color=SUBTEXT, labelpad=6)
    ax.set_title(title, fontsize=20, fontweight="bold", color=TEXT, pad=5)


draw_panel(ax_or, "OR", "IC_inf", "IC_sup", "p_OR",
           "OR bruto (IC 95%)", xlim=(0.02, 30))
draw_panel(ax_ora, "ORa", "ICa_inf", "ICa_sup", "p_ORa",
           "OR ajustado (IC 95%)", xlim=(0.02, 30))

legend_elements = [
    mpatches.Patch(facecolor=COR_PROT, edgecolor=COR_PROT,
                   label="Favorável (menos óbito / mais alta)"),
    mpatches.Patch(facecolor=COR_RISCO, edgecolor=COR_RISCO,
                   label="Desfavorável (mais óbito / menos alta)"),
    Line2D([0], [0], marker="D", color="none", markerfacecolor=TEXT,
           markeredgecolor=TEXT, markersize=5, label="Losango = p < 0,05"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=BG,
           markeredgecolor=TEXT, markeredgewidth=1.2, markersize=6,
           label="Círculo aberto = p ≥ 0,05"),
    Line2D([0], [0], color=GOLD, linewidth=1.2, linestyle="--",
           label="Linha de referência (OR = 1)"),
]
ax_or.legend(handles=legend_elements, fontsize=8, frameon=True, edgecolor=BORDER,
             facecolor=BG, labelcolor=TEXT, loc="lower left", framealpha=0.97,
             borderpad=0.9, handlelength=0.5)

fig.text(0.50, 1.14,
          "Forest Plot — Esquema Vacinal Completo vs. Incompleto",
          ha="center", va="top", fontsize=30, fontweight="bold", color=TEXT)
subtitle = (f"Referência: esquema incompleto | Completo = 2 doses "
            f"(Pfizer/AstraZeneca/Coronavac) ou 1 dose (Janssen) | "
            f"n analisado = {n_analisado} (Completo={n_completo}, "
            f"Incompleto={n_incompleto}) | não vacinados excluídos "
            f"(n={n_nao_vac})")
fig.text(0.50, 1.02, subtitle, ha="center", va="top", fontsize=17, color=SUBTEXT)
fig.add_artist(plt.Line2D([0.13, 0.97], [0.96, 0.96], transform=fig.transFigure,
                           color=BORDER, linewidth=1.8))
fig.text(0.03, -0.14,
          "*** p<0,001 ** p<0,01 * p<0,05 | OR = Odds Ratio; IC = Intervalo "
          "de Confiança de 95% | Referência (OR=1) = esquema vacinal "
          "incompleto | Não vacinados excluídos desta comparação",
          color=SUBTEXT, fontsize=11.5, style="italic")
fig.text(0.03, -0.20,
          "Ajustado por sexo, idade, COVID crítica e nº de comorbidades (CP) "
          "— conjunto reduzido de covariáveis, já que a amostra (n=95) não "
          "comporta o conjunto completo de ajuste usado em outros forest "
          "plots deste projeto sem causar separação quase perfeita | "
          "Fonte: 703pacientes.xlsx",
          color=SUBTEXT, fontsize=11.5, style="italic")

plt.tight_layout(rect=[0, 0.03, 1, 1.94])
plt.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight", facecolor=BG)

_buf_png = OUTPUT_PNG.replace(".png", "_300dpi_tmp.png")
plt.savefig(_buf_png, dpi=300, bbox_inches="tight", facecolor=BG)
Image.open(_buf_png).save(OUTPUT_TIFF, dpi=(300, 300), compression="tiff_lzw")
os.remove(_buf_png)

print(f"Gráfico salvo em: {OUTPUT_PNG} e {OUTPUT_TIFF}")
plt.show()
