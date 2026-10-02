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
# Forest plot: esquema vacinal incompleto vs. completo (referência), em
# relação ao risco de óbito hospitalar. "Óbito" é a única coluna de
# desfecho na planilha (0 = alta, 1 = óbito) — alta é só o complemento
# de óbito, então uma única linha de forest plot já descreve os dois
# (OR para alta seria exatamente 1/OR para óbito, redundante).
#
# Grupos lidos diretamente das colunas já classificadas na planilha:
#   Ecompleto   = 1 → esquema vacinal completo — grupo de REFERÊNCIA
#                 (OR=1) desta análise.
#   Eincompleto = 1 → esquema vacinal incompleto — grupo de exposição,
#                 cuja associação com o óbito é o que está sendo testado.
# As duas colunas são mutuamente exclusivas e cobrem os 703 pacientes.
#
# Fonte: 703pacientes.xlsx.
# ─────────────────────────────────────────────────────────────────────────
XLSX_PATH = "703pacientes.xlsx"
OUTPUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "forest_plot_esquema_vacinal_completo.png")
OUTPUT_TIFF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "forest_plot_esquema_vacinal_completo.tiff")

# Conjunto de ajuste padrão do projeto (n=703 comporta as mesmas
# covariáveis usadas nos outros forest plots, sem separação quase
# perfeita).
ADJUST_VARS = [
    "Sexo", "Prob_Card", "CP", "Diabetes", "SRAG", "Choques", "Prob_neurol",
    "Prob_Hemat", "Cancer", "Prob_Resp", "Prob_Metab", "Prob_TGI",
    "Prob_Hep", "Prob_Hid_Elet", "Prob_AI_Infla", "Febre", "Outros",
    "Traumatismo", "COVID_CRÍTICA", "Prob_Renal", "LRA", "Dias_permanência",
    "Estado_Civil_1", "Estado_Civil_2", "Idade_cat_1", "Idade_cat_2",
    "Idade_cat_3", "Grau_Instrucao_1", "Grau_Instrucao_2",
]

# ════════════════════════════════════════════════════════════
# 1. CARREGAR DADOS — grupos já vêm prontos em Ecompleto/Eincompleto
# ════════════════════════════════════════════════════════════
dados_todos = pd.read_excel(XLSX_PATH)
n_total = len(dados_todos)

assert ((dados_todos["Ecompleto"] + dados_todos["Eincompleto"]) == 1).all(), (
    "Ecompleto e Eincompleto deveriam ser mutuamente exclusivos e cobrir "
    "todos os pacientes (soma = 1 em cada linha)")

dados = dados_todos.copy()
# Variável de exposição = esquema INCOMPLETO (Eincompleto=1); referência
# (0) = esquema completo (Ecompleto=1), conforme pedido.
dados["Incompleto_bin"] = dados["Eincompleto"].astype(int)
n_completo = int(dados["Ecompleto"].sum())
n_incompleto = int(dados["Eincompleto"].sum())
n_obitos = int(dados["Óbito"].sum())


def fit_or(formula, var, data):
    modelo = smf.glm(formula=formula, data=data,
                      family=sm.families.Binomial()).fit()
    beta = modelo.params[var]
    p = modelo.pvalues[var]
    lo, hi = modelo.conf_int().loc[var]
    return np.exp(beta), np.exp(lo), np.exp(hi), p


# Óbito e Alta são o mesmo desfecho binário (0/1 na coluna "Óbito"; Alta
# é só o complemento) — uma única linha de forest plot já descreve os
# dois (OR para Alta seria exatamente 1/OR para Óbito, redundante).
n_evento = int(dados["Óbito"].sum())
n_evento_incompleto = int(dados.loc[dados["Incompleto_bin"] == 1, "Óbito"].sum())

OR, lo, hi, p = fit_or("Óbito ~ Incompleto_bin", "Incompleto_bin", dados)
formula_adj = "Óbito ~ Incompleto_bin + " + " + ".join(ADJUST_VARS)
ORa, loa, hia, pa = fit_or(formula_adj, "Incompleto_bin", dados)

df_raw = pd.DataFrame([{
    "label": "Óbito hospitalar", "n_geral": n_evento,
    "n_incompleto_evento": n_evento_incompleto,
    "OR": OR, "IC_inf": lo, "IC_sup": hi, "p_OR": p,
    "ORa": ORa, "ICa_inf": loa, "ICa_sup": hia, "p_ORa": pa,
}])

print("=" * 70)
print(f"Tabela lida de: {XLSX_PATH}")
print(f"n total = {n_total} (Ecompleto={n_completo}, "
      f"Eincompleto={n_incompleto}) | Óbitos={n_obitos}")
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
POL_CABECALHO = 3.0  # título + subtítulo + legenda, em polegadas
POL_RODAPE = 0.85    # rótulo do eixo x
altura_conteudo = n_rows * 0.9 + 1.2
fig_h = POL_CABECALHO + altura_conteudo + POL_RODAPE
top_frac = 1 - POL_CABECALHO / fig_h
bottom_frac = POL_RODAPE / fig_h

fig, axes = plt.subplots(
    1, 4, figsize=(16, fig_h), facecolor=BG,
    gridspec_kw={"width_ratios": [2.5, 5, 2.5, 5], "wspace": 0.04,
                 "top": top_frac, "bottom": bottom_frac},
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

        cor = COR_PROT if OR_ < 1 else COR_RISCO
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
           "OR bruto (IC 95%)", xlim=(0.03, 12))
draw_panel(ax_ora, "ORa", "ICa_inf", "ICa_sup", "p_ORa",
           "OR ajustado (IC 95%)", xlim=(0.03, 12))

legend_elements = [
    mpatches.Patch(facecolor=COR_PROT, edgecolor=COR_PROT,
                   label="Fator protetor (OR < 1, menos óbito)"),
    mpatches.Patch(facecolor=COR_RISCO, edgecolor=COR_RISCO,
                   label="Fator de risco (OR > 1, mais óbito)"),
    Line2D([0], [0], marker="D", color="none", markerfacecolor=TEXT,
           markeredgecolor=TEXT, markersize=5, label="Losango = p < 0,05"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=BG,
           markeredgecolor=TEXT, markeredgewidth=1.2, markersize=6,
           label="Círculo aberto = p ≥ 0,05"),
    Line2D([0], [0], color=GOLD, linewidth=1.2, linestyle="--",
           label="Linha de referência (OR = 1)"),
]
# Posições em fração de figura, mas definidas a partir de polegadas
# contadas do topo (1 - polegadas/fig_h) — mais previsível que valores
# fixos de fração, pois se adapta à altura real da figura.
y_titulo = 1 - 0.42 / fig_h
y_subtitulo = 1 - 1.05 / fig_h
y_divisor = 1 - 1.35 / fig_h
y_legenda = 1 - 1.80 / fig_h

fig.text(0.50, y_titulo,
          "Forest Plot — Esquema Vacinal Incompleto vs. Completo",
          ha="center", va="top", fontsize=30, fontweight="bold", color=TEXT)
subtitle = (f"Referência: esquema completo | Exposição: esquema "
            f"incompleto | n = {n_total} (Completo={n_completo}, "
            f"Incompleto={n_incompleto})")
fig.text(0.50, y_subtitulo, subtitle, ha="center", va="top", fontsize=17,
          color=SUBTEXT)
fig.add_artist(plt.Line2D([0.13, 0.97], [y_divisor, y_divisor],
                           transform=fig.transFigure, color=BORDER, linewidth=1.8))

fig.legend(handles=legend_elements, loc="upper center",
           bbox_to_anchor=(0.5, y_legenda), ncol=5, frameon=False,
           fontsize=11.5, labelcolor=TEXT, columnspacing=1.4, handlelength=1.3)

plt.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight", facecolor=BG)

_buf_png600 = OUTPUT_PNG.replace(".png", "_600dpi_tmp.png")
plt.savefig(_buf_png600, dpi=600, bbox_inches="tight", facecolor=BG)
OUTPUT_TIFF_600 = OUTPUT_TIFF.replace(".tiff", "_600dpi.tiff")
Image.open(_buf_png600).save(OUTPUT_TIFF_600, dpi=(600, 600), compression="tiff_lzw")
os.remove(_buf_png600)

_buf_png = OUTPUT_PNG.replace(".png", "_300dpi_tmp.png")
plt.savefig(_buf_png, dpi=300, bbox_inches="tight", facecolor=BG)
Image.open(_buf_png).save(OUTPUT_TIFF, dpi=(300, 300), compression="tiff_lzw")
os.remove(_buf_png)

print(f"Gráfico salvo em: {OUTPUT_PNG}, {OUTPUT_TIFF} e {OUTPUT_TIFF_600}")
plt.show()
