#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Painel de exames HEMATOLOGICOS e GASOMETRICOS (arterial/venosa) por desfecho -
mediana por paciente unico, apresentacao em box plot, com janela de
referencia clinica sombreada, no mesmo molde do painel bioquimico ja
validado (painel_exames_bioquimicos_medianapaciente_boxplot_enxuto_pt).

Fonte dos dados: ExamesB_filtradaa.xlsx (mesma subamostra de 508 pacientes
ja usada nos paineis bioquimicos), com mediana calculada por paciente
(nao por registro/exame), depois resumida por grupo (Alta/Obito).

Notas metodologicas importantes:
- Duas fontes de hemograma coexistem na base bruta ("HEMOGRAMA COMPLETO
  (T+E)" e "HEMOGRAMA COMPLETO"), com nomenclaturas de DESCRICAO diferentes
  para o mesmo exame; ambas foram combinadas por variavel.
- Neutrofilos segmentados: os valores brutos desta variavel estao em
  PERCENTUAL nas duas fontes, mas a unica referencia disponivel na
  planilha e em contagem absoluta (10^3/uL) - unidades incompativeis.
  NAO se aplica referencia a essa variavel (nao inventada).
- Linfocitos: unidade exibida corrigida para "%" (o rotulo "x10^3/uL"
  estava errado); valores e janela de referencia mantidos.
- "GASOMETRIA" sem rotulo foi tratada como arterial (decisao acordada
  anteriormente); GASOMETRIA VENOSA + GASOMETRIA VENOSA CENTRAL combinadas
  como venosa.
- Sem referencia disponivel na planilha-fonte: FiO2, P50c e Temperatura
  (arterial e venosa) - painel sem linha/janela de referencia para essas.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pickle
import os

COLOR_RISK       = "#E07B39"
COLOR_PROTECTIVE = "#1D9E75"
COLOR_REF_M      = "#3B6FA6"
COLOR_REF_F      = "#B23B8C"
COLOR_REF_GERAL  = "#9C9C9C"
COLOR_TEXT_DARK  = "#3A3F44"
COLOR_TEXT_MUTED = "#8A94A0"
COLOR_GROUP_BG   = "#DCE3EA"
COLOR_AXIS       = "#5B6470"

plt.rcParams["font.family"] = "DejaVu Sans"

# arquivos gerados por painel_hemato_gasometria_dados.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "hemato_gaso_stats.pkl"), "rb") as f:
    STATS = pickle.load(f)
with open(os.path.join(BASE_DIR, "hemato_gaso_raw.pkl"), "rb") as f:
    RAW = pickle.load(f)
with open(os.path.join(BASE_DIR, "hemato_gaso_refs.pkl"), "rb") as f:
    REFS = pickle.load(f)

# nome interno -> (rotulo exibido)
LABELS = {
    "Plaquetas": ("Plaquetas", "x10³/µL"),
    "Glóbulos brancos": ("Glóbulos brancos", "x10³/µL"),
    "Glóbulos vermelhos": ("Glóbulos vermelhos", "x10⁶/µL"),
    "Hematócrito": ("Hematócrito", "%"),
    "RDW": ("RDW", "%"),
    "Monócitos": ("Monócitos", "x10³/µL"),
    "Neutrófilos segment.": ("Neutrófilos segment.", "%"),
    "Linfócitos": ("Linfócitos", "%"),
    "Basófilos": ("Basófilos", "x10³/µL"),
    "Eosinófilos": ("Eosinófilos", "x10³/µL"),
    "CHCM": ("CHCM", "g/dL"),
    "HCM": ("HCM", "pg"),
    "PCO2_art": ("PCO2", "mmHg"),
    "BE_art": ("BE", "mEq/L"),
    "HCO3_art": ("HCO3", "mEq/L"),
    "FiO2_art": ("FiO2", "%"),
    "P50c_art": ("P50c", "mmHg"),
    "SaO2_art": ("SaO2", "%"),
    "PO2_art": ("PO2", "mmHg"),
    "pH_art": ("pH", ""),
    "Temperatura_art": ("Temperatura", "°C"),
    "BE_ven": ("BE", "mEq/L"),
    "PCO2_ven": ("PCO2", "mmHg"),
    "FiO2_ven": ("FiO2", "%"),
    "PO2_ven": ("PO2", "mmHg"),
    "P50c_ven": ("P50c", "mmHg"),
    "HCO3_ven": ("HCO3", "mEq/L"),
    "SvO2_ven": ("SvO2", "%"),
    "Temperatura_ven": ("Temperatura", "°C"),
    "pH_ven": ("pH", ""),
}

DOMAINS = [
    ("Hematológicos", ["Plaquetas", "Glóbulos brancos", "Glóbulos vermelhos", "Hematócrito",
                        "RDW", "Monócitos", "Neutrófilos segment.", "Linfócitos",
                        "Basófilos", "Eosinófilos", "CHCM", "HCM"]),
    ("Gasometria arterial", ["PCO2_art", "BE_art", "HCO3_art", "FiO2_art", "P50c_art",
                              "SaO2_art", "PO2_art", "pH_art", "Temperatura_art"]),
    ("Gasometria venosa", ["BE_ven", "PCO2_ven", "FiO2_ven", "PO2_ven", "P50c_ven",
                            "HCO3_ven", "SvO2_ven", "Temperatura_ven", "pH_ven"]),
]

# apenas variaveis significativas (p < 0,05)
DOMAINS = [(dom, [v for v in names if STATS[v]["p"] < 0.05]) for dom, names in DOMAINS]


def get_ref_windows(var):
    entries = REFS.get(var, [])
    windows = []
    for sexo, lo, hi in entries:
        color = COLOR_REF_GERAL
        if sexo == "M":
            color = COLOR_REF_M
        elif sexo == "F":
            color = COLOR_REF_F
        windows.append((lo, hi, color))
    return windows


def fmt_p(p):
    if p is None or np.isnan(p):
        return "n/a"
    return "p<0,001" if p < 0.001 else f"p={p:.3f}".replace(".", ",")


def draw_panel(fig, ax_left, ax_top_in, ax_width, ax_height_in, fig_h, var):
    s = STATS[var]
    ax = fig.add_axes([ax_left, (ax_top_in - ax_height_in) / fig_h, ax_width, ax_height_in / fig_h])

    windows = get_ref_windows(var)
    raw = RAW.get(var, {"alta": np.array([]), "obito": np.array([])})

    vals = []
    for k in ("med_alta", "q1_alta", "q3_alta", "med_obito", "q1_obito", "q3_obito"):
        v = s[k]
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            vals.append(v)
    for arr in (raw["alta"], raw["obito"]):
        if len(arr):
            vals.append(np.percentile(arr, 2))
            vals.append(np.percentile(arr, 90))
    for lo, hi, _ in windows:
        if lo is not None:
            vals.append(lo)
        if hi is not None:
            vals.append(hi)

    if not vals:
        vals = [0, 1]
    vmin, vmax = min(vals), max(vals)
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5
    pad = (vmax - vmin) * 0.12
    xlim = (vmin - pad, vmax + pad)

    for lo, hi, color in windows:
        lo_ = lo if lo is not None else xlim[0]
        hi_ = hi if hi is not None else xlim[1]
        ax.axvspan(lo_, hi_, color=color, alpha=0.16, zorder=1, linewidth=0)
        if lo is not None:
            ax.axvline(lo, color=color, linewidth=1.7, linestyle=(0, (4, 2)), zorder=2, alpha=0.85)
        if hi is not None:
            ax.axvline(hi, color=color, linewidth=1.7, linestyle=(0, (4, 2)), zorder=2, alpha=0.85)

    box_data, positions, colors = [], [], []
    if len(raw["alta"]):
        box_data.append(raw["alta"]); positions.append(0); colors.append(COLOR_PROTECTIVE)
    if len(raw["obito"]):
        box_data.append(raw["obito"]); positions.append(1); colors.append(COLOR_RISK)
    if box_data:
        bp = ax.boxplot(box_data, positions=positions, vert=False, widths=0.55,
                         patch_artist=True, showfliers=True, zorder=4,
                         flierprops=dict(marker="o", markersize=4.5, markerfacecolor="none",
                                          alpha=0.5, markeredgewidth=0.6),
                         medianprops=dict(color="white", linewidth=2.4),
                         whiskerprops=dict(linewidth=1.7), capprops=dict(linewidth=1.7))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
            patch.set_edgecolor(color)
        for i, color in enumerate(colors):
            bp["whiskers"][2*i].set_color(color)
            bp["whiskers"][2*i+1].set_color(color)
            bp["caps"][2*i].set_color(color)
            bp["caps"][2*i+1].set_color(color)
            bp["fliers"][i].set_markeredgecolor(color)

    ax.set_xlim(xlim)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10], min_n_ticks=3))
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Alta", "Óbito"], fontsize=14.5, fontweight="bold")
    ax.get_yticklabels()[0].set_color(COLOR_PROTECTIVE)
    ax.get_yticklabels()[1].set_color(COLOR_RISK)
    ax.tick_params(axis="x", labelsize=11.5, colors=COLOR_AXIS)
    ax.tick_params(axis="y", length=0)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(COLOR_AXIS)

    p = s["p"]
    sig = (p is not None) and (not np.isnan(p)) and p < 0.05
    name, unit = LABELS.get(var, (var, ""))
    ax.set_title(name, fontsize=15.5, fontweight="bold", color=COLOR_TEXT_DARK, pad=26)
    if unit:
        ax.text(0.5, 1.46, unit, transform=ax.transAxes, fontsize=11.5,
                color=COLOR_TEXT_MUTED, ha="center", va="bottom", style="italic")
    ax.text(0.5, 1.71, fmt_p(p), transform=ax.transAxes, fontsize=13.5,
            fontweight="bold" if sig else "normal",
            color=(COLOR_TEXT_DARK if sig else COLOR_TEXT_MUTED), ha="center", va="bottom")
    ax.text(0.0, -0.30, f"Alta n={s['n_alta']}", transform=ax.transAxes, fontsize=11.0,
            color=COLOR_PROTECTIVE, ha="left", va="top", style="italic", fontweight="bold")
    ax.text(1.0, -0.30, f"Óbito n={s['n_obito']}", transform=ax.transAxes, fontsize=11.0,
            color=COLOR_RISK, ha="right", va="top", style="italic", fontweight="bold")


def build_figure():
    cols_per_domain = 4

    def rows_for(names):
        return int(np.ceil(len(names) / cols_per_domain))

    left_margin = 0.028
    right_margin = 0.012
    cell_w = (1.0 - left_margin - right_margin) / cols_per_domain

    panel_h = 4.35
    header_h = 0.50
    section_gap = 0.34
    fig_w = 24.0
    top_used = 1.85
    bottom_margin = 0.35

    n_rows_total = sum(rows_for(names) for _, names in DOMAINS)
    fig_h = (n_rows_total * panel_h + len(DOMAINS) * header_h
             + (len(DOMAINS) - 1) * section_gap + top_used + bottom_margin)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
    fig.patch.set_facecolor("white")

    title = "Exames Hematológicos e Gasométricos por Desfecho — Mediana por Paciente — Box Plot — Variáveis Significativas"
    fig.text(0.008, 1 - 0.42 / fig_h, title, fontsize=25, fontweight="bold",
              color=COLOR_TEXT_DARK, va="top", ha="left")
    subtitle = "Box plot por paciente, por Alta vs. Óbito hospitalar  |  HCFMRP-USP (subamostra n=508)"
    fig.text(0.008, 1 - 0.85 / fig_h, subtitle, fontsize=15.5, color=COLOR_TEXT_MUTED, va="top", ha="left")

    legend_handles = [
        mpatches.Patch(facecolor=COLOR_PROTECTIVE, alpha=0.85, edgecolor=COLOR_PROTECTIVE, label="Alta — box plot"),
        mpatches.Patch(facecolor=COLOR_RISK, alpha=0.85, edgecolor=COLOR_RISK, label="Óbito — box plot"),
        mpatches.Patch(color=COLOR_REF_GERAL, alpha=0.35, label="Janela de referência (geral)"),
        mpatches.Patch(color=COLOR_REF_M, alpha=0.35, label="Janela de referência (masc.)"),
        mpatches.Patch(color=COLOR_REF_F, alpha=0.35, label="Janela de referência (fem.)"),
    ]
    fig.legend(handles=legend_handles, loc="upper left",
               bbox_to_anchor=(0.008, 1 - 1.20 / fig_h),
               ncol=5, frameon=False, fontsize=13.5, columnspacing=1.3, handletextpad=0.5)

    y_cursor_in = fig_h - top_used

    for dom_idx, (dom_name, names) in enumerate(DOMAINS):
        if dom_idx > 0:
            y_cursor_in -= section_gap

        head_y0 = y_cursor_in - header_h
        fig.patches.append(plt.Rectangle((0.004, head_y0 / fig_h), 0.992, header_h / fig_h,
                                          transform=fig.transFigure, facecolor=COLOR_GROUP_BG,
                                          edgecolor="none", zorder=0))
        fig.text(0.014, (head_y0 + header_h / 2) / fig_h, dom_name, fontsize=20,
                  fontweight="bold", color=COLOR_TEXT_DARK, va="center", ha="left")
        y_cursor_in = head_y0

        n_rows_dom = rows_for(names)
        grid_top_in = y_cursor_in
        grid_bottom_in = grid_top_in - n_rows_dom * panel_h

        for idx, var in enumerate(names):
            row = idx // cols_per_domain
            col = idx % cols_per_domain
            ax_left = left_margin + col * cell_w + 0.020
            ax_width = cell_w - 0.040
            ax_top_in = grid_top_in - row * panel_h - 1.55
            ax_height_in = panel_h - 2.95
            draw_panel(fig, ax_left, ax_top_in, ax_width, ax_height_in, fig_h, var)

        y_cursor_in = grid_bottom_in

    return fig


if __name__ == "__main__":
    outdir = BASE_DIR
    fig = build_figure()
    base = "painel_hemato_gasometria_boxplot_medianapaciente_pt"
    fig.savefig(os.path.join(outdir, base + ".pdf"), format="pdf")
    fig.savefig(os.path.join(outdir, base + ".tiff"), format="tiff", dpi=600,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("saved")
