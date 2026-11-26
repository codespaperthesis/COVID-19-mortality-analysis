#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paineis de exames laboratoriais (Figura 16) - UMA FIGURA POR DOMINIO.

Derivado do script "Paineis de exames laboratoriais - mediana por paciente
unico, v2". Mesma representacao grafica (box plot por paciente, janelas de
referencia, p-valor, n por grupo), mas cada dominio (Funcao renal, Perfil
trombo-hemostatico, Controle glicemico, ...) e salvo como um arquivo
separado em TIFF 600 dpi (LZW), alem de PDF vetorial e PNG.

Uso:
    python 12_paineis_laboratoriais_por_dominio.py \
        --data-dir /caminho/dos/pkl --outdir figuras_dominios \
        [--style box|point] [--todas]

Arquivos esperados em --data-dir:
    patient_median_data.pkl        (STATS)
    patient_median_raw_arrays.pkl  (RAW)
    ref_lookup.pkl                 (REFS)

Por padrao gera a versao "enxuta" (apenas variaveis com p<0,05 e n>=10 em
ambos os grupos), igual a Figura 16 da tese. Use --todas para incluir as 36
variaveis.
"""

import argparse
import os
import pickle
import re
import unicodedata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

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

STATS, RAW, REFS = {}, {}, {}

ADULT_LABEL = {
    "Colesterol total": "2a1d - 150a",
    "Triglicérides": "20-150a",
    "HDL colesterol": "20a1d-150a",
}

ALL_DOMAINS = [
    ("Função renal", ["RPCU", "Creatinina", "Ureia"]),
    ("Perfil trombo-hemostático", ["D-dímero", "Fibrinogênio", "TTPA", "TP/INR"]),
    ("Controle glicêmico", ["Hb1Ac", "Glicemia/jejum"]),
    ("Função hepática", ["TGO", "TGP", "Bilirrubina", "Fosfatase alcalina", "Gama GT", "Albumina"]),
    ("Função cardíaca", ["Troponina I", "CPK", "CK-MB"]),
    ("Lipidograma", ["Colesterol total", "Triglicérides", "LDL colesterol", "HDL colesterol"]),
    ("Perfil inflamatório", ["Ferritina", "Proteína C reativa", "Lactato", "LDH", "IL-6"]),
    ("Eletrólitos", ["Cálcio total", "Cálcio ionizado", "Sódio", "Potássio", "Cloro", "Magnésio"]),
    ("Outros marcadores", ["LCR-Lactato", "Procalcitonina", "Ácido úrico"]),
]

# Figura 15 - hematologicos e gasometricos (chaves "|art"/"|ven" separam a
# mesma variavel nas duas gasometrias; o sufixo nao aparece no grafico).
FIG15_DOMAINS = [
    ("Hematológicos", ["Plaquetas", "Glóbulos brancos", "Glóbulos vermelhos", "Hematócrito", "RDW",
                       "Monócitos", "Neutrófilos segment.", "Linfócitos", "Basófilos", "Eosinófilos",
                       "CHCM", "HCM"]),
    ("Gasometria arterial", ["PCO2|art", "BE|art", "FiO2|art", "P50c|art", "SaO2|art", "pH|art",
                             "HCO3|art", "PO2|art", "Temperatura|art"]),
    ("Gasometria venosa", ["BE|ven", "PCO2|ven", "FiO2|ven", "PO2|ven", "P50c|ven", "HCO3|ven",
                           "pH|ven", "SvO2|ven", "Temperatura|ven"]),
]

DISPLAY = {"Hb1Ac": "HbA1c"}


def display_name(var):
    v = var.split("|")[0]
    return DISPLAY.get(v, v)


def load_data(data_dir):
    global STATS, RAW, REFS
    with open(os.path.join(data_dir, "patient_median_data.pkl"), "rb") as f:
        STATS = pickle.load(f)
    with open(os.path.join(data_dir, "patient_median_raw_arrays.pkl"), "rb") as f:
        RAW = pickle.load(f)
    with open(os.path.join(data_dir, "ref_lookup.pkl"), "rb") as f:
        REFS = pickle.load(f)


def is_significant_and_robust(var, min_n=10):
    s = STATS.get(var)
    if s is None:
        return False
    p = s["p"]
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return False
    if p >= 0.05:
        return False
    if s["n_alta"] < min_n or s["n_obito"] < min_n:
        return False
    return True


def selected_domains(leaner=True, domains=None):
    out = []
    for name, names in (domains or ALL_DOMAINS):
        kept = [v for v in names if v in STATS and (is_significant_and_robust(v) if leaner else True)]
        if kept:
            out.append((name, kept))
    return out


def get_ref_windows(var):
    entries = REFS.get(var, [])
    if var in ADULT_LABEL:
        entries = [e for e in entries if e[0] == ADULT_LABEL[var]]
    windows = []
    for rotulo, (lo, hi), unidade in entries:
        rl = (rotulo or "").lower()
        color = COLOR_REF_GERAL
        if "masculino" in rl:
            color = COLOR_REF_M
        elif "feminino" in rl:
            color = COLOR_REF_F
        windows.append((lo, hi, color))
    return windows


def fmt_p(p):
    if p is None or np.isnan(p):
        return "n/a"
    return "p<0,001" if p < 0.001 else f"p={p:.3f}".replace(".", ",")


def draw_panel(fig, ax_left, ax_top_in, ax_width, ax_height_in, fig_h, var, style="box"):
    """Identica a versao v2 do script original."""
    s = STATS[var]
    ax = fig.add_axes([ax_left, (ax_top_in - ax_height_in) / fig_h, ax_width, ax_height_in / fig_h])

    windows = get_ref_windows(var)

    vals = []
    for k in ("med_alta", "q1_alta", "q3_alta", "med_obito", "q1_obito", "q3_obito"):
        v = s[k]
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            vals.append(v)
    raw = RAW.get(var, {"alta": np.array([]), "obito": np.array([])})
    if style == "box":
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
            ax.axvline(lo, color=color, linewidth=1.3, linestyle=(0, (4, 2)), zorder=2, alpha=0.85)
        if hi is not None:
            ax.axvline(hi, color=color, linewidth=1.3, linestyle=(0, (4, 2)), zorder=2, alpha=0.85)

    if style == "point":
        def plot_point(y, med, q1, q3, color, n):
            if n == 0 or med is None or (isinstance(med, float) and np.isnan(med)):
                return
            if q1 is not None and q3 is not None and not np.isnan(q1) and not np.isnan(q3):
                ax.plot([q1, q3], [y, y], color=color, linewidth=2.2, zorder=3, solid_capstyle="round")
            ax.plot(med, y, marker="o", markersize=8.5, color=color, zorder=4,
                    markeredgecolor="white", markeredgewidth=1.0)

        plot_point(0, s["med_alta"], s["q1_alta"], s["q3_alta"], COLOR_PROTECTIVE, s["n_alta"])
        plot_point(1, s["med_obito"], s["q1_obito"], s["q3_obito"], COLOR_RISK, s["n_obito"])
    else:
        box_data, positions, colors = [], [], []
        if len(raw["alta"]):
            box_data.append(raw["alta"]); positions.append(0); colors.append(COLOR_PROTECTIVE)
        if len(raw["obito"]):
            box_data.append(raw["obito"]); positions.append(1); colors.append(COLOR_RISK)
        if box_data:
            orient = ({"orientation": "horizontal"}
                      if tuple(int(x) for x in matplotlib.__version__.split(".")[:2]) >= (3, 10)
                      else {"vert": False})
            bp = ax.boxplot(box_data, positions=positions, widths=0.5, **orient,
                            patch_artist=True, showfliers=True, zorder=4,
                            flierprops=dict(marker="o", markersize=3.2, markerfacecolor="none",
                                            alpha=0.5, markeredgewidth=0.6),
                            medianprops=dict(color="white", linewidth=1.8),
                            whiskerprops=dict(linewidth=1.3), capprops=dict(linewidth=1.3))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.85)
                patch.set_edgecolor(color)
            for i, color in enumerate(colors):
                bp["whiskers"][2 * i].set_color(color)
                bp["whiskers"][2 * i + 1].set_color(color)
                bp["caps"][2 * i].set_color(color)
                bp["caps"][2 * i + 1].set_color(color)
                bp["fliers"][i].set_markeredgecolor(color)

    ax.set_xlim(xlim)
    nbins_override = {"Ferritina": 5}
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins_override.get(var, 9),
                                           steps=[1, 2, 2.5, 5, 10], min_n_ticks=5))
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Alta", "Óbito"], fontsize=10.0, fontweight="bold")
    ax.get_yticklabels()[0].set_color(COLOR_PROTECTIVE)
    ax.get_yticklabels()[1].set_color(COLOR_RISK)
    ax.tick_params(axis="x", labelsize=8.6, colors=COLOR_AXIS)
    ax.tick_params(axis="y", length=0)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(COLOR_AXIS)

    p = s["p"]
    sig = (p is not None) and (not np.isnan(p)) and p < 0.05
    ax.set_title(display_name(var), fontsize=11.6, fontweight="bold", color=COLOR_TEXT_DARK, pad=8)
    ax.text(0.5, 1.42, fmt_p(p), transform=ax.transAxes, fontsize=9.4,
            fontweight="bold" if sig else "normal",
            color=(COLOR_TEXT_DARK if sig else COLOR_TEXT_MUTED), ha="center", va="bottom")
    ax.text(0.0, -0.30, f"Alta n={s['n_alta']}", transform=ax.transAxes, fontsize=7.7,
            color=COLOR_PROTECTIVE, ha="left", va="top", style="italic", fontweight="bold")
    ax.text(1.0, -0.30, f"Óbito n={s['n_obito']}", transform=ax.transAxes, fontsize=7.7,
            color=COLOR_RISK, ha="right", va="top", style="italic", fontweight="bold")


def legend_handles(style):
    if style == "point":
        h = [Line2D([0], [0], marker="o", color=COLOR_PROTECTIVE, linewidth=2.2, markersize=9,
                    label="Alta — mediana (IQR)"),
             Line2D([0], [0], marker="o", color=COLOR_RISK, linewidth=2.2, markersize=9,
                    label="Óbito — mediana (IQR)")]
    else:
        h = [mpatches.Patch(facecolor=COLOR_PROTECTIVE, alpha=0.85, edgecolor=COLOR_PROTECTIVE, label="Alta — box plot"),
             mpatches.Patch(facecolor=COLOR_RISK, alpha=0.85, edgecolor=COLOR_RISK, label="Óbito — box plot")]
    return h + [
        mpatches.Patch(color=COLOR_REF_GERAL, alpha=0.35, label="Janela de referência (geral)"),
        mpatches.Patch(color=COLOR_REF_M, alpha=0.35, label="Janela de referência (masc.)"),
        mpatches.Patch(color=COLOR_REF_F, alpha=0.35, label="Janela de referência (fem.)"),
    ]


def build_domain_figure(dom_name, names, style="box", cols=3, show_legend=True):
    """Uma figura com a faixa do dominio + seus paineis (ate `cols` por linha)."""
    cols = min(cols, len(names)) if len(names) < cols else cols
    n_rows = int(np.ceil(len(names) / cols))
    cell_w_in = 3.3
    panel_h = 2.75
    header_h = 0.46
    legend_h = 0.55 if show_legend else 0.0
    left_in, right_in, bottom_in = 0.55, 0.20, 0.25

    fig_w = max(left_in + right_in + cols * cell_w_in, 7.2 if show_legend else 0)
    fig_h = legend_h + header_h + n_rows * panel_h + bottom_in

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
    fig.patch.set_facecolor("white")

    if show_legend:
        fig.legend(handles=legend_handles(style), loc="upper left",
                   bbox_to_anchor=(0.01, 1 - 0.06 / fig_h), ncol=(5 if fig_w >= 10 else 3), frameon=False,
                   fontsize=8.6, columnspacing=1.0, handletextpad=0.5)

    head_y0 = fig_h - legend_h - header_h
    fig.patches.append(plt.Rectangle((0.004, head_y0 / fig_h), 0.992, header_h / fig_h,
                                     transform=fig.transFigure, facecolor=COLOR_GROUP_BG,
                                     edgecolor="none", zorder=0))
    fig.text(0.015, (head_y0 + header_h / 2) / fig_h, dom_name, fontsize=14,
             fontweight="bold", color=COLOR_TEXT_DARK, va="center", ha="left")

    grid_top_in = head_y0
    usable_w = 1.0 - (left_in + right_in) / fig_w
    cell_w = usable_w / cols
    x0 = left_in / fig_w
    for idx, var in enumerate(names):
        row, col = divmod(idx, cols)
        ax_left = x0 + col * cell_w + 0.10 / fig_w * 3
        ax_width = cell_w - 0.75 / fig_w
        ax_top_in = grid_top_in - row * panel_h - 0.80
        ax_height_in = panel_h - 1.55
        draw_panel(fig, ax_left, ax_top_in, ax_width, ax_height_in, fig_h, var, style=style)
    return fig


def slug(text):
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]+", "_", t).strip("_")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="/home/claude")
    ap.add_argument("--outdir", default="figuras_dominios")
    ap.add_argument("--style", choices=["box", "point"], default="box")
    ap.add_argument("--todas", action="store_true", help="inclui variaveis nao significativas")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--sem-legenda", action="store_true", help="omite a legenda de cores no topo")
    ap.add_argument("--figura", choices=["15", "16"], default="16",
                    help="15 = hematologicos/gasometricos; 16 = bioquimicos (padrao)")
    args = ap.parse_args()

    load_data(args.data_dir)
    os.makedirs(args.outdir, exist_ok=True)
    letters = "abcdefghijklmnopqrstuvwxyz"
    doms = FIG15_DOMAINS if args.figura == "15" else ALL_DOMAINS
    cols = 4 if args.figura == "15" else 3
    for i, (dom, names) in enumerate(selected_domains(leaner=not args.todas, domains=doms)):
        fig = build_domain_figure(dom, names, style=args.style, cols=cols, show_legend=not args.sem_legenda)
        base = os.path.join(args.outdir, f"Fig{args.figura}{letters[i]}_{slug(dom)}")
        fig.savefig(base + ".tiff", format="tiff", dpi=args.dpi, pil_kwargs={"compression": "tiff_lzw"})
        fig.savefig(base + ".pdf", format="pdf")
        fig.savefig(base + ".png", format="png", dpi=300)
        plt.close(fig)
        print("salvo:", base + ".tiff", f"({len(names)} variáveis)")


if __name__ == "__main__":
    main()
