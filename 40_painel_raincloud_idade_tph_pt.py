import os
import matplotlib as mpl
import matplotlib.font_manager as fm


# ── Registrar Arial (com fallback local, sem depender de rede) ─────────────
def _registrar_arial():
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


_FONTE = _registrar_arial()
mpl.rcParams["font.family"] = _FONTE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import FancyBboxPatch
from scipy.stats import gaussian_kde, mannwhitneyu
import statsmodels.formula.api as smf

# ─────────────────────────────────────────────────────────────────────────
# Painel combinado (A = Idade; B = Tempo de internação), Geral/Óbito/Alta,
# com dados reais de 703pacientes.xlsx.
#   Painel A: comparação Óbito x Alta feita por teste de Mann-Whitney U
#             (Idade é variável contínua, não de contagem).
#   Painel B: comparação Óbito x Alta feita por regressão binomial negativa
#             (RTI), apropriada para dados de contagem (dias de internação).
# ─────────────────────────────────────────────────────────────────────────
XLSX_PATH = "703pacientes.xlsx"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PDF = os.path.join(OUTPUT_DIR, "painel_raincloud_idade_tph_pt.pdf")
OUT_PNG = os.path.join(OUTPUT_DIR, "painel_raincloud_idade_tph_pt.png")
OUT_SVG = os.path.join(OUTPUT_DIR, "painel_raincloud_idade_tph_pt.svg")

rng = np.random.default_rng(42)

df = pd.read_excel(XLSX_PATH)

CORES = {"Geral": "#7c6fc0", "Óbito": "#e07a2a", "Alta": "#2a9d8f"}
ORDEM = ["Geral", "Óbito", "Alta"]


def bootstrap_ic_mediana(x, n_boot=5000):
    x = np.asarray(x)
    medianas = np.empty(n_boot)
    for i in range(n_boot):
        amostra = rng.choice(x, size=len(x), replace=True)
        medianas[i] = np.median(amostra)
    lo, hi = np.percentile(medianas, [2.5, 97.5])
    return lo, hi


def resumir(coluna):
    grupos = {}
    for nome, sub in [("Geral", df), ("Óbito", df[df["Óbito"] == 1]),
                       ("Alta", df[df["Óbito"] == 0])]:
        x = sub[coluna].dropna().to_numpy()
        md = float(np.median(x))
        lo, hi = bootstrap_ic_mediana(x)
        q1, q3 = np.percentile(x, [25, 75])
        grupos[nome] = dict(
            dados=x, n=len(x), med=md, med_lo=lo, med_hi=hi,
            q1=float(q1), q3=float(q3),
            xbar=float(x.mean()), dp=float(x.std(ddof=1)),
            vmin=float(x.min()), vmax=float(x.max()),
        )
    return grupos


grupos_idade = resumir("Idade")
grupos_tph = resumir("Dias_permanência")

# ── Painel A: teste de Mann-Whitney U (Idade, Óbito x Alta) ────────────────
u_idade, p_idade = mannwhitneyu(
    grupos_idade["Óbito"]["dados"], grupos_idade["Alta"]["dados"],
    alternative="two-sided")

# ── Painel B: regressão binomial negativa (RTI, Tempo de internação) ──────
df_nb = df[["Dias_permanência", "Óbito"]].dropna().rename(
    columns={"Dias_permanência": "dias", "Óbito": "obito"})
m_nb = smf.negativebinomial("dias ~ obito", data=df_nb).fit(disp=False)
beta = m_nb.params["obito"]
beta_lo, beta_hi = m_nb.conf_int().loc["obito"]
p_tph = m_nb.pvalues["obito"]
rti = np.exp(beta)
rti_lo, rti_hi = np.exp(beta_lo), np.exp(beta_hi)


def formata_p(p):
    return "p < 0,001" if p < 0.001 else f"p = {p:.3f}".replace(".", ",")


delta_md_idade = grupos_idade["Óbito"]["med"] - grupos_idade["Alta"]["med"]
delta_md_tph = grupos_tph["Óbito"]["med"] - grupos_tph["Alta"]["med"]

# ── Desenho de um painel (raincloud de 3 grupos) ───────────────────────────
JITTER_N = 80


def desenha_painel(ax, grupos, xlim, xlabel, titulo, texto_stat, delta_md, unidade):
    ys = {"Geral": 2, "Óbito": 1, "Alta": 0}
    x_kde = np.linspace(max(0, xlim[0]), xlim[1], 300)
    KDE_MAX = 0.42

    for nome in ORDEM:
        g = grupos[nome]
        cor = CORES[nome]
        y0 = ys[nome]
        amostra = g["dados"]

        # 1. Nuvem KDE (metade superior, "raincloud")
        kde = gaussian_kde(amostra, bw_method=0.25)
        dens = kde(x_kde)
        dens_n = dens / dens.max() * KDE_MAX
        ax.fill_between(x_kde, y0, y0 + dens_n, color=cor, alpha=0.50, zorder=2)
        ax.plot(x_kde, y0 + dens_n, color=cor, lw=1.4, alpha=0.75, zorder=3)

        # 2. Jitter (amostra de pontos individuais)
        n_jit = min(len(amostra), JITTER_N)
        idx = rng.choice(len(amostra), n_jit, replace=False)
        jitter = rng.uniform(-0.12, 0.0, n_jit)
        ax.scatter(amostra[idx], y0 + jitter - 0.05,
                   color=cor, alpha=0.38, s=16, linewidths=0, zorder=3)

        # 3. Caixa IIQ + bigodes (mín–máx)
        BH = 0.10
        by = y0 - 0.30
        ax.hlines(by, g["vmin"], g["q1"], colors=cor, lw=2.2, ls=(0, (4, 3)), zorder=4)
        ax.hlines(by, g["q3"], g["vmax"], colors=cor, lw=2.2, ls=(0, (4, 3)), zorder=4)
        for xw in [g["vmin"], g["vmax"]]:
            ax.vlines(xw, by - BH * 0.7, by + BH * 0.7, colors=cor, lw=2.2, zorder=4)
        caixa = FancyBboxPatch(
            (g["q1"], by - BH), g["q3"] - g["q1"], 2 * BH,
            boxstyle="round,pad=0.3",
            linewidth=1.8, edgecolor=cor,
            facecolor=cor + "33", zorder=5,
        )
        ax.add_patch(caixa)

        # 4. Mediana: diamante + IC 95% bootstrap (barra horizontal de erro)
        ax.hlines(by, g["med_lo"], g["med_hi"], colors=cor, lw=2.4, zorder=6)
        for xcap in [g["med_lo"], g["med_hi"]]:
            ax.vlines(xcap, by - BH * 0.45, by + BH * 0.45, colors=cor, lw=2.0, zorder=6)
        ax.scatter([g["med"]], [by], marker="D", s=110, color=cor,
                   edgecolors="white", linewidths=1.2, zorder=7)

        # 5. Média (linha tracejada cinza)
        ax.vlines(g["xbar"], by - BH * 0.8, by + BH * 0.8, colors="gray", lw=2.2,
                   ls=(0, (5, 4)), zorder=6)

        # 6. Rótulos
        ax.text(LABEL_X[titulo], y0 + 0.08, nome, fontsize=18, fontweight="bold",
                color=cor, ha="right", va="center", fontfamily=_FONTE)
        ax.text(LABEL_X[titulo], y0 - 0.09, f"n = {g['n']}", fontsize=14,
                fontweight="bold", color="#888", ha="right", va="center",
                fontfamily=_FONTE)

    # ── Seta ΔMd entre Óbito e Alta ─────────────────────────────────────────
    y_obito = ys["Óbito"] - 0.30
    y_alta = ys["Alta"] - 0.30
    x_med_obito = grupos["Óbito"]["med"]
    x_med_alta = grupos["Alta"]["med"]
    y_meio = (y_obito + y_alta) / 2

    ax.annotate("", xy=(x_med_obito, y_meio), xytext=(x_med_alta, y_meio),
                arrowprops=dict(arrowstyle="<->", color="#555", lw=1.8,
                                 shrinkA=0, shrinkB=0), zorder=8)
    for x_m, y_g in [(x_med_alta, y_alta), (x_med_obito, y_obito)]:
        ax.plot([x_m, x_m], [y_g, y_meio], color="#555", lw=1.3,
                 ls=(0, (4, 3)), zorder=7)

    sinal = f"+{delta_md:.0f}" if delta_md >= 0 else f"{delta_md:.0f}"
    ax.text((x_med_alta + x_med_obito) / 2, y_meio + 0.07,
            f"Δ Md = {sinal} {unidade}", fontsize=13, color="#555",
            fontweight="bold", ha="center", va="bottom", fontfamily=_FONTE,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#bbb", alpha=0.88), zorder=9)

    # ── Caixa de estatística (canto superior direito) ──────────────────────
    ax.text(0.985, 0.95, texto_stat, transform=ax.transAxes,
             fontsize=12.5, color="#222", fontweight="bold", ha="right", va="top",
             fontfamily=_FONTE, linespacing=1.6,
             bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                       edgecolor="#ccc", alpha=0.95))

    ax.set_title(titulo, fontsize=16, fontweight="bold", color="#1F2328",
                 loc="left", fontfamily=_FONTE)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.75, 2.55)
    ax.set_xlabel(xlabel, fontsize=14, color="#555", fontfamily=_FONTE)
    ax.set_yticks([])
    ax.xaxis.set_tick_params(labelsize=11, labelcolor="#666")
    ax.grid(axis="x", color="#ddd", linewidth=0.7, linestyle="--", alpha=0.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#ccc")
    for lbl in ax.get_xticklabels():
        lbl.set_fontfamily(_FONTE)


LABEL_X = {"A) Idade": -9, "B) Tempo de internação": -11}

fig, (axA, axB) = plt.subplots(2, 1, figsize=(14, 13), facecolor="white")
for ax in (axA, axB):
    ax.set_facecolor("#fafafa")

texto_A = (f"Teste de Mann-Whitney U\nU = {u_idade:.0f}\n{formata_p(p_idade)}")
texto_B = (f"Modelo binomial negativo\nRTI = {rti:.2f} (IC 95% {rti_lo:.2f}–{rti_hi:.2f})\n"
           f"{formata_p(p_tph)}")

desenha_painel(axA, grupos_idade, xlim=(-7, 110), xlabel="Idade (anos)",
               titulo="A) Idade", texto_stat=texto_A,
               delta_md=delta_md_idade, unidade="anos")
desenha_painel(axB, grupos_tph, xlim=(-5, 95), xlabel="Tempo de internação (dias)",
               titulo="B) Tempo de internação", texto_stat=texto_B,
               delta_md=delta_md_tph, unidade="dias")

fig.suptitle("Distribuição de Idade e Tempo de Internação por Desfecho Hospitalar",
             fontsize=19, fontweight="bold", color="#1F2328", y=0.985,
             fontfamily=_FONTE)

# ── Legenda única, na parte inferior da figura ─────────────────────────────
l_box = mpatches.Patch(facecolor="#88888855", edgecolor="#888", lw=1.5,
                        label="IIQ (P25–P75)")
l_med = mlines.Line2D([], [], marker="D", color="#888", markersize=9,
                       lw=2.2, label="Mediana (IC 95% bootstrap)")
l_mean = mlines.Line2D([], [], color="gray", lw=1.6, ls=(0, (5, 4)), label="Média")
l_seta = mlines.Line2D([], [], color="#555", lw=1.8, marker="|", markersize=10,
                        label="Diferença entre medianas (Óbito − Alta)")
fig.legend(handles=[l_box, l_med, l_mean, l_seta], loc="lower center",
           ncol=4, frameon=False, fontsize=12,
           prop={"family": _FONTE, "size": 12}, bbox_to_anchor=(0.5, -0.01))

plt.tight_layout(rect=[0.03, 0.035, 1, 0.96])

fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(OUT_SVG, bbox_inches="tight", facecolor="white")
fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor="white")

print("── Painel A (Idade) — Mann-Whitney U ──")
print(f"  Geral: n={grupos_idade['Geral']['n']} Md={grupos_idade['Geral']['med']:.0f} "
      f"IC95%[{grupos_idade['Geral']['med_lo']:.0f}-{grupos_idade['Geral']['med_hi']:.0f}] "
      f"IIQ[{grupos_idade['Geral']['q1']:.0f}-{grupos_idade['Geral']['q3']:.0f}] "
      f"x̄={grupos_idade['Geral']['xbar']:.1f}±{grupos_idade['Geral']['dp']:.1f} "
      f"máx={grupos_idade['Geral']['vmax']:.0f}")
print(f"  Óbito: n={grupos_idade['Óbito']['n']} Md={grupos_idade['Óbito']['med']:.0f} "
      f"IC95%[{grupos_idade['Óbito']['med_lo']:.0f}-{grupos_idade['Óbito']['med_hi']:.0f}] "
      f"IIQ[{grupos_idade['Óbito']['q1']:.0f}-{grupos_idade['Óbito']['q3']:.0f}] "
      f"x̄={grupos_idade['Óbito']['xbar']:.1f}±{grupos_idade['Óbito']['dp']:.1f} "
      f"máx={grupos_idade['Óbito']['vmax']:.0f}")
print(f"  Alta: n={grupos_idade['Alta']['n']} Md={grupos_idade['Alta']['med']:.0f} "
      f"IC95%[{grupos_idade['Alta']['med_lo']:.0f}-{grupos_idade['Alta']['med_hi']:.0f}] "
      f"IIQ[{grupos_idade['Alta']['q1']:.0f}-{grupos_idade['Alta']['q3']:.0f}] "
      f"x̄={grupos_idade['Alta']['xbar']:.1f}±{grupos_idade['Alta']['dp']:.1f} "
      f"máx={grupos_idade['Alta']['vmax']:.0f}")
print(f"  U = {u_idade:.0f}; {formata_p(p_idade)}; ΔMd = {delta_md_idade:+.0f} anos")

print("\n── Painel B (Tempo de internação) — Binomial negativa (RTI) ──")
print(f"  RTI = {rti:.2f} (IC95% {rti_lo:.2f}-{rti_hi:.2f}); {formata_p(p_tph)}; "
      f"ΔMd = {delta_md_tph:+.0f} dias")

print(f"\nExportado:\n  {OUT_PDF}\n  {OUT_PNG}\n  {OUT_SVG}")
