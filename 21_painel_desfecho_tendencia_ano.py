"""
Painel A) Distribuição do Desfecho (Óbito x Alta)
Painel B) Tendência de Alta e Óbito por Ano
Fonte: New_pacientes703.xlsx
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

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

GREEN = "#A8D8C8"   # Alta (bar)
ORANGE = "#FFCB99"  # Óbito (bar)
LINE_GREEN = "#1D9E75"
LINE_ORANGE = "#E07B39"

# ----------------------------------------------------------------------
df = pd.read_excel(DATA_PATH, sheet_name="Sheet1")
df["Ano"] = pd.to_datetime(df["Data de Entrada"]).dt.year

contagem = df["Óbito"].value_counts().sort_index()
trend = df.groupby(["Ano", "Óbito"]).size().reset_index(name="Contagem")
alta = trend[trend["Óbito"] == 0].set_index("Ano")["Contagem"]
obito = trend[trend["Óbito"] == 1].set_index("Ano")["Contagem"]
anos = sorted(trend["Ano"].unique())

# ----------------------------------------------------------------------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

# ---------------------------- PANEL A ------------------------------------
bars = axA.bar(["0", "1"], contagem.values, color=[GREEN, ORANGE], width=0.6, edgecolor="none")
for b, v in zip(bars, contagem.values):
    axA.text(b.get_x() + b.get_width() / 2, v / 2, str(v), ha="center", va="center",
              fontsize=16, fontweight="bold", color="#222222")

axA.set_title("Distribuição do Desfecho", fontsize=15, fontweight="bold", pad=12)
axA.set_xlabel("Óbito ou Alta", fontsize=11, fontweight="bold")
axA.set_ylabel("")
axA.set_yticks([])
for spine in ["top", "right", "left"]:
    axA.spines[spine].set_visible(False)
axA.text(-0.14, 1.08, "A)", transform=axA.transAxes, fontsize=16, fontweight="bold")

# ---------------------------- PANEL B -------------------------------------
axB.fill_between(alta.index, alta.values, color=LINE_GREEN, alpha=0.12, zorder=1)
axB.fill_between(obito.index, obito.values, color="#8a7a4a", alpha=0.12, zorder=1)
axB.plot(alta.index, alta.values, color=LINE_GREEN, linewidth=2, marker="o",
          markersize=7, markerfacecolor="white", markeredgecolor=LINE_GREEN,
          markeredgewidth=2, label="Alta (0)", zorder=3)
axB.plot(obito.index, obito.values, color=LINE_ORANGE, linewidth=2, marker="o",
          markersize=7, markerfacecolor="white", markeredgecolor=LINE_ORANGE,
          markeredgewidth=2, label="Óbito (1)", zorder=3)

for serie, cor in [(alta, LINE_GREEN), (obito, LINE_ORANGE)]:
    for ano, val in serie.items():
        if ano == 2021:
            offset_x = -14 if cor == LINE_GREEN else 14
            ha = "right" if cor == LINE_GREEN else "left"
        else:
            offset_x = 0
            ha = "center"
        axB.annotate(str(val), xy=(ano, val), xytext=(offset_x, 8), textcoords="offset points",
                      ha=ha, va="bottom", fontsize=11, color=cor, fontweight="bold")

axB.set_title("TENDÊNCIA DE ALTA E ÓBITO POR ANO", fontsize=13.5, fontweight="bold", pad=12)
axB.set_xlabel("Ano de internação", fontsize=11)
axB.set_ylabel("Número de pacientes", fontsize=11)
axB.set_xticks(anos)
axB.set_xticklabels([str(a) for a in anos])
axB.set_ylim(0, 180)
axB.grid(axis="y", color="#e0ddd8", linewidth=0.6, linestyle="--")
axB.set_axisbelow(True)
for spine in ["top", "right"]:
    axB.spines[spine].set_visible(False)
axB.legend(loc="upper left", frameon=False, fontsize=10)
axB.text(-0.07, 1.08, "B)", transform=axB.transAxes, fontsize=16, fontweight="bold")

fig.tight_layout()

for dpi, suffix in [(180, ""), (600, "_600dpi")]:
    out_path = os.path.join(BASE_DIR, f"painel_desfecho_tendencia_ano{suffix}.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print("saved:", out_path)

plt.close(fig)
