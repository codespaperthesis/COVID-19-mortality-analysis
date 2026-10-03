import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────
# Curvas de Kaplan-Meier — sobrevida hospitalar por esquema vacinal
# (completo vs. incompleto, 2021-2022, período em que a vacinação já
# existia), com teste de log-rank e razão de risco (HR) de Cox ajustada
# por ano.
#
# Grupos lidos diretamente das colunas já classificadas na planilha:
#   Ecompleto=1   → esquema vacinal completo
#   Eincompleto=1 → esquema vacinal incompleto (inclui não vacinados)
# ─────────────────────────────────────────────────────────────────────────
XLSX_PATH = "703pacientes.xlsx"
OUTPUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "kaplan_meier_cox_esquema_vacinal.png")
OUTPUT_TIFF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "kaplan_meier_cox_esquema_vacinal.tiff")

# ── Paleta ────────────────────────────────────────────────────────────────
BG = "#FFFFFF"
TEXT = "#1F2328"
SUBTEXT = "#57606A"
COR_INCOMPLETO = "#E0703A"
COR_COMPLETO = "#1D9E75"

# ════════════════════════════════════════════════════════════
# 1. CARREGAR DADOS (restritos a 2021-2022, período com vacinação)
# ════════════════════════════════════════════════════════════
dados = pd.read_excel(XLSX_PATH)
dados["Data de Entrada"] = pd.to_datetime(dados["Data de Entrada"])
dados["Ano"] = dados["Data de Entrada"].dt.year
dados = dados[dados["Ano"].isin([2021, 2022])].copy()

assert ((dados["Ecompleto"] + dados["Eincompleto"]) == 1).all(), (
    "Ecompleto e Eincompleto deveriam ser mutuamente exclusivos e cobrir "
    "todos os pacientes (soma = 1 em cada linha)")

data_min = dados["Data de Entrada"].min()
dados["Dias_desde_inicio"] = (dados["Data de Entrada"] - data_min).dt.days

dados["Grupo"] = dados["Ecompleto"].map({0: "Esquema incompleto", 1: "Esquema completo"})
n_incompleto = int((dados["Ecompleto"] == 0).sum())
n_completo = int((dados["Ecompleto"] == 1).sum())

CORES = {"Esquema incompleto": COR_INCOMPLETO, "Esquema completo": COR_COMPLETO}


def formata_p(p):
    return "p<0,001" if p < 0.001 else f"p={p:.3f}".replace(".", ",")


MESES_PT = {1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
            7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"}


# ════════════════════════════════════════════════════════════
# 2. FUNÇÕES AUXILIARES
# ════════════════════════════════════════════════════════════
def plota_km(ax, coluna_tempo, titulo_eixo_x, eixo_como_data=False):
    for grupo in ["Esquema incompleto", "Esquema completo"]:
        sub = dados[dados["Grupo"] == grupo]
        kmf = KaplanMeierFitter()
        kmf.fit(sub[coluna_tempo], event_observed=sub["Óbito"],
                label=f"{grupo} (n={len(sub)})")
        kmf.plot_survival_function(ax=ax, color=CORES[grupo], linewidth=2.3,
                                    ci_alpha=0.15)

    ax.set_xlabel(titulo_eixo_x, fontsize=12, color=SUBTEXT)
    ax.set_ylabel("Probabilidade de sobrevida", fontsize=12, color=SUBTEXT)
    ax.set_ylim(0, 1.03)
    ax.set_title("2021-2022", fontsize=14, fontweight="bold", color=TEXT,
                 loc="left")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.grid(alpha=0.25)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    if eixo_como_data:
        def fmt_data(x, _):
            data = data_min + pd.Timedelta(days=x)
            return f"{MESES_PT[data.month]}/{data.year}"
        ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_data))
        plt.setp(ax.get_xticklabels(), rotation=0)


def teste_logrank(coluna_tempo):
    a = dados[dados["Grupo"] == "Esquema incompleto"]
    b = dados[dados["Grupo"] == "Esquema completo"]
    resultado = logrank_test(a[coluna_tempo], b[coluna_tempo],
                              event_observed_A=a["Óbito"],
                              event_observed_B=b["Óbito"])
    return resultado.p_value


def cox_hr_ajustado(coluna_tempo):
    cph = CoxPHFitter()
    base = dados[[coluna_tempo, "Óbito", "Ecompleto", "Ano"]].rename(
        columns={coluna_tempo: "tempo"})
    cph.fit(base, duration_col="tempo", event_col="Óbito")
    hr = np.exp(cph.params_["Ecompleto"])
    lo, hi = np.exp(cph.confidence_intervals_.loc["Ecompleto"])
    p = cph.summary.loc["Ecompleto", "p"]
    return hr, lo, hi, p


# ════════════════════════════════════════════════════════════
# 3. GRÁFICO 2×2
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(15, 11), facecolor=BG)

CAIXA_ANOTACAO = dict(boxstyle="round,pad=0.35", fc="white", ec="#D0D7DE",
                       alpha=0.95, linewidth=0.8)

# Linha 1: curvas de KM + teste de log-rank
plota_km(axes[0, 0], "Dias_desde_inicio", "Data de admissão", eixo_como_data=True)
p_lr_data = teste_logrank("Dias_desde_inicio")
axes[0, 0].text(0.03, 0.03, f"Log-rank: {formata_p(p_lr_data)}",
                 transform=axes[0, 0].transAxes, ha="left", va="bottom",
                 fontsize=10, style="italic", color=TEXT, bbox=CAIXA_ANOTACAO)

plota_km(axes[0, 1], "Dias_permanência", "Tempo de internação (dias)")
p_lr_dias = teste_logrank("Dias_permanência")
axes[0, 1].text(0.03, 0.03, f"Log-rank: {formata_p(p_lr_dias)}",
                 transform=axes[0, 1].transAxes, ha="left", va="bottom",
                 fontsize=10, style="italic", color=TEXT, bbox=CAIXA_ANOTACAO)

# Linha 2: mesmas curvas + HR de Cox ajustado por ano
plota_km(axes[1, 0], "Dias_desde_inicio", "Data de admissão", eixo_como_data=True)
hr, lo, hi, p_cox = cox_hr_ajustado("Dias_desde_inicio")
axes[1, 0].text(0.03, 0.03,
                 f"HR de Cox (ajustado por ano) = {hr:.2f} ({lo:.2f}–{hi:.2f})\n"
                 f"{formata_p(p_cox)}",
                 transform=axes[1, 0].transAxes, ha="left", va="bottom",
                 fontsize=10, style="italic", color=TEXT, bbox=CAIXA_ANOTACAO)

plota_km(axes[1, 1], "Dias_permanência", "Tempo de internação (dias)")
hr2, lo2, hi2, p_cox2 = cox_hr_ajustado("Dias_permanência")
axes[1, 1].text(0.03, 0.03,
                 f"HR de Cox (ajustado por ano) = {hr2:.2f} ({lo2:.2f}–{hi2:.2f})\n"
                 f"{formata_p(p_cox2)}",
                 transform=axes[1, 1].transAxes, ha="left", va="bottom",
                 fontsize=10, style="italic", color=TEXT, bbox=CAIXA_ANOTACAO)

fig.suptitle("Curvas de Kaplan-Meier — Sobrevida Hospitalar por Esquema Vacinal "
             "(2021-2022)", fontsize=19, fontweight="bold", color=TEXT, y=0.985)
fig.text(0.5, 0.955,
          f"Esquema incompleto: n={n_incompleto} | Esquema completo: n={n_completo} | "
          "evento = óbito (alta = censura) | HR = razão de risco (hazard ratio)",
          ha="center", va="top", fontsize=11.5, color=SUBTEXT)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight", facecolor=BG)

# TIFF com compressão LZW (o TIFF sem compressão do matplotlib passa de 50MB)
from PIL import Image  # noqa: E402
_buf_png = OUTPUT_PNG.replace(".png", "_300dpi_tmp.png")
plt.savefig(_buf_png, dpi=300, bbox_inches="tight", facecolor=BG)
Image.open(_buf_png).save(OUTPUT_TIFF, dpi=(300, 300), compression="tiff_lzw")
os.remove(_buf_png)

print(f"Gráfico salvo em: {OUTPUT_PNG} e {OUTPUT_TIFF}")
print(f"Log-rank (data de admissão): p={p_lr_data:.4f}")
print(f"Log-rank (dias de internação): p={p_lr_dias:.4f}")
print(f"Cox HR (data de admissão, ajustado por ano): {hr:.3f} [{lo:.3f}-{hi:.3f}] p={p_cox:.4f}")
print(f"Cox HR (dias de internação, ajustado por ano): {hr2:.3f} [{lo2:.3f}-{hi2:.3f}] p={p_cox2:.4f}")
plt.show()
