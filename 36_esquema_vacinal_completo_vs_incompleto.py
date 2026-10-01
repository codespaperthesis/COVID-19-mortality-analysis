import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import fisher_exact
from PIL import Image

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────
# Esquema vacinal completo vs. incompleto e óbito hospitalar.
#
# Definição (fornecida pelo usuário):
#   Completo   = 2 doses de Pfizer/Moderna/AstraZeneca/Coronavac/Novavax,
#                OU 1 dose de Janssen (vacina de dose única).
#   Incompleto = menos doses que o necessário para completar o esquema
#                acima (ex.: 1 dose de uma vacina de 2 doses).
# Pacientes não vacinados ficam de fora da comparação completo/incompleto
# (são mostrados apenas como referência).
#
# A base de dados "Vacinas" (nº de doses) e "Óbito" é lida de
# New_pacientes703.xlsx (versão mais recente dos dados). Como esse
# arquivo não traz o fabricante da vacina — necessário para identificar
# o caso de dose única (Janssen) —, o fabricante é obtido do
# 703pacientes.xlsx (versão anterior), cruzado por "Registro".
# A mesma análise é refeita inteiramente a partir do 703pacientes.xlsx
# (dados de doses e fabricante da própria planilha), como segunda versão,
# para checar se a conclusão se mantém.
# ─────────────────────────────────────────────────────────────────────────
XLSX_NEW = "New_pacientes703.xlsx"
XLSX_OLD = "703pacientes.xlsx"
OUTPUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "esquema_vacinal_completo_vs_incompleto.png")
OUTPUT_TIFF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "esquema_vacinal_completo_vs_incompleto.tiff")

BG = "#FFFFFF"
TEXT = "#1F2328"
SUBTEXT = "#57606A"
COR_COMPLETO = "#1D9E75"
COR_INCOMPLETO = "#E0703A"
COR_NAO_VAC = "#9AA3AC"


def classifica_esquema(vacinas, fabricante, vacinado):
    """Completo = >=2 doses, ou 1 dose exclusivamente de Janssen."""
    if vacinado == 0 or vacinas == 0:
        return "Não vacinado"
    janssen_unico = isinstance(fabricante, str) and fabricante.strip() == "Janssen"
    completo = (vacinas >= 2) or (vacinas == 1 and janssen_unico)
    return "Completo" if completo else "Incompleto"


def formata_p(p):
    return "p<0,001" if p < 0.001 else f"p={p:.3f}".replace(".", ",")


# ════════════════════════════════════════════════════════════
# 1. CARREGAR E CLASSIFICAR AS DUAS VERSÕES DA BASE
# ════════════════════════════════════════════════════════════
df_old = pd.read_excel(XLSX_OLD)
df_new = pd.read_excel(XLSX_NEW)

# Versão "703pacientes.xlsx" (doses e fabricante da própria planilha)
versao_old = df_old.copy()
versao_old["Esquema"] = [
    classifica_esquema(v, f, vc) for v, f, vc in
    zip(versao_old["Vacinas"], versao_old["Fabricante_"], versao_old["Vacinado"])
]

# Versão "New_pacientes703.xlsx" (doses/óbito atualizados; fabricante
# obtido do 703pacientes.xlsx via Registro, só para identificar Janssen)
fabricante_por_registro = df_old.drop_duplicates("Registro").set_index("Registro")["Fabricante_"]
versao_new = df_new.copy()
versao_new["Fabricante_"] = versao_new["Registro"].map(fabricante_por_registro)
versao_new["Esquema"] = [
    classifica_esquema(v, f, vc) for v, f, vc in
    zip(versao_new["Vacinas"], versao_new["Fabricante_"], versao_new["Vacinado"])
]

VERSOES = {
    "New_pacientes703.xlsx\n(doses/óbito atualizados)": versao_new,
    "703pacientes.xlsx\n(versão anterior)": versao_old,
}


def estatisticas(d):
    sub = d[d["Esquema"].isin(["Completo", "Incompleto"])].copy()
    sub["Completo_bin"] = (sub["Esquema"] == "Completo").astype(int)

    tab = pd.crosstab(sub["Esquema"], sub["Óbito"]).reindex(
        index=["Completo", "Incompleto"], columns=[0, 1], fill_value=0)
    n_completo = int(tab.loc["Completo"].sum())
    n_incompleto = int(tab.loc["Incompleto"].sum())
    ob_completo = int(tab.loc["Completo", 1])
    ob_incompleto = int(tab.loc["Incompleto", 1])

    modelo = smf.glm("Óbito ~ Completo_bin", data=sub,
                      family=sm.families.Binomial()).fit()
    orr = float(np.exp(modelo.params["Completo_bin"]))
    ci = np.exp(modelo.conf_int().loc["Completo_bin"]).values
    p_logit = float(modelo.pvalues["Completo_bin"])

    _, p_fisher = fisher_exact(
        [[ob_completo, n_completo - ob_completo],
         [ob_incompleto, n_incompleto - ob_incompleto]])

    n_nao_vac = int((d["Esquema"] == "Não vacinado").sum())
    ob_nao_vac = int(d.loc[d["Esquema"] == "Não vacinado", "Óbito"].sum())

    return dict(
        n_completo=n_completo, ob_completo=ob_completo,
        pct_completo=100 * ob_completo / n_completo,
        n_incompleto=n_incompleto, ob_incompleto=ob_incompleto,
        pct_incompleto=100 * ob_incompleto / n_incompleto,
        n_nao_vac=n_nao_vac, ob_nao_vac=ob_nao_vac,
        pct_nao_vac=100 * ob_nao_vac / n_nao_vac if n_nao_vac else np.nan,
        OR=orr, IC=ci, p_logit=p_logit, p_fisher=p_fisher,
    )


resultados = {nome: estatisticas(d) for nome, d in VERSOES.items()}

print("=" * 90)
for nome, r in resultados.items():
    print(nome.replace("\n", " "))
    print(f"  Completo:    n={r['n_completo']:3d}  óbitos={r['ob_completo']:3d} "
          f"({r['pct_completo']:.1f}%)")
    print(f"  Incompleto:  n={r['n_incompleto']:3d}  óbitos={r['ob_incompleto']:3d} "
          f"({r['pct_incompleto']:.1f}%)")
    print(f"  Não vacinado: n={r['n_nao_vac']:3d}  óbitos={r['ob_nao_vac']:3d} "
          f"({r['pct_nao_vac']:.1f}%)")
    print(f"  OR (Completo vs Incompleto) = {r['OR']:.3f} "
          f"[{r['IC'][0]:.3f}-{r['IC'][1]:.3f}]  "
          f"p(logística)={r['p_logit']:.4g}  p(Fisher)={r['p_fisher']:.4g}")
print("=" * 90)

# ════════════════════════════════════════════════════════════
# 2. FIGURA: LINE PLOT — % óbito por grupo (Não vacinado → Incompleto →
#    Completo), uma linha por versão da base de dados
# ════════════════════════════════════════════════════════════
CORES_VERSAO = {
    "New_pacientes703.xlsx\n(doses/óbito atualizados)": "#1D9E75",
    "703pacientes.xlsx\n(versão anterior)": "#3A6EA5",
}
MARCADORES_VERSAO = {
    "New_pacientes703.xlsx\n(doses/óbito atualizados)": "o",
    "703pacientes.xlsx\n(versão anterior)": "s",
}

categorias = ["Não vacinado", "Incompleto", "Completo"]
x = np.arange(len(categorias))

fig, ax = plt.subplots(figsize=(10.5, 7), facecolor=BG)

for nome, r in resultados.items():
    y = [r["pct_nao_vac"], r["pct_incompleto"], r["pct_completo"]]
    ns = [r["n_nao_vac"], r["n_incompleto"], r["n_completo"]]
    ob = [r["ob_nao_vac"], r["ob_incompleto"], r["ob_completo"]]
    cor = CORES_VERSAO[nome]
    marcador = MARCADORES_VERSAO[nome]

    ax.plot(x, y, color=cor, marker=marcador, markersize=9, linewidth=2.4,
            label=nome.replace("\n", " "), zorder=3)
    for xi, yi, n, o in zip(x, y, ns, ob):
        deslocamento = 14 if nome.startswith("New") else -14
        va = "bottom" if nome.startswith("New") else "top"
        ax.annotate(f"{yi:.1f}%", (xi, yi), textcoords="offset points",
                    xytext=(0, deslocamento), ha="center", va=va,
                    fontsize=11, fontweight="bold", color=cor)

ax.set_xticks(x)
ax.set_xticklabels(categorias, fontsize=12.5, color=TEXT)
ax.set_ylabel("Óbito hospitalar (%)", fontsize=12, color=SUBTEXT)
ax.set_ylim(0, 95)
ax.set_xlim(-0.3, len(categorias) - 0.7)
ax.grid(axis="y", alpha=0.25, zorder=0)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.tick_params(axis="y", labelsize=10, colors=SUBTEXT)

# n= / óbitos= de cada ponto (uma versão por vez, para não poluir)
nome_new = [n for n in resultados if n.startswith("New")][0]
for xi, n, o in zip(x, [resultados[nome_new]["n_nao_vac"],
                        resultados[nome_new]["n_incompleto"],
                        resultados[nome_new]["n_completo"]],
                    [resultados[nome_new]["ob_nao_vac"],
                     resultados[nome_new]["ob_incompleto"],
                     resultados[nome_new]["ob_completo"]]):
    ax.text(xi, -0.14, f"n={n}\nóbitos={o}", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9, color=SUBTEXT)

ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=1, frameon=False,
          fontsize=10.5, labelcolor=TEXT)

caixa = dict(boxstyle="round,pad=0.45", fc="white", ec="#D0D7DE",
             alpha=0.95, linewidth=0.8)
texto_caixa = "Esquema completo vs. incompleto (regressão logística):\n" + "\n".join(
    f"{nome.split(chr(10))[0]}: OR={r['OR']:.2f} "
    f"({r['IC'][0]:.2f}–{r['IC'][1]:.2f}), {formata_p(r['p_logit'])}"
    for nome, r in resultados.items())
ax.text(0.98, 0.97, texto_caixa, transform=ax.transAxes, ha="right", va="top",
        fontsize=9.5, color=TEXT, bbox=caixa)

fig.suptitle("Esquema Vacinal Completo vs. Incompleto e Óbito Hospitalar",
             fontsize=17, fontweight="bold", color=TEXT, y=1.03)
fig.text(0.5, 0.975,
         "Completo = 2 doses (Pfizer/AstraZeneca/Coronavac) ou 1 dose "
         "(Janssen) | Incompleto = menos doses que o esquema completo",
         ha="center", va="top", fontsize=10.5, color=SUBTEXT)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight", facecolor=BG)

_buf_png = OUTPUT_PNG.replace(".png", "_300dpi_tmp.png")
plt.savefig(_buf_png, dpi=300, bbox_inches="tight", facecolor=BG)
Image.open(_buf_png).save(OUTPUT_TIFF, dpi=(300, 300), compression="tiff_lzw")
os.remove(_buf_png)

print(f"Gráfico salvo em: {OUTPUT_PNG} e {OUTPUT_TIFF}")
plt.show()
