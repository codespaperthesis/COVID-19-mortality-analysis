"""
PRISMA 2020 flow diagram of the systematic review (economic burden of Long COVID).
Full-text counts come from the review spreadsheet (Check_list_systematic_review_-_Price.xlsx):
  Planilha1  -> reports assessed for eligibility
  Excluidos  -> reports excluded at full text
  Incluidos  -> studies included in the review
Identification/screening counts are not in the spreadsheet: fill them in IDENTIFICATION below.
Generates PRISMA_flowchart_PT.(png|tiff) and PRISMA_flowchart_EN.(png|tiff).
"""
import os
import textwrap
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Check_list_systematic_review_-_Price.xlsx")
DPI = 600

# Counts before full-text assessment (database search, deduplication, title/abstract screening).
# None = not yet informed -> drawn as "n = ?". Replace with the real numbers from the search log.
IDENTIFICATION = {
    "databases": None,        # records identified from databases
    "registers": None,        # records identified from registers (0 if none)
    "duplicates": None,       # duplicate records removed
    "automation": None,       # records marked ineligible by automation tools (0 if none)
    "other_removed": None,    # records removed for other reasons (0 if none)
    "screened": None,         # records screened (title/abstract)
    "screen_excluded": None,  # records excluded at title/abstract
    "not_retrieved": None,    # reports not retrieved
}

# Reason for each full-text exclusion (sheet "Excluidos"), summarised from the reviewers' notes.
EXCLUSION_REASON = {
    "The economic burden of long COVID in Australia": "analog",
    "Estimating the economic burden of long-Covid": "separate",
    "Inequalities in worsening work and income": "no_costs",
    "A preliminary estimate of the economic impact of long COVID": "preliminary",
    "Budget impact of oral nirmatrelvir/ritonavir": "budget",
}

TEXT, SUBTEXT, BORDER = "#1F2328", "#57606A", "#8C959F"
FILL = {"main": "#EAF2FB", "side": "#F6F8FA", "included": "#E3F4EC", "stage": "#1F4E79"}

LANG = {
    "PT": dict(
        title="Identificação de estudos via bases de dados e registros",
        stages=["Identificação", "Triagem", "Incluídos"],
        identified="Registros identificados:\nBases de dados (n = {databases})\nRegistros (n = {registers})",
        removed="Registros removidos antes da triagem:\nDuplicatas (n = {duplicates})\n"
                "Marcados como inelegíveis por ferramentas\nde automação (n = {automation})\n"
                "Removidos por outros motivos (n = {other_removed})",
        screened="Registros triados\n(título e resumo)\n(n = {screened})",
        screen_excl="Registros excluídos\n(n = {screen_excluded})",
        sought="Relatórios buscados para recuperação\n(n = {sought})",
        not_retr="Relatórios não recuperados\n(n = {not_retrieved})",
        assessed="Relatórios avaliados quanto à\nelegibilidade (texto completo)\n(n = {assessed})",
        excluded="Relatórios excluídos (n = {excluded}):",
        included="Estudos incluídos na revisão\n(n = {included})\n{countries} países",
        reasons={
            "no_costs": "Sem avaliação de custos ou de uso de serviços de saúde",
            "analog": "Custos estimados a partir de condição análoga (EM/SFC)",
            "separate": "Destinado a análise separada",
            "preliminary": "Estimativa preliminar",
            "budget": "Modelo de impacto orçamentário",
        },
    ),
    "EN": dict(
        title="Identification of studies via databases and registers",
        stages=["Identification", "Screening", "Included"],
        identified="Records identified from:\nDatabases (n = {databases})\nRegisters (n = {registers})",
        removed="Records removed before screening:\nDuplicate records (n = {duplicates})\n"
                "Records marked as ineligible by\nautomation tools (n = {automation})\n"
                "Records removed for other reasons (n = {other_removed})",
        screened="Records screened\n(title and abstract)\n(n = {screened})",
        screen_excl="Records excluded\n(n = {screen_excluded})",
        sought="Reports sought for retrieval\n(n = {sought})",
        not_retr="Reports not retrieved\n(n = {not_retrieved})",
        assessed="Reports assessed for eligibility\n(full text)\n(n = {assessed})",
        excluded="Reports excluded (n = {excluded}):",
        included="Studies included in review\n(n = {included})\n{countries} countries",
        reasons={
            "no_costs": "No assessment of costs or health service use",
            "analog": "Costs estimated from an analogous condition (ME/CFS)",
            "separate": "Allocated to a separate analysis",
            "preliminary": "Preliminary estimate",
            "budget": "Budget impact model",
        },
    ),
}


# ── Data ────────────────────────────────────────────────────────────────────
def titles(sheet, header):
    df = pd.read_excel(DATA_PATH, sheet_name=sheet, header=0 if header else None)
    s = df.iloc[:, 0].dropna().astype(str).str.replace("\xa0", " ").str.strip()
    return s[s != ""].tolist()


assessed = titles("Planilha1", header=True)
excluded = titles("Excluidos", header=False)
included = titles("Incluidos", header=False)
if len(assessed) != len(excluded) + len(included):
    print(f"WARNING: assessed ({len(assessed)}) != excluded ({len(excluded)}) + included ({len(included)})")

reason_keys = []
for t in excluded:
    key = next((v for k, v in EXCLUSION_REASON.items() if t.startswith(k)), None)
    if key is None:
        raise ValueError(f"No exclusion reason defined for: {t}")
    reason_keys.append(key)

# Number of countries is shared with the map script
from importlib import import_module
n_countries = import_module("13_world_map_included_studies").country_counts()["iso3"].nunique()


def fmt(v):
    return "?" if v is None else str(v)


def counts():
    c = {k: fmt(v) for k, v in IDENTIFICATION.items()}
    c["sought"] = fmt(len(assessed) + IDENTIFICATION["not_retrieved"]
                      if IDENTIFICATION["not_retrieved"] is not None else None)
    c.update(assessed=len(assessed), excluded=len(excluded), included=len(included),
             countries=n_countries)
    return c


# ── Drawing ─────────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, text, fill, size=8.2, align="center", weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=fill, ec=BORDER, lw=0.9))
    tx = x + w / 2 if align == "center" else x + 0.15
    ax.text(tx, y + h / 2, text, ha=align, va="center", fontsize=size, color=TEXT,
            weight=weight, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=SUBTEXT, lw=1.0, mutation_scale=10,
                                shrinkA=0, shrinkB=0))


def stage(ax, y, h, label):
    ax.add_patch(FancyBboxPatch((0.1, y), 0.45, h, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=FILL["stage"], ec="none"))
    ax.text(0.325, y + h / 2, label, rotation=90, ha="center", va="center",
            fontsize=9, color="white", weight="bold")


def draw(lang):
    L = LANG[lang]
    c = counts()
    fig, ax = plt.subplots(figsize=(8.6, 10.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14.2)
    ax.axis("off")

    MX, MW = 0.85, 4.9    # main column
    SX, SW = 6.55, 5.3    # side column
    ax.add_patch(FancyBboxPatch((MX, 13.35), SX + SW - MX, 0.65, boxstyle="round,pad=0,rounding_size=0.08",
                                fc="#D6E4F0", ec="none"))
    ax.text((MX + SX + SW) / 2, 13.675, L["title"], ha="center", va="center",
            fontsize=10, color=TEXT, weight="bold")

    # Identification
    stage(ax, 10.6, 2.5, L["stages"][0])
    box(ax, MX, 11.0, MW, 1.7, L["identified"].format(**c), FILL["main"])
    box(ax, SX, 10.6, SW, 2.5, L["removed"].format(**c), FILL["side"], align="left")
    arrow(ax, MX + MW, 11.85, SX, 11.85)
    arrow(ax, MX + MW / 2, 11.0, MX + MW / 2, 9.75)

    # Screening
    stage(ax, 3.05, 7.05, L["stages"][1])
    box(ax, MX, 8.65, MW, 1.1, L["screened"].format(**c), FILL["main"])
    box(ax, SX, 8.75, SW, 0.9, L["screen_excl"].format(**c), FILL["side"])
    arrow(ax, MX + MW, 9.2, SX, 9.2)
    arrow(ax, MX + MW / 2, 8.65, MX + MW / 2, 7.65)

    box(ax, MX, 6.75, MW, 0.9, L["sought"].format(**c), FILL["main"])
    box(ax, SX, 6.75, SW, 0.9, L["not_retr"].format(**c), FILL["side"])
    arrow(ax, MX + MW, 7.2, SX, 7.2)
    arrow(ax, MX + MW / 2, 6.75, MX + MW / 2, 5.65)

    reasons = pd.Series(reason_keys).value_counts(sort=False)
    lines = [L["excluded"].format(**c)]
    for key, n in reasons.items():
        wrapped = textwrap.wrap(L["reasons"][key], 44)
        lines.append(f"•  {wrapped[0]} (n = {n})" if len(wrapped) == 1 else f"•  {wrapped[0]}")
        for i, part in enumerate(wrapped[1:], 1):
            lines.append(f"    {part}" + (f" (n = {n})" if i == len(wrapped) - 1 else ""))
    box(ax, MX, 4.35, MW, 1.3, L["assessed"].format(**c), FILL["main"])
    box(ax, SX, 3.05, SW, 3.25, "\n".join(lines), FILL["side"], size=7.8, align="left")
    arrow(ax, MX + MW, 5.0, SX, 5.0)
    arrow(ax, MX + MW / 2, 4.35, MX + MW / 2, 2.6)

    # Included
    stage(ax, 1.1, 1.5, L["stages"][2])
    box(ax, MX, 1.1, MW, 1.5, L["included"].format(**c), FILL["included"], size=8.8, weight="bold")

    fig.tight_layout()
    for ext in ("png", "tiff"):
        out = os.path.join(BASE_DIR, f"PRISMA_flowchart_{lang}.{ext}")
        kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tiff" else {}
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white", **kw)
        print("saved", out)
    plt.close(fig)


if __name__ == "__main__":
    missing = [k for k, v in IDENTIFICATION.items() if v is None]
    if missing:
        print("NOTE: identification/screening counts not informed (drawn as 'n = ?'):", ", ".join(missing))
    print(f"assessed={len(assessed)} excluded={len(excluded)} included={len(included)} countries={n_countries}")
    for lang in LANG:
        draw(lang)
