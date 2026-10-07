"""
Outcome-oriented versions (death vs. discharge) of the long-COVID temporal and
pharmacotherapeutic figures (scripts 12 and 13).

Temporal phases follow the short/post-acute/chronic scheme (3 and 12 weeks), but
here they are defined by HOSPITAL time only (`Dias_permanência`, mother sheet),
for both outcomes. Post-discharge lab follow-up exists only for survivors, so
using it to compare deaths with discharges would create immortal-time bias.
Only 1 admission exceeds 12 weeks, so the phases compared are ≤ 3 weeks and
> 3 weeks of hospital stay.

Figure 1 — Figure_desfecho_temporal_{PT,EN} (n = 703 admissions, mother sheet)
    A  hospital trajectory of each admission, by outcome
    B  in-hospital mortality by length-of-stay bin (Wilson 95% CI)
    C  OR for death of clinical variables (age-adjusted), overall and per phase
    D  admission labs (first result ≤ 72 h): % out of range, death vs discharge
Figure 2 — Figure_desfecho_farmaco_{PT,EN} (n = 508 patients of the prescription
sheet, admission/stay/outcome from the mother sheet; see script 13)
    A  prevalence of drug classes by outcome, overall and per phase
    B  OR for death, crude and adjusted (age, critical COVID, log(1 + hospital days))
    C  polypharmacy by outcome and phase
    D  mortality among users vs. non-users across length-of-stay bins

Tables: desfecho_temporal_clinico.csv, desfecho_exames_admissao.csv,
desfecho_mortalidade_permanencia.csv, desfecho_farmaco_classes.csv,
desfecho_farmaco_polifarmacia.csv.

Usage: python 14_outcome_association_temporal_pharmaco.py [--data-dir DIR] [--out-dir DIR]
"""
import argparse
import importlib.util
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import fisher_exact, mannwhitneyu
import statsmodels.api as sm

sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(BASE_DIR, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _load_module("temporal", "12_long_covid_temporal_profile.py")
P = _load_module("pharmaco", "13_long_covid_pharmacotherapy.py")

DPI = 600
TEXT, SUBTEXT, GRID = T.TEXT, T.SUBTEXT, T.GRID
OUT = {"alta": "#1D9E75", "obito": "#E07B39"}            # outcome colours used in the repo's Figure 2
PHASE = {"all": TEXT, "acute": "#2a78d6", "prol": "#4a3aa7"}
ACUTE_MAX = T.SHORT_MAX                                   # 21 days
LOS_BINS = [-0.5, 1, 3, 7, 14, 21, 42, 90]
LOS_BINS_D = [-0.5, 7, 21, 42, 90]
MIN_TESTED = 15

CLIN = [("age61", "Idade ≥ 61 anos", "Age ≥ 61 years"), ("sex1", "Sexo (código 1)", "Sex (code 1)"),
        ("vac", "Vacinado", "Vaccinated"), ("crit", "COVID crítica", "Critical COVID"),
        ("srag", "SRAG", "SARS"), ("card", "Prob. cardíaco", "Cardiac disease"),
        ("diab", "Diabetes", "Diabetes"), ("resp", "Prob. respiratório", "Respiratory disease"),
        ("renal", "Prob. renal", "Renal disease"), ("lra", "LRA", "AKI"),
        ("neuro", "Prob. neurológico", "Neurological disease"), ("cancer", "Câncer", "Cancer"),
        ("choque", "Choque", "Shock")]

TXT = {
    "PT": dict(
        out={"alta": "Alta", "obito": "Óbito"},
        ph={"all": "Total", "acute": "≤ 3 sem", "prol": "> 3 sem"},
        t1="Trajetória temporal e desfecho hospitalar (óbito × alta)",
        t2="Perfil farmacoterapêutico e desfecho hospitalar (óbito × alta)",
        A1="A  Tempo de internação de cada paciente, por desfecho (n = {n})",
        B1="B  Mortalidade por tempo de internação",
        C1="C  Variáveis clínicas: OR de óbito por fase",
        D1="D  Exames da admissão (≤ 72 h) e óbito",
        A2="A  Prevalência das classes por desfecho e fase (%)",
        B2="B  Associação com óbito",
        C2="C  Polifarmácia por desfecho",
        D2="D  Mortalidade em usuários × não usuários, por tempo de internação",
        xlos="Dias de internação (escala raiz quadrada)", xbin="Dias de internação",
        ymort="Óbito hospitalar (%)", overall="Mortalidade geral {m}%",
        orlab="OR de óbito (IC 95%), escala log", oradj="OR aj. (IC 95%)", padj="p aj.",
        c1note="Ajustado por idade (exceto Idade ≥ 61). Fase = tempo de internação.",
        d1x="Pacientes com 1º resultado fora da referência (%)", d1hdr="OR (IC 95%)",
        d1note="Primeiro resultado de cada exame até 72 h da admissão; OR de Fisher.",
        crude="bruto", adjusted="ajustado",
        b2note="Ajustado por idade, COVID crítica e log(1 + dias de internação)",
        yC="Princípios ativos distintos por paciente",
        users="Usuários", nonusers="Não usuários", ymortD="Óbito (%)",
        week3="3 sem", week12="12 sem",
        foot1=("Fases definidas apenas pelo tempo de internação (planilha mãe) para óbitos e altas, evitando viés de "
               "tempo imortal do seguimento pós-alta; só 1 internação passou de 12 semanas, por isso as fases comparadas "
               "são ≤ 3 e > 3 semanas.\nSexo conforme codificação da planilha mãe. Associação não implica causalidade."),
        foot2=("Coorte: {n} pacientes da planilha de prescrição; admissão, internação e desfecho da planilha mãe. "
               "Prescrições sem data: a exposição pode ter ocorrido em qualquer momento da internação, por isso o "
               "ajuste por tempo de internação\ne a estratificação por faixas de permanência (painel D). Associação "
               "não implica efeito causal (indicação por gravidade)."),
    ),
    "EN": dict(
        out={"alta": "Discharge", "obito": "Death"},
        ph={"all": "Overall", "acute": "≤ 3 wk", "prol": "> 3 wk"},
        t1="Temporal trajectory and hospital outcome (death vs. discharge)",
        t2="Pharmacotherapeutic profile and hospital outcome (death vs. discharge)",
        A1="A  Length of stay of each patient, by outcome (n = {n})",
        B1="B  Mortality by length of stay",
        C1="C  Clinical variables: OR for death by phase",
        D1="D  Admission labs (≤ 72 h) and death",
        A2="A  Prevalence of drug classes by outcome and phase (%)",
        B2="B  Association with death",
        C2="C  Polypharmacy by outcome",
        D2="D  Mortality in users vs. non-users, by length of stay",
        xlos="Hospital days (square-root scale)", xbin="Hospital days",
        ymort="In-hospital death (%)", overall="Overall mortality {m}%",
        orlab="OR for death (95% CI), log scale", oradj="Adj. OR (95% CI)", padj="adj. p",
        c1note="Adjusted for age (except Age ≥ 61). Phase = hospital stay.",
        d1x="Patients with out-of-range first result (%)", d1hdr="OR (95% CI)",
        d1note="First result of each test within 72 h of admission; Fisher OR.",
        crude="crude", adjusted="adjusted",
        b2note="Adjusted for age, critical COVID and log(1 + hospital days)",
        yC="Distinct active ingredients per patient",
        users="Users", nonusers="Non-users", ymortD="Death (%)",
        week3="3 wk", week12="12 wk",
        foot1=("Phases defined by hospital stay only (mother sheet) for deaths and discharges, avoiding immortal-time "
               "bias from post-discharge follow-up; only 1 admission exceeded 12 weeks, so the phases compared are "
               "≤ 3 and > 3 weeks.\nSex as coded in the mother sheet. Association does not imply causation."),
        foot2=("Cohort: {n} patients of the prescription sheet; admission, stay and outcome from the mother sheet. "
               "Prescriptions are undated: exposure may have occurred at any time during the stay, hence the "
               "adjustment for length of stay\nand the stratification by length-of-stay bins (panel D). Association "
               "does not imply a causal drug effect (confounding by severity)."),
    ),
}


# ----------------------------------------------------------------------------- stats helpers
def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h


def logit_or(df, x, covars, y="obito"):
    """OR of x for y with covariates; NaN when a cell is too small or the fit is unstable."""
    s = df[[y, x] + covars].dropna().astype(float)
    tab = pd.crosstab(s[x], s[y])
    if tab.shape != (2, 2) or (tab.values < 3).any():
        return dict(OR=np.nan, lo=np.nan, hi=np.nan, p=np.nan, n=len(s))
    try:
        fit = sm.Logit(s[y], sm.add_constant(s[[x] + covars])).fit(disp=0, maxiter=200)
        ci = fit.conf_int().loc[x]
        r = dict(OR=np.exp(fit.params[x]), lo=np.exp(ci[0]), hi=np.exp(ci[1]), p=fit.pvalues[x], n=len(s))
        if not np.isfinite(r["hi"]) or r["hi"] / r["lo"] > 2000:
            raise ValueError
        return r
    except Exception:
        return dict(OR=np.nan, lo=np.nan, hi=np.nan, p=np.nan, n=len(s))


def fisher_or(a_yes, a_n, b_yes, b_n):
    """OR (Woolf CI) of 'yes' in group a vs b."""
    tab = [[a_yes, a_n - a_yes], [b_yes, b_n - b_yes]]
    OR, p = fisher_exact(tab)
    lo, hi = P._woolf(tab)
    return OR, lo, hi, p


def fmt(v, dec, nd=2):
    return f"{v:.{nd}f}".replace(".", dec)


def fmt_p(p, dec):
    return T.fmt_p(p, dec) if np.isfinite(p) else "–"


def phase(los):
    return np.where(los <= ACUTE_MAX, "acute", "prol")


def style(ax, left=True):
    for s in ["top", "right"] + ([] if left else ["left"]):
        ax.spines[s].set_visible(False)


def setup_rc():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                         "axes.labelcolor": TEXT, "xtick.color": SUBTEXT, "ytick.color": SUBTEXT})


def save(fig, base):
    fig.savefig(base + ".png", dpi=200, facecolor="white")
    fig.savefig(base + ".tiff", dpi=DPI, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


# ----------------------------------------------------------------------------- figure 1 data
def temporal_tables(d, e):
    d = d.copy()
    d["fase"] = phase(d["los"])
    # B: mortality by LOS bin
    d["losbin"] = pd.cut(d["los"], LOS_BINS)
    mort = d.groupby("losbin", observed=False)["obito"].agg(["sum", "size"]).reset_index()
    mort[["lo", "hi"]] = [wilson(k, n) for k, n in zip(mort["sum"], mort["size"])]
    mort["pct"] = 100 * mort["sum"] / mort["size"]
    # C: clinical ORs by phase
    rows = []
    for key, pt, en in CLIN:
        cov = [] if key == "age61" else ["Idade"]
        r = {"var": key, "PT": pt, "EN": en}
        for ph, sub in [("all", d), ("acute", d[d["fase"] == "acute"]), ("prol", d[d["fase"] == "prol"])]:
            o = logit_or(sub, key, cov)
            r.update({f"{ph}_{k}": v for k, v in o.items()})
            r[f"{ph}_pct_obito"] = 100 * sub.loc[sub["obito"] == 1, key].mean()
            r[f"{ph}_pct_alta"] = 100 * sub.loc[sub["obito"] == 0, key].mean()
        rows.append(r)
    clin = pd.DataFrame(rows)
    # D: admission labs (first result within 72 h)
    adm = e[(e["t"] >= 0) & (e["t"] <= 3)].sort_values("t").groupby(["ep", "VARIAVEL"]).first().reset_index()
    adm = adm.merge(d[["ep", "obito"]], on="ep", how="inner", suffixes=("_x", ""))
    rows = []
    for var, s in adm.groupby("VARIAVEL"):
        dd, aa = s[s["obito"] == 1], s[s["obito"] == 0]
        if len(dd) < MIN_TESTED or len(aa) < MIN_TESTED:
            continue
        OR, lo, hi, p = fisher_or(dd["alt"].sum(), len(dd), aa["alt"].sum(), len(aa))
        rows.append(dict(marcador=var, n_obito=len(dd), n_alta=len(aa),
                         pct_obito=100 * dd["alt"].mean(), pct_alta=100 * aa["alt"].mean(),
                         OR=OR, lo=lo, hi=hi, p=p))
    labs = pd.DataFrame(rows).sort_values("OR", ascending=False).reset_index(drop=True)
    return d, mort, clin, labs


def figure1(d, mort, clin, labs, lang, base):
    L, Lt = TXT[lang], T.LANG[lang]
    dec = Lt["dec"]
    setup_rc()
    fig = plt.figure(figsize=(19, 15), facecolor="white")
    fig.text(0.02, 0.975, L["t1"], fontsize=14, fontweight="bold", color=TEXT)

    # ---------- A: swimmer by outcome
    ax = fig.add_axes([0.06, 0.60, 0.76, 0.32])
    SQ = T.SQ
    y0, gap, spans = 0, 25, {}
    for o, flag in [("alta", 0), ("obito", 1)]:
        sub = d[d["obito"] == flag].sort_values("los").reset_index(drop=True)
        ys = np.arange(y0, y0 + len(sub))
        ax.hlines(-ys, 0, SQ(sub["los"].clip(lower=0.15)), color=OUT[o], lw=0.9)
        if o == "obito":
            ax.scatter(SQ(sub["los"].clip(lower=0.15)), -ys, marker="x", s=5, lw=0.6, color=TEXT, zorder=3)
        spans[o] = (y0, y0 + len(sub))
        n_ac, n_pr = int((sub["fase"] == "acute").sum()), int((sub["fase"] == "prol").sum())
        ax.text(SQ(90) * 1.01, -(y0 + y0 + len(sub)) / 2,
                f"{L['out'][o]}\nn = {len(sub)}\n{L['ph']['acute']}: {n_ac}\n{L['ph']['prol']}: {n_pr}",
                va="center", fontsize=9.5, color=TEXT, fontweight="bold")
        y0 += len(sub) + gap
    for xv, lab in [(ACUTE_MAX, L["week3"]), (T.POST_MAX, L["week12"])]:
        ax.axvline(SQ(xv), color=TEXT, ls=(0, (4, 3)), lw=1.1)
        ax.text(SQ(xv), 12, lab, ha="center", va="bottom", fontsize=10, fontweight="bold", color=TEXT)
    ax.axvline(0, color=TEXT, lw=1.4)
    ticks = [0, 1, 3, 7, 14, 21, 42, 84]
    ax.set_xticks(SQ(ticks)); ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlim(-0.03, SQ(90)); ax.set_ylim(-y0 + gap - 10, 60)
    ax.set_yticks([]); ax.set_xlabel(L["xlos"]); style(ax, left=False)
    # phase mortality annotations
    for ph, xm in [("acute", SQ(ACUTE_MAX) / 2), ("prol", (SQ(ACUTE_MAX) + SQ(90)) / 2)]:
        s = d[d["fase"] == ph]
        m = 100 * s["obito"].mean()
        ax.text(xm, 14, f"{L['ymort'].split(' (')[0]}: {fmt(m, dec, 1)}% ({int(s['obito'].sum())}/{len(s)})",
                ha="center", va="bottom", fontsize=9, color=TEXT,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=GRID))
    fig.text(0.06, 0.94, L["A1"].format(n=len(d)), fontsize=11.5, fontweight="bold", color=TEXT)

    # ---------- B: mortality by LOS bin
    ax = fig.add_axes([0.06, 0.12, 0.22, 0.36])
    labels = ["0–1", "2–3", "4–7", "8–14", "15–21", "22–42", "43–87"]
    xs = np.arange(len(mort))
    ax.bar(xs, mort["pct"], color=OUT["obito"], width=0.72, edgecolor="white", lw=1.5)
    ax.errorbar(xs, mort["pct"], yerr=[mort["pct"] - 100 * mort["lo"], 100 * mort["hi"] - mort["pct"]],
                fmt="none", ecolor=TEXT, elinewidth=1, capsize=3)
    for x, (_, r) in zip(xs, mort.iterrows()):
        ax.text(x, 100 * r["hi"] + 2, f"{r['pct']:.0f}%", ha="center", fontsize=8, color=TEXT, fontweight="bold")
        ax.text(x, -9, f"{int(r['sum'])}/{int(r['size'])}", ha="center", fontsize=7, color=SUBTEXT)
    m_all = 100 * d["obito"].mean()
    ax.axhline(m_all, color=SUBTEXT, ls=(0, (2, 2)), lw=1)
    ax.text(-0.4, 96, "- - " + L["overall"].format(m=fmt(m_all, dec, 1)), ha="left", fontsize=8, color=SUBTEXT)
    ax.axvline(4.5, color=TEXT, ls=(0, (4, 3)), lw=1)
    ax.text(4.5, 101, L["week3"], ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=TEXT)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(-12, 100); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel(L["ymort"])
    ax.set_xlabel(L["xbin"], labelpad=20); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.spines["bottom"].set_position(("data", 0)); style(ax)
    fig.text(0.03, 0.52, L["B1"], fontsize=11, fontweight="bold", color=TEXT)

    # ---------- C: clinical forest by phase
    ax = fig.add_axes([0.43, 0.12, 0.17, 0.36])
    tr = ax.get_yaxis_transform()
    n = len(clin)
    yy = np.arange(n)
    for yi, (_, r) in zip(yy, clin.iterrows()):
        ax.axhline(yi, color=GRID, lw=0.4, zorder=0)
        for ph, dy, mk in [("all", -0.25, "o"), ("acute", 0, "s"), ("prol", 0.25, "D")]:
            if np.isfinite(r[f"{ph}_OR"]):
                sig = r[f"{ph}_p"] < 0.05
                ax.plot([r[f"{ph}_lo"], r[f"{ph}_hi"]], [yi + dy] * 2, color=PHASE[ph], lw=1.2)
                ax.scatter(r[f"{ph}_OR"], yi + dy, marker=mk, s=26, color=PHASE[ph] if sig else "white",
                           edgecolor=PHASE[ph], lw=1, zorder=3)
        ax.text(-0.04, yi, r[lang], transform=tr, ha="right", va="center", fontsize=8.5, color=TEXT)
        sig = r["all_p"] < 0.05
        txt = (f"{fmt(r['all_OR'], dec)} ({fmt(r['all_lo'], dec)}–{fmt(r['all_hi'], dec)})"
               if np.isfinite(r["all_OR"]) else "–")
        ax.text(1.04, yi, txt, transform=tr, va="center", fontsize=7.5, color=TEXT if sig else SUBTEXT,
                fontweight="bold" if sig else None)
        ax.text(1.62, yi, fmt_p(r["all_p"], dec), transform=tr, va="center", fontsize=7.5,
                color=TEXT if sig else SUBTEXT, fontweight="bold" if sig else None)
    ax.text(1.04, -1.0, f"{L['oradj']} – {L['ph']['all']}", transform=tr, fontsize=7.5, fontweight="bold", color=TEXT)
    ax.text(1.62, -1.0, L["padj"], transform=tr, fontsize=7.5, fontweight="bold", color=TEXT)
    ax.axvline(1, color=TEXT, lw=0.8, ls=(0, (3, 2)))
    ax.set_xscale("log"); ax.set_xlim(0.1, 100); ax.set_ylim(n - 0.4, -1.2)
    ax.set_xticks([0.1, 0.3, 1, 3, 10, 30, 100]); ax.set_xticklabels(["0.1", "0.3", "1", "3", "10", "30", "100"])
    ax.set_yticks([]); ax.set_xlabel(L["orlab"]); style(ax, left=False)
    ax.legend(handles=[Line2D([], [], color=PHASE[p], marker=m, ls="-", label=f"{L['ph'][p]} (n={c})")
                       for p, m, c in [("all", "o", len(d)), ("acute", "s", int((d['fase'] == 'acute').sum())),
                                       ("prol", "D", int((d['fase'] == 'prol').sum()))]],
              loc="upper left", bbox_to_anchor=(-0.9, -0.09), ncol=3, frameon=False, fontsize=8)
    fig.text(0.33, 0.035, L["c1note"], fontsize=7.5, color=SUBTEXT)
    fig.text(0.33, 0.52, L["C1"], fontsize=11, fontweight="bold", color=TEXT)

    # ---------- D: admission labs dumbbell
    ax = fig.add_axes([0.80, 0.12, 0.10, 0.36])
    tr = ax.get_yaxis_transform()
    n = len(labs)
    yy = np.arange(n)
    for yi, (_, r) in zip(yy, labs.iterrows()):
        ax.axhline(yi, color=GRID, lw=0.4, zorder=0)
        ax.plot([r["pct_alta"], r["pct_obito"]], [yi, yi], color=SUBTEXT, lw=1.2, zorder=1)
        ax.scatter(r["pct_alta"], yi, s=30, color=OUT["alta"], marker="o", edgecolor="white", lw=0.6, zorder=3)
        ax.scatter(r["pct_obito"], yi, s=34, color=OUT["obito"], marker="X", edgecolor="white", lw=0.6, zorder=3)
        name = r["marcador"] if lang == "PT" else T.MARKER_EN.get(r["marcador"], r["marcador"])
        ax.text(-0.05, yi, name, transform=tr, ha="right", va="center", fontsize=8, color=TEXT)
        sig = r["p"] < 0.05
        ax.text(1.05, yi, f"{fmt(r['OR'], dec)} ({fmt(r['lo'], dec)}–{fmt(r['hi'], dec)})", transform=tr,
                va="center", fontsize=7.2, color=TEXT if sig else SUBTEXT, fontweight="bold" if sig else None)
    ax.text(1.05, -1.0, L["d1hdr"], transform=tr, fontsize=7.5, fontweight="bold", color=TEXT)
    ax.set_xlim(0, 100); ax.set_ylim(n - 0.4, -1.2); ax.set_yticks([])
    ax.set_xticks([0, 50, 100]); ax.set_xlabel(L["d1x"], fontsize=8); style(ax, left=False)
    ax.legend(handles=[Line2D([], [], color=OUT["alta"], marker="o", ls="", label=L["out"]["alta"]),
                       Line2D([], [], color=OUT["obito"], marker="X", ls="", label=L["out"]["obito"])],
              loc="upper left", bbox_to_anchor=(-0.9, -0.09), ncol=2, frameon=False, fontsize=8)
    fig.text(0.66, 0.035, L["d1note"], fontsize=7.5, color=SUBTEXT)
    fig.text(0.66, 0.52, L["D1"], fontsize=11, fontweight="bold", color=TEXT)

    fig.text(0.02, 0.004, L["foot1"], fontsize=8, color=SUBTEXT, va="bottom")
    save(fig, base)


# ----------------------------------------------------------------------------- figure 2 data
def pharmaco_tables(p):
    p = p.copy()
    p["fase"] = phase(p["los"])
    cols = [("all", "alta"), ("all", "obito"), ("acute", "alta"), ("acute", "obito"),
            ("prol", "alta"), ("prol", "obito")]
    rows = []
    for key, dom, pt, en, _, _ in P.CLASSES:
        r = {"key": key, "domain": dom, "PT": pt, "EN": en}
        for ph, o in cols:
            s = p if ph == "all" else p[p["fase"] == ph]
            s = s[s["obito"] == (o == "obito")]
            r[f"{ph}_{o}"] = 100 * s[key].mean()
        users = p[key] == 1
        OR, lo, hi, pv = fisher_or(p.loc[users, "obito"].sum(), users.sum(),
                                   p.loc[~users, "obito"].sum(), (~users).sum())
        r.update(OR_crude=OR, OR_crude_lo=lo, OR_crude_hi=hi, p_crude=pv, users=int(users.sum()))
        o = logit_or(p, key, ["Idade", "crit", "log_los"])
        r.update(OR_adj=o["OR"], OR_adj_lo=o["lo"], OR_adj_hi=o["hi"], p_adj=o["p"])
        rows.append(r)
    ct = pd.DataFrame(rows)
    poly = []
    for ph in ["acute", "prol"]:
        s = p[p["fase"] == ph]
        a, b = s.loc[s["obito"] == 0, "n_ingred"], s.loc[s["obito"] == 1, "n_ingred"]
        poly.append(dict(fase=ph, n_alta=len(a), n_obito=len(b), med_alta=a.median(), med_obito=b.median(),
                         p=mannwhitneyu(a, b).pvalue))
    return p, ct, pd.DataFrame(poly)


def figure2(p, ct, poly, lang, base):
    L, Lt = TXT[lang], T.LANG[lang]
    dec = Lt["dec"]
    setup_rc()
    fig = plt.figure(figsize=(21, 16), facecolor="white")
    fig.text(0.02, 0.975, L["t2"], fontsize=14, fontweight="bold", color=TEXT)

    ypos, dom_rows, y = [], [], 0
    for dom in P.DOMAINS:
        dom_rows.append((dom, y)); y += 1.1
        for i in ct.index[ct["domain"] == dom]:
            ypos.append((i, y)); y += 1
        y += 0.35
    ymax = y

    # ---------- A: heatmap outcome × phase
    cols = [("all", "alta"), ("all", "obito"), ("acute", "alta"), ("acute", "obito"),
            ("prol", "alta"), ("prol", "obito")]
    xpos = [0, 1, 2.3, 3.3, 4.6, 5.6]
    axA = fig.add_axes([0.165, 0.10, 0.25, 0.80])
    for i, yy in ypos:
        for xp, (ph, o) in zip(xpos, cols):
            v = ct.loc[i, f"{ph}_{o}"]
            axA.add_patch(plt.Rectangle((xp + 0.04, yy - 0.46), 0.92, 0.92, color=P.HEAT(v / 100), lw=0))
            axA.text(xp + 0.5, yy, f"{v:.0f}", ha="center", va="center", fontsize=8,
                     color="white" if v >= 55 else TEXT)
        axA.text(-0.1, yy, ct.loc[i, lang], ha="right", va="center", fontsize=8.5, color=TEXT)
    for dom, yy in dom_rows:
        axA.text(-0.1, yy + 0.2, P.DOMAINS[dom][0 if lang == "PT" else 1].upper(), ha="right", va="center",
                 fontsize=8.5, fontweight="bold", color=SUBTEXT)
    for k, ph in enumerate(["all", "acute", "prol"]):
        s = p if ph == "all" else p[p["fase"] == ph]
        axA.text(xpos[2 * k] + 1, -2.15, L["ph"][ph], ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                 color=TEXT)
        axA.plot([xpos[2 * k] + 0.05, xpos[2 * k] + 1.95], [-2.0, -2.0], color=TEXT, lw=0.8, clip_on=False)
        for j, o in enumerate(["alta", "obito"]):
            nn = int((s["obito"] == (o == "obito")).sum())
            axA.text(xpos[2 * k + j] + 0.5, -0.85, f"{L['out'][o]}\n(n={nn})", ha="center", va="bottom",
                     fontsize=8, color=TEXT)
            axA.add_patch(plt.Rectangle((xpos[2 * k + j] + 0.04, -0.75), 0.92, 0.22, color=OUT[o], lw=0,
                                        clip_on=False))
    axA.set_xlim(0, 6.6); axA.set_ylim(ymax, -0.9); axA.axis("off")
    fig.text(0.02, 0.945, L["A2"], fontsize=11.5, fontweight="bold", color=TEXT)

    # ---------- B: forest OR for death
    axB = fig.add_axes([0.435, 0.10, 0.12, 0.80])
    tr = axB.get_yaxis_transform()
    for i, yy in ypos:
        r = ct.loc[i]
        axB.axhline(yy, color=GRID, lw=0.4, zorder=0)
        axB.plot([r["OR_crude_lo"], r["OR_crude_hi"]], [yy - 0.18] * 2, color=SUBTEXT, lw=1, alpha=0.6)
        axB.scatter(r["OR_crude"], yy - 0.18, s=16, color="white", edgecolor=SUBTEXT, lw=0.9, zorder=3)
        if np.isfinite(r["OR_adj"]):
            sig = r["p_adj"] < 0.05
            col = OUT["obito"] if r["OR_adj"] >= 1 else OUT["alta"]
            axB.plot([r["OR_adj_lo"], r["OR_adj_hi"]], [yy + 0.18] * 2, color=TEXT, lw=1.4)
            axB.scatter(r["OR_adj"], yy + 0.18, s=32, marker="s", color=col if sig else "white",
                        edgecolor=col, lw=1.1, zorder=3)
            axB.text(1.04, yy, f"{fmt(r['OR_adj'], dec)} ({fmt(r['OR_adj_lo'], dec)}–{fmt(r['OR_adj_hi'], dec)})",
                     transform=tr, va="center", fontsize=7.5, color=TEXT if sig else SUBTEXT,
                     fontweight="bold" if sig else None)
            axB.text(1.66, yy, fmt_p(r["p_adj"], dec), transform=tr, va="center", fontsize=7.5,
                     color=TEXT if sig else SUBTEXT, fontweight="bold" if sig else None)
        else:
            axB.text(1.04, yy, "–", transform=tr, va="center", fontsize=7.5, color=SUBTEXT)
    axB.axvline(1, color=TEXT, lw=0.8, ls=(0, (3, 2)))
    axB.set_xscale("log"); axB.set_xlim(0.03, 100); axB.set_ylim(ymax, -0.9)
    axB.set_xticks([0.1, 0.3, 1, 3, 10, 30]); axB.set_xticklabels(["0.1", "0.3", "1", "3", "10", "30"])
    axB.set_yticks([]); axB.set_xlabel(L["orlab"]); style(axB, left=False)
    axB.text(1.04, -0.3, L["oradj"], transform=tr, fontsize=8, fontweight="bold", color=TEXT)
    axB.text(1.66, -0.3, L["padj"], transform=tr, fontsize=8, fontweight="bold", color=TEXT)
    axB.text(0.04, -0.75, "← " + L["out"]["alta"], transform=tr, fontsize=8, color=OUT["alta"], fontweight="bold")
    axB.text(0.96, -0.75, L["out"]["obito"] + " →", transform=tr, fontsize=8, color=OUT["obito"],
             fontweight="bold", ha="right")
    axB.legend(handles=[Line2D([], [], color=SUBTEXT, marker="o", mfc="white", ls="-", label=L["crude"]),
                        Line2D([], [], color=TEXT, marker="s", mfc=OUT["obito"], mec=OUT["obito"], ls="-",
                               label=L["adjusted"])],
               loc="upper left", bbox_to_anchor=(-0.02, -0.045), ncol=2, frameon=False, fontsize=8)
    fig.text(0.435, 0.045, L["b2note"], fontsize=7.5, color=SUBTEXT, va="top")
    fig.text(0.435, 0.945, L["B2"], fontsize=11.5, fontweight="bold", color=TEXT)

    # ---------- C: polypharmacy by outcome × phase
    axC = fig.add_axes([0.73, 0.66, 0.25, 0.25])
    rng = np.random.default_rng(42)
    groups = [("acute", "alta"), ("acute", "obito"), ("prol", "alta"), ("prol", "obito")]
    xs = [0, 1, 2.5, 3.5]
    for x, (ph, o) in zip(xs, groups):
        v = p.loc[(p["fase"] == ph) & (p["obito"] == (o == "obito")), "n_ingred"].dropna().values
        axC.scatter(x + rng.uniform(-0.25, 0.25, len(v)), v, s=6, color=OUT[o], alpha=0.4, lw=0)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        axC.vlines(x, q1, q3, color=TEXT, lw=3); axC.hlines(med, x - 0.3, x + 0.3, color=TEXT, lw=2)
        axC.text(x + 0.33, med, f"{med:.0f}", va="center", fontsize=8.5, fontweight="bold", color=TEXT)
    top = p["n_ingred"].max()
    for k, (_, r) in enumerate(poly.iterrows()):
        xm = xs[2 * k] + 0.5
        axC.plot([xs[2 * k], xs[2 * k], xs[2 * k + 1], xs[2 * k + 1]], [top + 3, top + 5, top + 5, top + 3],
                 color=SUBTEXT, lw=0.8)
        axC.text(xm, top + 6, f"p {fmt_p(r['p'], dec)}", ha="center", fontsize=8, color=TEXT)
    axC.set_xticks(xs)
    axC.set_xticklabels([f"{L['out'][o]}\n{L['ph'][ph]}" for ph, o in groups], color=TEXT, fontsize=8.5)
    axC.set_ylim(0, top + 14); axC.set_ylabel(L["yC"])
    axC.grid(axis="y", color=GRID, lw=0.5); axC.set_axisbelow(True); style(axC)
    fig.text(0.70, 0.945, L["C2"], fontsize=11.5, fontweight="bold", color=TEXT)

    # ---------- D: mortality users vs non-users by LOS bin, top 8 classes by adjusted p
    sel = ct.dropna(subset=["p_adj"]).sort_values("p_adj").head(8)
    p["losbin"] = pd.cut(p["los"], LOS_BINS_D)
    blabels = ["0–7", "8–21", "22–42", ">42"]
    x0, y0, w, h, gx, gy = 0.70, 0.10, 0.11, 0.08, 0.045, 0.045
    fig.text(x0 - 0.035, 0.585, L["D2"], fontsize=11.5, fontweight="bold", color=TEXT)
    fig.legend(handles=[Patch(color=OUT["obito"], label=L["users"]), Patch(color="#B8B8B4", label=L["nonusers"])],
               loc="upper left", bbox_to_anchor=(x0 - 0.04, 0.58), ncol=2, frameon=False, fontsize=8)
    xb = np.arange(len(blabels))
    for k, (_, r) in enumerate(sel.iterrows()):
        rr, cc = divmod(k, 2)
        ax = fig.add_axes([x0 + cc * (w + gx), y0 + (3 - rr) * (h + gy), w, h])
        g = p.groupby(["losbin", p[r["key"]] == 1], observed=False)["obito"].mean().unstack()
        g = g.reindex(columns=[False, True])
        ax.bar(xb - 0.2, 100 * g[True].values, width=0.38, color=OUT["obito"], edgecolor="white", lw=0.8)
        ax.bar(xb + 0.2, 100 * g[False].values, width=0.38, color="#B8B8B4", edgecolor="white", lw=0.8)
        for xv, v1, v0 in zip(xb, g[True].values, g[False].values):
            for dx, v in [(-0.2, v1), (0.2, v0)]:
                if not np.isfinite(v) or v == 0:
                    ax.text(xv + dx, 2, "–" if not np.isfinite(v) else "0", ha="center", va="bottom",
                            fontsize=7.5, color=SUBTEXT, fontweight="bold")
        ax.axvline(1.5, color=TEXT, ls=(0, (3, 2)), lw=0.8)
        ax.set_xlim(-0.6, len(xb) - 0.4)
        ax.set_ylim(0, 100); ax.set_yticks([0, 50, 100]); ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True); style(ax)
        ax.set_title(f"{r[lang]}  (OR aj. {fmt(r['OR_adj'], dec)})" if lang == "PT"
                     else f"{r[lang]}  (adj. OR {fmt(r['OR_adj'], dec)})", loc="left", fontsize=7.4,
                     fontweight="bold", color=TEXT, pad=3)
        ax.set_xticks(xb)
        ax.set_xticklabels(blabels if rr == 3 else [], fontsize=7.5)
        if rr == 3:
            ax.set_xlabel(L["xbin"], fontsize=8)
        if cc == 0:
            ax.set_ylabel(L["ymortD"], fontsize=8)
    nb = p.groupby("losbin", observed=False).size().values
    fig.text(x0 - 0.035, y0 - 0.06, "n = " + " · ".join(map(str, nb)) +
             (" (por faixa); – = sem pacientes no subgrupo; 0 = nenhum óbito" if lang == "PT"
              else " (per bin); – = no patients in subgroup; 0 = no deaths"),
             fontsize=7.5, color=SUBTEXT)

    fig.text(0.02, 0.004, L["foot2"].format(n=len(p)), fontsize=8, color=SUBTEXT, va="bottom")
    save(fig, base)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=BASE_DIR)
    ap.add_argument("--out-dir", default=BASE_DIR)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    mae, ea, eb, presc = T.load(a.data_dir)
    d, e, _ = T.build(mae, ea, eb, presc)
    d, mort, clin, labs = temporal_tables(d, e)
    p = P.build(a.data_dir, loaded=(mae, ea, eb, presc, d.drop(columns=["fase", "losbin"])))
    p, ct, poly = pharmaco_tables(p)

    print("Figure 1 — n =", len(d), "| deaths by phase:",
          d.groupby("fase")["obito"].agg(["sum", "size"]).to_dict("index"))
    print(mort[["losbin", "sum", "size", "pct"]].to_string(index=False))
    print(clin[["PT", "all_OR", "all_p", "acute_OR", "acute_p", "prol_OR", "prol_p"]].round(3).to_string(index=False))
    print(labs.round(3).to_string(index=False))
    print("\nFigure 2 — n =", len(p), "| deaths:", int(p["obito"].sum()))
    print(ct[["PT", "all_alta", "all_obito", "OR_crude", "OR_adj", "OR_adj_lo", "OR_adj_hi", "p_adj"]]
          .round(3).to_string(index=False))
    print(poly.round(4).to_string(index=False))

    mort.assign(losbin=mort["losbin"].astype(str)).to_csv(
        os.path.join(a.out_dir, "desfecho_mortalidade_permanencia.csv"), index=False)
    clin.to_csv(os.path.join(a.out_dir, "desfecho_temporal_clinico.csv"), index=False)
    labs.to_csv(os.path.join(a.out_dir, "desfecho_exames_admissao.csv"), index=False)
    ct.to_csv(os.path.join(a.out_dir, "desfecho_farmaco_classes.csv"), index=False)
    poly.to_csv(os.path.join(a.out_dir, "desfecho_farmaco_polifarmacia.csv"), index=False)
    for lang in ["PT", "EN"]:
        figure1(d, mort, clin, labs, lang, os.path.join(a.out_dir, f"Figure_desfecho_temporal_{lang}"))
        figure2(p, ct, poly, lang, os.path.join(a.out_dir, f"Figure_desfecho_farmaco_{lang}"))
    print("Saved to", a.out_dir)


if __name__ == "__main__":
    main()
