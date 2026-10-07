"""
Figure — Pharmacotherapeutic profile of the long-COVID temporal groups.

Uses the temporal classification built in 12_long_covid_temporal_profile.py
(short ≤ 3 wk / post-acute 3–12 wk / chronic > 12 wk, in-hospital deaths apart;
mother sheet New_pacientes703.xlsx defines the n) and links it to
PRESC_ORIGINAL_MODIF_703pacientes.xlsx.

The prescription sheet has no dates, so drugs cannot be placed on the time axis
themselves. Instead, each patient's prescriptions are mapped to therapeutic
classes (regex on NOME_MEDIC, topical/ophthalmic forms excluded from systemic
classes) and compared across the temporal groups and across bins of observed
illness time T (days since admission). Survivors with long COVID are compared
with short COVID by Fisher's exact test (crude OR) and by logistic regression
adjusted for age, critical COVID and log(1 + hospital days), since more days in
hospital mechanically mean more prescriptions.

Cohort: the 508 patients of the prescription sheet, all of whom are in the
mother sheet, one row per patient. Admission date, hospital days, outcome and
clinical variables come from the mother sheet. The 3 of them with two
admissions (re-admitted after 2–3 days) are collapsed to one row: first
admission date, summed hospital days, lab follow-up counted from the first
admission, worst outcome/condition across both admissions.

Outputs (in --out-dir): Figure_long_covid_farmaco_PT/EN (.png, .tiff),
long_covid_farmaco_classes.csv, long_covid_farmaco_polifarmacia.csv.

Usage: python 13_long_covid_pharmacotherapy.py [--data-dir DIR] [--out-dir DIR]
"""
import argparse
import sys
import importlib.util
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.stats import fisher_exact, kruskal, mannwhitneyu
import statsmodels.api as sm

sys.dont_write_bytecode = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "temporal", os.path.join(BASE_DIR, "12_long_covid_temporal_profile.py"))
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

DPI = 600
MIN_USERS = 10                     # min. survivors exposed to model a class
T_BINS = [0, 7, 21, 42, 84, 180, np.inf]
TEXT, SUBTEXT, GRID = T.TEXT, T.SUBTEXT, T.GRID
COLORS = T.COLORS
GROUPS = T.GROUPS
HEAT = LinearSegmentedColormap.from_list("heat", ["#F4F7FB", "#9CC0EA", "#2a78d6", "#123F78"])

TOPICAL = r"COLIRIO|POMADA|CREME|\bOFT|\bGEL\b|TOPIC|NASAL|LOCAO|NASONEX|DERSANI|PAPAINA"
NON_DRUG = r"FISIOLOGIC|FISIOLÃ|SORO |CONTRASTE|COLD CREAM|VASELINA|AGUA |CLOREXIDINA|GLICERINADA|ENEMA"

# (key, domain, PT label, EN label, regex, systemic-only)
CLASSES = [
    ("cort", "covid", "Corticoide sistêmico", "Systemic corticosteroid",
     r"DEXAMETASONA|METILPREDNISOLONA|HIDROCORTISONA|PREDNISONA|PREDNISOLONA", True),
    ("anticoag", "covid", "Anticoagulante", "Anticoagulant",
     r"HEPARINA|ENOXAPARINA|VARFARINA|RIVAROXABANA|FONDAPARINUX|APIXABANA|DABIGATRANA", True),
    ("azt", "covid", "Azitromicina", "Azithromycin", r"AZITROMICINA", True),
    ("ivt_clq", "covid", "Ivermectina/cloroquina/nitazoxanida", "Ivermectin/chloroquine/nitazoxanide",
     r"IVERMECTINA|CLOROQUINA|HIDROXICLOROQUINA|NITAZOXANIDA", True),
    ("oselt", "covid", "Oseltamivir", "Oseltamivir", r"OSELTAMIVIR", True),
    ("atb_broad", "infec", "Antibiótico de amplo espectro", "Broad-spectrum antibiotic",
     r"MEROPENEM|IMIPENEM|VANCOMICINA|POLIMIXINA|COLISTIMETATO|PIPERACILINA|CEFEPIMA|CEFTAZIDIMA|"
     r"TIGECICLINA|AMICACINA|LINEZOLIDA|TEICOPLANINA", True),
    ("atb_other", "infec", "Outro antibiótico", "Other antibiotic",
     r"CEFTRIAXONA|AMOXICILINA|AMPICILINA|CEFALEXINA|CEFALOTINA|CEFAZOLINA|CEFUROXIMA|CLARITROMICINA|"
     r"LEVOFLOXACINO|CIPROFLOXACINO|CLINDAMICINA|OXACILINA|GENTAMICINA|METRONIDAZOL|SULFAMETOXAZOL|"
     r"DOXICICLINA|FOSFOMICINA|NITROFURANTOINA", True),
    ("antifung", "infec", "Antifúngico sistêmico", "Systemic antifungal",
     r"FLUCONAZOL|MICAFUNGINA|ANFOTERICINA|VORICONAZOL|CASPOFUNGINA", True),
    ("sedat", "icu", "Sedativo/anestésico", "Sedative/anaesthetic",
     r"MIDAZOLAM|PROPOFOL|DEXMEDETOMIDINA|ESCETAMINA|CETAMINA|ETOMIDATO", True),
    ("nmb", "icu", "Bloqueador neuromuscular", "Neuromuscular blocker",
     r"CISATRACURIO|ATRACURIO|ROCURONIO|PANCURONIO|VECURONIO|SUXAMETONIO", True),
    ("vaso", "icu", "Vasopressor/inotrópico", "Vasopressor/inotrope",
     r"NOREPINEFRINA|VASOPRESSINA|ADRENALINA|EPINEFRINA|DOBUTAMINA|DOPAMINA|MILRINONA", True),
    ("opi_strong", "icu", "Opioide forte (EV)", "Strong opioid (IV)",
     r"FENTANILA|MORFINA|REMIFENTANILA|SUFENTANILA|NALBUFINA", True),
    ("metadona", "neuro", "Metadona (desmame de opioide)", "Methadone (opioid weaning)", r"METADONA", True),
    ("opi_weak", "neuro", "Opioide fraco", "Weak opioid", r"TRAMADOL|CODE", True),
    ("antidep", "neuro", "Antidepressivo", "Antidepressant",
     r"SERTRALINA|AMITRIPTILINA|FLUOXETINA|PAROXETINA|ESCITALOPRAM|CITALOPRAM|TRAZODONA|NORTRIPTILINA|"
     r"VENLAFAXINA|DULOXETINA|MIRTAZAPINA", True),
    ("antipsy", "neuro", "Antipsicótico", "Antipsychotic",
     r"QUETIAPINA|RISPERIDONA|HALOPERIDOL|OLANZAPINA|CLORPROMAZINA|LEVOMEPROMAZINA", True),
    ("benzo", "neuro", "Benzodiazepínico/hipnótico oral", "Oral benzodiazepine/hypnotic",
     r"LORAZEPAM|CLONAZEPAM|DIAZEPAM|CLOBAZAM|ALPRAZOLAM|ZOLPIDEM", True),
    ("anticonv", "neuro", "Anticonvulsivante/gabapentinoide", "Anticonvulsant/gabapentinoid",
     r"GABAPENTINA|PREGABALINA|CARBAMAZEPINA|VALPROICO|FENITOINA|FENOBARBITAL|LEVETIRACETAM|TOPIRAMATO", True),
    ("musc", "neuro", "Relaxante muscular", "Muscle relaxant", r"BACLOFENO|CICLOBENZAPRINA", True),
    ("inhal", "resp", "Broncodilatador/corticoide inalatório", "Bronchodilator/inhaled steroid",
     r"SALBUTAMOL|IPRATROPIO|TERBUTALINA|BECLOMETASONA|BUDESONIDA|FORMOTEROL|FENOTEROL", False),
    ("insulin", "cardio", "Insulina", "Insulin", r"INSULINA", True),
    ("antiht", "cardio", "Anti-hipertensivo", "Antihypertensive",
     r"ANLODIPINO|ENALAPRIL|CAPTOPRIL|LOSARTANA|ATENOLOL|CARVEDILOL|METOPROLOL|PROPRANOLOL|BISOPROLOL|"
     r"HIDRALAZINA|CLONIDINA|DOXAZOSINA|HIDROCLOROTIAZIDA|ESPIRONOLACTONA|NIFEDIPINO", True),
    ("furos", "cardio", "Diurético de alça", "Loop diuretic", r"FUROSEMIDA|BUMETANIDA", True),
    ("antiarr", "cardio", "Antiarrítmico/digitálico", "Antiarrhythmic/digitalis", r"AMIODARONA|DESLANOSIDEO|DIGOXINA", True),
    ("statin", "cardio", "Estatina", "Statin", r"ATORVASTATINA|SINVASTATINA|ROSUVASTATINA", True),
    ("antiplat", "cardio", "Antiagregante plaquetário", "Antiplatelet", r"ACETILSALICILICO|CLOPIDOGREL|TICAGRELOR", True),
    ("hemat", "other", "Eritropoetina/ferro/folato/B12", "Erythropoietin/iron/folate/B12",
     r"ALFAEPOETINA|SULFATO FERROSO|SACARATO|ACIDO FOLICO|VITAMINA B12|CIANOCOBALAMINA", True),
]
DOMAINS = {
    "covid": ("Dirigidos à COVID-19", "COVID-19-directed"),
    "infec": ("Anti-infecciosos", "Anti-infectives"),
    "icu": ("Suporte intensivo", "Intensive support"),
    "neuro": ("Neuropsiquiátricos e dor", "Neuropsychiatric and pain"),
    "resp": ("Respiratórios", "Respiratory"),
    "cardio": ("Cardiometabólicos", "Cardiometabolic"),
    "other": ("Outros", "Other"),
}
# classes followed across illness time in panel D
TRACK = ["cort", "anticoag", "atb_broad", "metadona", "antipsy", "anticonv", "antidep", "inhal"]

TXT = {
    "PT": dict(
        title="Perfil farmacoterapêutico segundo a classificação temporal da COVID longa",
        tA="A  Prevalência das classes por grupo (% dos pacientes com prescrição)",
        tB="B  COVID longa × curta (sobreviventes)",
        tC="C  Polifarmácia",
        tD="D  Prevalência pelo tempo de doença (sobreviventes)",
        orlab="OR (IC 95%), escala log", orhdr="OR aj. (IC 95%)", phdr="p aj.",
        adj="Ajustado por idade, COVID crítica e log(1 + dias de internação)",
        crude="bruto", adjusted="ajustado", cmp="Curta × COVID longa",
        yC="Princípios ativos distintos por paciente", xD="Dias desde a admissão",
        weeks3="3 sem", weeks12="12 sem", nD="n",
        foot=("Prescrições sem data: as classes não podem ser posicionadas no tempo; o painel D mostra a prevalência "
              "em pacientes agrupados pelo tempo de doença observado (internação ou último exame em até 365 dias).\n"
              "Coorte: os {n} pacientes da planilha de prescrição, com admissão e internação da planilha mãe (Curta {short}, Pós-aguda {post}, "
              "Crônica {chronic}, Óbito {death}). Classes por expressão regular sobre NOME_MEDIC; formas tópicas/"
              "oftálmicas excluídas das classes sistêmicas. Associação não implica efeito causal do medicamento."),
    ),
    "EN": dict(
        title="Pharmacotherapeutic profile by long-COVID temporal classification",
        tA="A  Prevalence of drug classes by group (% of patients with prescriptions)",
        tB="B  Long vs. short COVID (survivors)",
        tC="C  Polypharmacy",
        tD="D  Prevalence by illness time (survivors)",
        orlab="OR (95% CI), log scale", orhdr="Adj. OR (95% CI)", phdr="adj. p",
        adj="Adjusted for age, critical COVID and log(1 + hospital days)",
        crude="crude", adjusted="adjusted", cmp="Short vs. long COVID",
        yC="Distinct active ingredients per patient", xD="Days since admission",
        weeks3="3 wk", weeks12="12 wk", nD="n",
        foot=("Prescriptions are undated: drug classes cannot be placed on the time axis; panel D shows prevalence "
              "in patients grouped by observed illness time (hospital stay or last lab record within 365 days).\n"
              "Cohort: the {n} patients of the prescription sheet, with admission and stay from the mother sheet (Short {short}, Post-acute {post}, "
              "Chronic {chronic}, Death {death}). Classes by regular expression on NOME_MEDIC; topical/ophthalmic "
              "forms excluded from systemic classes. Association does not imply a causal drug effect."),
    ),
}
STEM2 = {"ACIDO", "SOLUCAO", "CLORETO", "SULFATO", "VITAMINA", "INSULINA", "BICARBONATO", "CARBONATO",
         "GLICONATO", "HIDROXIDO", "NITRATO", "CITRATO", "FOSFATO", "OLEO", "SAIS", "NUTRICAO", "SOL."}


# ----------------------------------------------------------------------------- data
def drug_classes(presc):
    rx = presc[["COD_PACIENTE", "NOME_MEDIC"]].dropna().copy()
    name = rx["NOME_MEDIC"].astype(str).str.upper()
    topical = name.str.contains(TOPICAL, regex=True)
    rx["drug"] = ~name.str.contains(NON_DRUG, regex=True)
    for key, _, _, _, rgx, systemic in CLASSES:
        hit = name.str.contains(rgx, regex=True)
        rx[key] = hit & ~topical if systemic else hit
    words = name.str.replace(r"[^A-Z0-9. ]", " ", regex=True).str.split()
    rx["ingred"] = [(" ".join(w[:2]) if w and w[0] in STEM2 else (w[0] if w else "")) for w in words]
    keys = [c[0] for c in CLASSES]
    per_pt = rx.groupby("COD_PACIENTE")[keys].any().astype(float)
    per_pt["n_ingred"] = rx[rx["drug"]].groupby("COD_PACIENTE")["ingred"].nunique()
    per_pt["n_items"] = rx.groupby("COD_PACIENTE").size()
    return per_pt


FLAGS = ["obito", "crit", "has_lab", "post_discharge"]


def per_patient(d):
    """One row per patient (mother-sheet data). Patients with two admissions keep the first
    admission date, the summed hospital days and lab follow-up counted from that first date."""
    d = d.copy()
    d["lab_end"] = d["entrada"] + pd.to_timedelta(d["last_lab"], unit="D")
    d["n_adm"] = d.groupby("Registro")["ep"].transform("size")
    agg = {c: "first" for c in d.columns if c not in ("Registro",)}
    agg.update({"entrada": "min", "los": "sum", "lab_end": "max", **{f: "max" for f in FLAGS}})
    p = d.sort_values("entrada").groupby("Registro", as_index=False).agg(agg)
    p["last_lab"] = (p["lab_end"] - p["entrada"]).dt.total_seconds() / 86400
    p["T"] = np.where(p["obito"] == 1, p["los"], np.fmax(p["los"], p["last_lab"].fillna(0)))
    cls = lambda t, o: "death" if o else ("short" if t <= T.SHORT_MAX else ("post" if t <= T.POST_MAX else "chronic"))
    p["grupo"] = [cls(t, o) for t, o in zip(p["T"], p["obito"])]
    p["grupo_los"] = [cls(t, o) for t, o in zip(p["los"], p["obito"])]
    p["long_covid"] = p["grupo"].isin(["post", "chronic"]).astype(int)
    return p


def build(data_dir):
    mae, ea, eb, presc = T.load(data_dir)
    d, _, _ = T.build(mae, ea, eb, presc)
    pp = drug_classes(presc)
    d = d.drop(columns=[c for c in pp.columns if c in d.columns])     # replace the coarse flags of script 12
    d = per_patient(d[d["Registro"].isin(pp.index)])
    d = d.merge(pp, left_on="Registro", right_index=True, how="left")
    d["items_day"] = d["n_items"] / d["los"].clip(lower=1)
    d["log_los"] = np.log1p(d["los"])
    d["tbin"] = pd.cut(d["T"], T_BINS, right=True, include_lowest=True)
    return d


# ----------------------------------------------------------------------------- stats
def class_table(d):
    surv = d[d["obito"] == 0]
    rows = []
    for key, dom, pt, en, _, _ in CLASSES:
        r = {"key": key, "domain": dom, "PT": pt, "EN": en}
        for g in GROUPS:
            s = d.loc[d["grupo"] == g, key]
            r[g] = 100 * s.mean()
            r[f"{g}_n"] = int(s.sum())
        a, b = surv.loc[surv["long_covid"] == 0, key], surv.loc[surv["long_covid"] == 1, key]
        r["users_surv"] = int(surv[key].sum())
        tab = [[b.sum(), len(b) - b.sum()], [a.sum(), len(a) - a.sum()]]
        r["OR_crude"], r["p_crude"] = fisher_exact(tab)
        r["OR_crude_lo"], r["OR_crude_hi"] = _woolf(tab)
        r.update(dict(OR_adj=np.nan, OR_adj_lo=np.nan, OR_adj_hi=np.nan, p_adj=np.nan))
        if r["users_surv"] >= MIN_USERS and r["users_surv"] <= len(surv) - MIN_USERS:
            X = sm.add_constant(surv[[key, "Idade", "crit", "log_los"]].astype(float))
            try:
                fit = sm.Logit(surv["long_covid"].astype(float), X).fit(disp=0, maxiter=200)
                ci = fit.conf_int().loc[key]
                r.update(OR_adj=np.exp(fit.params[key]), OR_adj_lo=np.exp(ci[0]),
                         OR_adj_hi=np.exp(ci[1]), p_adj=fit.pvalues[key])
            except Exception:                       # perfect separation etc.
                pass
        rows.append(r)
    return pd.DataFrame(rows)


def _woolf(tab):
    (a, b), (c, dd) = tab
    a, b, c, dd = [v + 0.5 for v in (a, b, c, dd)] if 0 in (a, b, c, dd) else (a, b, c, dd)
    lor, se = np.log(a * dd / (b * c)), np.sqrt(1 / a + 1 / b + 1 / c + 1 / dd)
    return np.exp(lor - 1.96 * se), np.exp(lor + 1.96 * se)


def poly_table(d):
    rows = []
    for g in GROUPS:
        s = d[d["grupo"] == g]
        rows.append({"grupo": g, "n": len(s),
                     "ingred_med": s["n_ingred"].median(), "ingred_q1": s["n_ingred"].quantile(.25),
                     "ingred_q3": s["n_ingred"].quantile(.75),
                     "poly10_pct": 100 * (s["n_ingred"] >= 10).mean(),
                     "poly20_pct": 100 * (s["n_ingred"] >= 20).mean(),
                     "items_day_med": s["items_day"].median()})
    t = pd.DataFrame(rows)
    surv = d[d["obito"] == 0]
    t.attrs["p_short_long"] = mannwhitneyu(surv.loc[surv["long_covid"] == 0, "n_ingred"],
                                           surv.loc[surv["long_covid"] == 1, "n_ingred"]).pvalue
    t.attrs["p_kw"] = kruskal(*[d.loc[d["grupo"] == g, "n_ingred"] for g in GROUPS]).pvalue
    return t


def fmt(v, dec, nd=2):
    return f"{v:.{nd}f}".replace(".", dec)


def fmt_p(p, dec):
    return T.fmt_p(p, dec) if np.isfinite(p) else "–"


# ----------------------------------------------------------------------------- figure
def figure(d, ct, poly, lang, out_base):
    L, Lt = TXT[lang], T.LANG[lang]
    dec = Lt["dec"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                         "axes.labelcolor": TEXT, "xtick.color": SUBTEXT, "ytick.color": SUBTEXT})
    fig = plt.figure(figsize=(20, 16), facecolor="white")
    fig.suptitle(L["title"], x=0.02, y=0.985, ha="left", fontsize=14, fontweight="bold", color=TEXT)

    # row layout with domain gaps
    ypos, dom_rows, y = [], [], 0
    for dom in DOMAINS:
        idx = ct.index[ct["domain"] == dom].tolist()
        dom_rows.append((dom, y))
        y += 1.1
        for i in idx:
            ypos.append((i, y)); y += 1
        y += 0.35
    ymax = y
    yof = dict(ypos)

    # ---------------- A: heatmap
    axA = fig.add_axes([0.175, 0.10, 0.225, 0.80])
    for i, yy in ypos:
        for j, g in enumerate(GROUPS):
            v = ct.loc[i, g]
            axA.add_patch(plt.Rectangle((j + 0.04, yy - 0.46), 0.92, 0.92, color=HEAT(v / 100), lw=0))
            axA.text(j + 0.5, yy, f"{v:.0f}", ha="center", va="center", fontsize=8,
                     color="white" if v >= 55 else TEXT)
        axA.text(-0.08, yy, ct.loc[i, lang], ha="right", va="center", fontsize=8.5, color=TEXT)
    for dom, yy in dom_rows:
        axA.text(-0.08, yy + 0.2, DOMAINS[dom][0 if lang == "PT" else 1].upper(), ha="right", va="center",
                 fontsize=8.5, fontweight="bold", color=SUBTEXT)
    for j, g in enumerate(GROUPS):
        n = int((d["grupo"] == g).sum())
        axA.text(j + 0.5, -0.9, f"{Lt['short'][g]}\n(n={n})", ha="center", va="bottom", fontsize=8.5,
                 fontweight="bold", color=TEXT)
        axA.add_patch(plt.Rectangle((j + 0.04, -0.75), 0.92, 0.22, color=COLORS[g], lw=0, clip_on=False))
    axA.plot([3, 3], [-0.2, ymax], color="white", lw=4)
    axA.set_xlim(0, 4); axA.set_ylim(ymax, -0.9)
    axA.axis("off")
    fig.text(0.02, 0.935, L["tA"], fontsize=11, fontweight="bold", color=TEXT)

    # ---------------- B: forest (shares rows with A)
    axB = fig.add_axes([0.42, 0.10, 0.12, 0.80])
    tr = axB.get_yaxis_transform()
    for i, yy in ypos:
        r = ct.loc[i]
        axB.hlines(yy, 0.05, 50, color=GRID, lw=0.4, zorder=0)
        axB.plot([r["OR_crude_lo"], r["OR_crude_hi"]], [yy - 0.18] * 2, color=SUBTEXT, lw=1, alpha=0.6)
        axB.scatter(r["OR_crude"], yy - 0.18, s=16, color="white", edgecolor=SUBTEXT, lw=0.9, zorder=3)
        if np.isfinite(r["OR_adj"]):
            sig = r["p_adj"] < 0.05
            axB.plot([r["OR_adj_lo"], r["OR_adj_hi"]], [yy + 0.18] * 2, color=TEXT, lw=1.4)
            axB.scatter(r["OR_adj"], yy + 0.18, s=30, marker="s", color=COLORS["chronic"] if sig else "white",
                        edgecolor=COLORS["chronic"], lw=1.1, zorder=3)
            txt = f"{fmt(r['OR_adj'], dec)} ({fmt(r['OR_adj_lo'], dec)}–{fmt(r['OR_adj_hi'], dec)})"
            axB.text(1.04, yy, txt, va="center", fontsize=7.5, color=TEXT if sig else SUBTEXT,
                     fontweight="bold" if sig else None, clip_on=False, transform=tr)
            axB.text(1.64, yy, fmt_p(r["p_adj"], dec), va="center", fontsize=7.5,
                     color=TEXT if sig else SUBTEXT, fontweight="bold" if sig else None, clip_on=False, transform=tr)
    axB.axvline(1, color=TEXT, lw=0.8, ls=(0, (3, 2)))
    axB.set_xscale("log"); axB.set_xlim(0.05, 50); axB.set_ylim(ymax, -0.9)
    axB.set_xticks([0.1, 0.3, 1, 3, 10, 30]); axB.set_xticklabels(["0.1", "0.3", "1", "3", "10", "30"])
    axB.set_yticks([]); axB.set_xlabel(L["orlab"])
    for s in ["top", "right", "left"]:
        axB.spines[s].set_visible(False)
    axB.text(1.04, -0.3, L["orhdr"], fontsize=8, fontweight="bold", color=TEXT, clip_on=False, transform=tr)
    axB.text(1.64, -0.3, L["phdr"], fontsize=8, fontweight="bold", color=TEXT, clip_on=False, transform=tr)
    fig.text(0.42, 0.935, L["tB"], fontsize=11, fontweight="bold", color=TEXT)
    axB.legend(handles=[Line2D([], [], color=SUBTEXT, marker="o", mfc="white", ls="-", label=L["crude"]),
                        Line2D([], [], color=COLORS["chronic"], marker="s", ls="-", label=L["adjusted"])],
               loc="upper left", bbox_to_anchor=(-0.02, -0.045), ncol=2, frameon=False, fontsize=8)
    fig.text(0.42, 0.045, L["adj"], fontsize=7.5, color=SUBTEXT, va="top")

    # ---------------- C: polypharmacy
    axC = fig.add_axes([0.73, 0.66, 0.25, 0.25])
    rng = np.random.default_rng(42)
    for j, g in enumerate(GROUPS):
        v = d.loc[d["grupo"] == g, "n_ingred"].dropna().values
        axC.scatter(j + rng.uniform(-0.25, 0.25, len(v)), v, s=6, color=COLORS[g], alpha=0.35, lw=0)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        axC.vlines(j, q1, q3, color=TEXT, lw=3)
        axC.hlines(med, j - 0.3, j + 0.3, color=TEXT, lw=2)
        axC.text(j + 0.33, med, f"{med:.0f}", va="center", fontsize=8.5, fontweight="bold", color=TEXT)
    axC.set_xticks(range(4)); axC.set_xticklabels([Lt["short"][g] for g in GROUPS], color=TEXT)
    axC.set_ylabel(L["yC"]); axC.grid(axis="y", color=GRID, lw=0.5); axC.set_axisbelow(True)
    for s in ["top", "right"]:
        axC.spines[s].set_visible(False)
    ps = fmt_p(poly.attrs["p_short_long"], dec)
    axC.text(0.02, 0.98, f"{L['cmp']}: p {ps}",
             transform=axC.transAxes, fontsize=8, color=SUBTEXT, va="top")
    axC.set_title(L["tC"], loc="left", fontsize=11, fontweight="bold", color=TEXT)

    # ---------------- D: small multiples across illness time (survivors)
    surv = d[d["obito"] == 0]
    bins = surv["tbin"].cat.categories
    mids = [3.5, 14, 31.5, 63, 132, 272]
    bin_col = [COLORS["short"]] * 2 + [COLORS["post"]] * 2 + [COLORS["chronic"]] * 2
    nbin = surv.groupby("tbin", observed=False).size().values
    x0, y0, w, h, gx, gy = 0.715, 0.10, 0.11, 0.085, 0.05, 0.045
    fig.text(x0 - 0.035, 0.625, L["tD"], fontsize=11, fontweight="bold", color=TEXT)
    wl = ["≤ 3 sem", "3–12 sem", "> 12 sem"] if lang == "PT" else ["≤ 3 wk", "3–12 wk", "> 12 wk"]
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COLORS[g]) for g in ["short", "post", "chronic"]],
               labels=wl, loc="upper left", bbox_to_anchor=(x0 - 0.04, 0.62), ncol=3, frameon=False, fontsize=8)
    xs = np.arange(len(bins))
    for k, key in enumerate(TRACK):
        r, c = divmod(k, 2)
        ax = fig.add_axes([x0 + c * (w + gx), y0 + (3 - r) * (h + gy), w, h])
        prev = 100 * surv.groupby("tbin", observed=False)[key].mean().values
        ax.bar(xs, prev, width=0.78, color=bin_col, edgecolor="white", lw=1)
        for xv in [1.5, 3.5]:
            ax.axvline(xv, color=TEXT, ls=(0, (3, 2)), lw=0.8)
        ax.set_ylim(0, 100); ax.set_yticks([0, 50, 100]); ax.set_xlim(-0.6, len(bins) - 0.4)
        ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
        name = ct.set_index("key").loc[key, lang]
        ax.set_title(name, loc="left", fontsize=8, color=TEXT, fontweight="bold", pad=3)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        if r == 3:
            ax.set_xticks(xs)
            ax.set_xticklabels(["0–7", "8–21", "22–42", "43–84", "85–180", ">180"], fontsize=6.5, rotation=40)
            ax.set_xlabel(L["xD"], fontsize=8)
        else:
            ax.set_xticks(xs); ax.set_xticklabels([])
        if c == 0:
            ax.set_ylabel("%", fontsize=8)
    fig.text(x0 - 0.035, y0 - 0.06, f"{L['nD']} = " + " · ".join(str(n) for n in nbin) + "  (" +
             ("por faixa" if lang == "PT" else "per bin") + ")", fontsize=7.5, color=SUBTEXT)

    ng = {g: int((d["grupo"] == g).sum()) for g in GROUPS}
    fig.text(0.02, 0.008, L["foot"].format(n=len(d), **ng), fontsize=8, color=SUBTEXT, va="bottom")
    fig.savefig(out_base + ".png", dpi=200, facecolor="white")
    fig.savefig(out_base + ".tiff", dpi=DPI, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=BASE_DIR)
    ap.add_argument("--out-dir", default=BASE_DIR)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    d = build(a.data_dir)
    ct = class_table(d)
    poly = poly_table(d)
    print(f"Patients: {len(d)} (unique: {d['Registro'].nunique()}; collapsed re-admissions: "
          f"{int((d['n_adm'] > 1).sum())})")
    print("Groups (stay + lab follow-up):", d["grupo"].value_counts().to_dict())
    print("Groups (mother-sheet stay only):", d["grupo_los"].value_counts().to_dict())
    cols = ["PT", "short", "post", "chronic", "death", "users_surv", "OR_crude", "p_crude",
            "OR_adj", "OR_adj_lo", "OR_adj_hi", "p_adj"]
    print(ct[cols].round(3).to_string(index=False))
    print(poly.round(1).to_string(index=False), poly.attrs)
    ct.to_csv(os.path.join(a.out_dir, "long_covid_farmaco_classes.csv"), index=False)
    poly.to_csv(os.path.join(a.out_dir, "long_covid_farmaco_polifarmacia.csv"), index=False)
    for lang in ["PT", "EN"]:
        figure(d, ct, poly, lang, os.path.join(a.out_dir, f"Figure_long_covid_farmaco_{lang}"))
    print("Saved to", a.out_dir)


if __name__ == "__main__":
    main()
