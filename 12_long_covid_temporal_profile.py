"""
Figure — Temporal profile of long COVID in the hospital cohort (n = 703 admissions).

Builds on the temporal definition used in the reference scheme:
    Short COVID        : illness ≤ 3 weeks
    Post-acute COVID   : > 3 weeks and ≤ 12 weeks   ┐
    Chronic COVID      : > 12 weeks                 ┘ long COVID

Mother sheet (defines the n): New_pacientes703.xlsx — one row per admission
(703 rows, 699 unique `Registro`; 4 patients were re-admitted).
Linked sheets (joined by `COD_PACIENTE` = `Registro`):
    ExamesB_filtradaa.xlsx                         — lab records with timestamp
    ExamesB_coorte703_36variaveis_corrigido.xlsx   — 36 lab variables + reference ranges
    PRESC_ORIGINAL_MODIF_703pacientes.xlsx         — prescriptions (no dates)

Time origin = admission date (`Data de Entrada`), used as a proxy for symptom
onset (the cohort has no symptom-onset date). For each admission:
    hospital time   = `Dias_permanência`
    follow-up time  = last lab record released within FOLLOWUP_MAX days after
                      admission (both lab sheets pooled)
    observed time T = max(hospital time, follow-up time) for survivors;
                      in-hospital deaths are kept as a separate group (T = LOS).
Survivors are then classified as Short (T ≤ 21 d), Post-acute (21 < T ≤ 84 d)
or Chronic (T > 84 d). Lab activity after discharge is a marker of continued
health-care contact, not proof of persistent symptoms; the figure and the
printed summary also show the LOS-only classification (mother sheet only).

Outputs (in --out-dir): Figure_long_covid_PT.png/.tiff, Figure_long_covid_EN.png/.tiff,
long_covid_classificacao.csv (per admission, no identifiers besides row id),
long_covid_perfil.csv, long_covid_marcadores.csv.

Usage: python 12_long_covid_temporal_profile.py [--data-dir DIR] [--out-dir DIR]
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import fisher_exact, mannwhitneyu

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
F_MAE = "New_pacientes703.xlsx"
F_EXA = "ExamesB_filtradaa.xlsx"
F_EXB = "ExamesB_coorte703_36variaveis_corrigido.xlsx"
F_PRESC = "PRESC_ORIGINAL_MODIF_703pacientes.xlsx"

SHORT_MAX, POST_MAX = 21, 84          # 3 and 12 weeks
FOLLOWUP_MAX = 365                    # lab records beyond 1 year are not counted
MIN_TESTED = 8                        # min. patients tested to show a marker
DPI = 600

TEXT, SUBTEXT, GRID, PANEL = "#1F2328", "#57606A", "#D0D7DE", "#F6F8FA"
COLORS = {"short": "#2a78d6", "post": "#eb6834", "chronic": "#4a3aa7", "death": "#8C8C88"}
GROUPS = ["short", "post", "chronic", "death"]

LANG = {
    "PT": dict(
        groups={"short": "COVID curta (≤ 3 sem)", "post": "COVID pós-aguda (3–12 sem)",
                "chronic": "COVID crônica (> 12 sem)", "death": "Óbito hospitalar"},
        short={"short": "Curta", "post": "Pós-aguda", "chronic": "Crônica", "death": "Óbito"},
        long_covid="COVID LONGA", weeks3="3 semanas", weeks12="12 semanas",
        xlab="Dias desde a admissão (escala raiz quadrada)",
        admission="Admissão\n(proxy do início\ndos sintomas)",
        hosp="Internação", follow="Seguimento laboratorial pós-alta", death_mark="Óbito",
        tA="A  Trajetória temporal de cada internação (n = {n})",
        tB="B  Classificação temporal",
        tC="C  Perfil dos sobreviventes",
        tD="D  Possíveis efeitos: exames alterados > 3 sem",
        schemeL="Somente\ninternação", schemeF="Internação +\nseguimento",
        xB="Internações (n)", xC="Prevalência no grupo (%)",
        xD="Pacientes com ≥ 1 resultado alterado (% dos testados)",
        winP="Janela 3–12 sem", winC="Janela > 12 sem",
        pnote="p: curta × longa (Fisher / Mann-Whitney)",
        dnote="Sobreviventes com COVID longa. n = testados na janela (3–12 sem | > 12 sem)",
        foot=("Classificação adaptada do esquema temporal de COVID curta/pós-aguda/crônica. "
              "Tempo zero = admissão. Sobreviventes classificados pelo maior entre tempo de internação e "
              "último exame em até 365 dias; óbitos mostrados à parte.\nSeguimento laboratorial indica "
              "contato continuado com o serviço, não comprova sintomas persistentes. Prescrições sem data "
              "(perfil apenas). Sexo conforme codificação da planilha mãe.\nSobreviventes sem exames "
              "(n = {nolab}) só podem ser classificados pela internação, o que infla o grupo COVID curta."),
        vars={"age61": "Idade ≥ 61 anos", "age": "Idade, mediana (anos)", "sex1": "Sexo (código 1)",
              "vac": "Vacinado", "crit": "COVID crítica", "srag": "SRAG", "card": "Prob. cardíaco",
              "diab": "Diabetes", "resp": "Prob. respiratório", "renal": "Prob. renal", "lra": "LRA",
              "neuro": "Prob. neurológico", "cancer": "Câncer", "choque": "Choque",
              "readm": "Reinternação", "cort": "Corticoide*", "antib": "Antibiótico*",
              "anticoag": "Anticoagulante*", "azt": "Azitromicina*"},
        rx_note="* entre pacientes com prescrição registrada",
        dec=",",
    ),
    "EN": dict(
        groups={"short": "Short COVID (≤ 3 wk)", "post": "Post-acute COVID (3–12 wk)",
                "chronic": "Chronic COVID (> 12 wk)", "death": "In-hospital death"},
        short={"short": "Short", "post": "Post-acute", "chronic": "Chronic", "death": "Death"},
        long_covid="LONG COVID", weeks3="3 weeks", weeks12="12 weeks",
        xlab="Days since admission (square-root scale)",
        admission="Admission\n(proxy for\nsymptom onset)",
        hosp="Hospital stay", follow="Post-discharge lab follow-up", death_mark="Death",
        tA="A  Temporal trajectory of each admission (n = {n})",
        tB="B  Temporal classification",
        tC="C  Profile of survivors",
        tD="D  Possible effects: abnormal labs > 3 wk",
        schemeL="Hospital\nstay only", schemeF="Stay +\nfollow-up",
        xB="Admissions (n)", xC="Prevalence in group (%)",
        xD="Patients with ≥ 1 out-of-range result (% of tested)",
        winP="3–12 wk window", winC="> 12 wk window",
        pnote="p: short vs. long (Fisher / Mann-Whitney)",
        dnote="Long-COVID survivors. n = tested in window (3–12 wk | > 12 wk)",
        foot=("Classification adapted from the short/post-acute/chronic COVID temporal scheme. "
              "Time zero = admission. Survivors classified by the longer of hospital stay and last "
              "lab record within 365 days; deaths shown separately.\nLab follow-up indicates continued "
              "contact with the service, not proof of persistent symptoms. Prescriptions are undated "
              "(profile only). Sex as coded in the mother sheet.\nSurvivors without lab records "
              "(n = {nolab}) can only be classified by hospital stay, which inflates the short-COVID group."),
        vars={"age61": "Age ≥ 61 years", "age": "Age, median (years)", "sex1": "Sex (code 1)",
              "vac": "Vaccinated", "crit": "Critical COVID", "srag": "SARS", "card": "Cardiac disease",
              "diab": "Diabetes", "resp": "Respiratory disease", "renal": "Renal disease", "lra": "AKI",
              "neuro": "Neurological disease", "cancer": "Cancer", "choque": "Shock",
              "readm": "Readmission", "cort": "Corticosteroid*", "antib": "Antibiotic*",
              "anticoag": "Anticoagulant*", "azt": "Azithromycin*"},
        rx_note="* among patients with recorded prescriptions",
        dec=".",
    ),
}
MARKER_EN = {"Proteína C reativa": "C-reactive protein", "D-dímero": "D-dimer", "Ferritina": "Ferritin",
             "Creatinina": "Creatinine", "Ureia": "Urea", "Potássio": "Potassium", "Sódio": "Sodium",
             "Cálcio ionizado": "Ionized calcium", "Cálcio total": "Total calcium", "Lactato": "Lactate",
             "Glicemia/jejum": "Fasting glucose", "Albumina": "Albumin", "Fibrinogênio": "Fibrinogen",
             "Bilirrubina": "Bilirubin", "Triglicérides": "Triglycerides", "Colesterol total": "Total cholesterol",
             "HDL colesterol": "HDL cholesterol", "LDL colesterol": "LDL cholesterol", "Troponina I": "Troponin I",
             "Ácido úrico": "Uric acid", "Hb1Ac": "HbA1c", "TP/INR": "PT/INR"}


# ----------------------------------------------------------------------------- data
def load(data_dir):
    mae = pd.read_excel(os.path.join(data_dir, F_MAE))
    mae = mae.reset_index().rename(columns={"index": "ep"})
    mae["entrada"] = pd.to_datetime(mae["Data de Entrada"])
    mae["los"] = mae["Dias_permanência"].astype(float)
    mae["obito"] = mae["Óbito"].astype(int)
    mae["readm"] = mae["Registro"].duplicated(keep=False).astype(int)

    ea = pd.read_excel(os.path.join(data_dir, F_EXA))
    eb = pd.read_excel(os.path.join(data_dir, F_EXB))
    presc = pd.read_excel(os.path.join(data_dir, F_PRESC))
    return mae, ea, eb, presc


def assign_episode(df, mae):
    """Attach each lab record to the admission of that patient that started last before it
    (records before the first admission go to the first admission, with negative time)."""
    df = df[df["COD_PACIENTE"].isin(mae["Registro"])].copy()
    df["DATA_HORA_LIB"] = pd.to_datetime(df["DATA_HORA_LIB"])
    df = df.dropna(subset=["DATA_HORA_LIB"]).sort_values("DATA_HORA_LIB")
    eps = mae[["ep", "Registro", "entrada"]].sort_values("entrada")
    out = pd.merge_asof(df, eps, left_on="DATA_HORA_LIB", right_on="entrada",
                        left_by="COD_PACIENTE", right_by="Registro", direction="backward")
    miss = out["ep"].isna()
    if miss.any():
        first = eps.groupby("Registro").first()
        out.loc[miss, "ep"] = out.loc[miss, "COD_PACIENTE"].map(first["ep"])
        out.loc[miss, "entrada"] = out.loc[miss, "COD_PACIENTE"].map(first["entrada"])
    out["ep"] = out["ep"].astype(int)
    out["t"] = (out["DATA_HORA_LIB"] - out["entrada"]).dt.total_seconds() / 86400
    return out


def num(x):
    s = str(x).strip().replace(" ", "").lstrip("<>=")
    if re.fullmatch(r"\d+(,\d+)?", s):
        s = s.replace(",", ".")
    elif re.fullmatch(r"\d{1,3}(\.\d{3})+,\d+", s):
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def ref_range(ref):
    """Return (low, high) from strings like 'até 1 MG/DL', '120 a 246 U/L', '> 40 MG/DL',
    'M: 0,7 a 1,3 | F: 0,55 a 1,02' (sex-specific: the most lenient bound is used)."""
    if not isinstance(ref, str) or "dispon" in ref.lower():
        return None
    lows, highs = [], []
    for part in ref.split("|"):
        p = part.lower()
        nums = [float(v.replace(",", ".")) for v in re.findall(r"\d+(?:,\d+)?(?:\.\d+)?", p)]
        if not nums:
            continue
        if "até" in p or "ate " in p:
            highs.append(nums[-1])
        elif p.strip().startswith(">") or " > " in p:
            lows.append(nums[0])
        elif len(nums) >= 2:
            lows.append(nums[0]); highs.append(nums[1])
    if not lows and not highs:
        return None
    return (min(lows) if lows else -np.inf, max(highs) if highs else np.inf)


def build(mae, ea, eb, presc):
    # ---- pooled lab timestamps (both lab sheets)
    cols = ["COD_PACIENTE", "DATA_HORA_LIB"]
    labs = pd.concat([ea[cols], eb[cols]]).drop_duplicates()
    labs = assign_episode(labs, mae)
    win = labs[(labs["t"] >= 0) & (labs["t"] <= FOLLOWUP_MAX)]
    last_lab = win.groupby("ep")["t"].max()

    d = mae.copy()
    d["has_lab"] = d["Registro"].isin(labs["COD_PACIENTE"]).astype(int)
    d["last_lab"] = d["ep"].map(last_lab)
    d["T"] = np.where(d["obito"] == 1, d["los"], np.fmax(d["los"], d["last_lab"].fillna(0)))
    d["post_discharge"] = ((d["obito"] == 0) & (d["last_lab"] > d["los"] + 1)).astype(int)

    def classify(t, death):
        if death:
            return "death"
        return "short" if t <= SHORT_MAX else ("post" if t <= POST_MAX else "chronic")

    d["grupo"] = [classify(t, o) for t, o in zip(d["T"], d["obito"])]
    d["grupo_los"] = [classify(t, o) for t, o in zip(d["los"], d["obito"])]
    d["long_covid"] = d["grupo"].isin(["post", "chronic"]).astype(int)

    # ---- prescriptions (undated): any 'Sim' per class per patient
    rx = presc.copy()
    for c in ["AZT", "Corticoides", "Antib", "Heparina", "Enoxaparina", "Varfarina"]:
        rx[c] = rx[c].astype(str).str.strip().str.lower().eq("sim")
    rx["anticoag"] = rx[["Heparina", "Enoxaparina", "Varfarina"]].any(axis=1)
    rxp = rx.groupby("COD_PACIENTE")[["Corticoides", "Antib", "anticoag", "AZT"]].any().astype(float)
    d["has_rx"] = d["Registro"].isin(rxp.index).astype(int)
    for src, dst in [("Corticoides", "cort"), ("Antib", "antib"), ("anticoag", "anticoag"), ("AZT", "azt")]:
        d[dst] = d["Registro"].map(rxp[src])            # NaN when no prescription record

    d["age61"] = (d["Idade"] >= 61).astype(int)
    d["age"] = d["Idade"]
    for src, dst in [("Sexo", "sex1"), ("Vacinado", "vac"), ("COVID_CRÍTICA", "crit"), ("SRAG", "srag"),
                     ("Prob_Card", "card"), ("Diabetes", "diab"), ("Prob_Resp", "resp"),
                     ("Prob_Renal", "renal"), ("LRA", "lra"), ("Prob_neurol", "neuro"),
                     ("Cancer", "cancer"), ("Choques", "choque")]:
        d[dst] = d[src].astype(int)

    # ---- lab abnormalities (36-variable sheet, reference ranges)
    e = assign_episode(eb, mae)
    e = e.merge(d[["ep", "grupo", "los"]], on="ep")
    e["v"] = e["RESULTADO"].map(num)
    rng = {v: ref_range(r) for v, r in e.groupby("VARIAVEL")["VALOR_REFERENCIA"]
           .agg(lambda s: s.dropna().value_counts().index[0] if s.notna().any() else None).items()}
    e["lo"] = e["VARIAVEL"].map(lambda v: rng[v][0] if rng.get(v) else np.nan)
    e["hi"] = e["VARIAVEL"].map(lambda v: rng[v][1] if rng.get(v) else np.nan)
    e = e.dropna(subset=["v", "lo"])
    e["alt"] = ((e["v"] < e["lo"]) | (e["v"] > e["hi"])).astype(int)
    e["janela"] = pd.cut(e["t"], [-np.inf, -1e-9, SHORT_MAX, POST_MAX, FOLLOWUP_MAX, np.inf],
                         labels=["pre", "acute", "post", "chronic", "late"])
    return d, e, labs


# ----------------------------------------------------------------------------- stats
def profile_table(d):
    surv = d[d["obito"] == 0]
    keys = ["age61", "age", "sex1", "vac", "crit", "srag", "card", "diab", "resp", "renal", "lra",
            "neuro", "cancer", "choque", "cort", "antib", "anticoag", "azt"]
    rows = []
    for k in keys:
        r = {"var": k}
        for g in GROUPS:
            s = d.loc[d["grupo"] == g, k].dropna()
            r[f"{g}_n"] = len(s)
            r[g] = s.median() if k == "age" else 100 * s.mean() if len(s) else np.nan
        a = surv.loc[surv["long_covid"] == 0, k].dropna()
        b = surv.loc[surv["long_covid"] == 1, k].dropna()
        if k == "age":
            r["p"] = mannwhitneyu(a, b).pvalue
            r["OR"] = np.nan
        else:
            tab = [[b.sum(), len(b) - b.sum()], [a.sum(), len(a) - a.sum()]]
            r["OR"], r["p"] = fisher_exact(tab)
        r["long"] = b.median() if k == "age" else 100 * b.mean()
        rows.append(r)
    return pd.DataFrame(rows)


def marker_table(e, d):
    lc = e[e["grupo"].isin(["post", "chronic"])]
    rows = []
    for var, s in lc.groupby("VARIAVEL"):
        r = {"marcador": var}
        for w in ["acute", "post", "chronic"]:
            sw = s[s["janela"] == w].groupby("ep")["alt"].max()
            r[f"{w}_n"] = len(sw)
            r[f"{w}_pct"] = 100 * sw.mean() if len(sw) else np.nan
        rows.append(r)
    t = pd.DataFrame(rows)
    t["n_after3w"] = t["post_n"] + t["chronic_n"]
    return t.sort_values("post_pct", ascending=False)


def fmt_p(p, dec):
    s = "< 0.001" if p < 0.001 else f"{p:.3f}"
    return s.replace(".", dec)


# ----------------------------------------------------------------------------- figure
SQ = lambda x: np.sqrt(np.clip(x, 0, None))


def figure(d, prof, mk, lang, out_base):
    L = LANG[lang]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                         "axes.labelcolor": TEXT, "xtick.color": SUBTEXT, "ytick.color": SUBTEXT})
    fig = plt.figure(figsize=(16, 14.5), facecolor="white")
    gs_top = fig.add_gridspec(1, 1, left=0.06, right=0.80, top=0.955, bottom=0.57)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.75, 1.1, 1.15], wspace=0.55,
                          left=0.08, right=0.93, top=0.47, bottom=0.13)

    # ---------------- A: swimmer plot
    ax = fig.add_subplot(gs_top[0, 0])
    order = {"short": 0, "post": 1, "chronic": 2, "death": 3}
    dd = d.assign(o=d["grupo"].map(order)).sort_values(["o", "T"], ascending=[True, True]).reset_index(drop=True)
    gap = 25
    y, ys, spans = 0, [], {}
    for g in GROUPS:
        sub = dd[dd["grupo"] == g]
        spans[g] = (y, y + len(sub))
        ys += list(range(y, y + len(sub)))
        y += len(sub) + gap
    dd["y"] = ys
    ymax = y - gap
    for g in GROUPS:
        sub = dd[dd["grupo"] == g]
        c = COLORS[g]
        ax.hlines(-sub["y"], 0, SQ(sub["los"]), color=c, lw=0.9, alpha=0.95)
        fu = sub[sub["post_discharge"] == 1]
        ax.hlines(-fu["y"], SQ(fu["los"]), SQ(fu["last_lab"]), color=c, lw=0.9, alpha=0.35)
        if g == "death":
            ax.scatter(SQ(sub["los"]), -sub["y"], marker="x", s=4, lw=0.5, color=TEXT, zorder=3)
        y0, y1 = spans[g]
        ax.text(SQ(FOLLOWUP_MAX) * 1.01, -(y0 + y1) / 2, f"{L['groups'][g]}\nn = {y1 - y0}",
                va="center", ha="left", fontsize=9, color=TEXT, fontweight="bold")
    for xv, lab in [(SHORT_MAX, L["weeks3"]), (POST_MAX, L["weeks12"])]:
        ax.axvline(SQ(xv), color=TEXT, ls=(0, (4, 3)), lw=1.1)
        ax.text(SQ(xv), 12, lab, ha="center", va="bottom", fontsize=10, fontweight="bold", color=TEXT)
    ax.axvline(0, color=TEXT, lw=1.4)
    ax.text(-0.1, 12, L["admission"], fontsize=8, color=SUBTEXT, va="bottom", ha="right")
    # long-COVID bracket
    yb0, yb1 = spans["post"][0], spans["chronic"][1]
    xb = SQ(FOLLOWUP_MAX) * 1.21
    ax.plot([xb - 0.25, xb, xb, xb - 0.25], [-yb0, -yb0, -yb1, -yb1], color=TEXT, lw=1.3, clip_on=False)
    ax.text(xb + 0.25, -(yb0 + yb1) / 2, L["long_covid"], rotation=90, va="center", ha="left",
            fontsize=11, fontweight="bold", color=TEXT, clip_on=False)
    ticks = [0, 3, 7, 14, 21, 42, 84, 180, 365]
    ax.set_xticks(SQ(ticks)); ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlim(-0.05, SQ(FOLLOWUP_MAX) + 0.1); ax.set_ylim(-ymax - 15, 70)
    ax.set_yticks([]); ax.set_xlabel(L["xlab"])
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.set_title(L["tA"].format(n=len(d)), loc="left", fontsize=12, fontweight="bold", color=TEXT)
    ax.legend(handles=[Line2D([], [], color=SUBTEXT, lw=2.5, label=L["hosp"]),
                       Line2D([], [], color=SUBTEXT, lw=2.5, alpha=0.35, label=L["follow"]),
                       Line2D([], [], color=TEXT, marker="x", ls="", label=L["death_mark"])],
              loc="lower right", frameon=False, fontsize=8.5)

    # ---------------- B: classification schemes
    ax = fig.add_subplot(gs[0, 0])
    for i, col in enumerate(["grupo_los", "grupo"]):
        left = 0
        for g in GROUPS:
            n = int((d[col] == g).sum())
            ax.barh(i, n, left=left, color=COLORS[g], height=0.55, edgecolor="white", lw=1.5)
            if n >= 25:
                ax.text(left + n / 2, i, str(n), ha="center", va="center", color="white",
                        fontsize=8.5, fontweight="bold")
            elif n > 0:
                ax.annotate(str(n), (left + n / 2, i + 0.28), (left + n / 2, i + 0.42), ha="center",
                            va="bottom", color=TEXT, fontsize=8, arrowprops=dict(arrowstyle="-", color=SUBTEXT, lw=0.6))
            left += n
    ax.set_yticks([0, 1]); ax.set_yticklabels([L["schemeL"], L["schemeF"]], color=TEXT)
    ax.set_ylim(-0.6, 1.75); ax.set_xlim(0, len(d)); ax.set_xlabel(L["xB"])
    ax.legend(handles=[Patch(color=COLORS[g], label=L["short"][g]) for g in GROUPS],
              loc="upper center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_title(L["tB"], loc="left", fontsize=11, fontweight="bold", color=TEXT)

    # ---------------- C: dot profile (survivors)
    ax = fig.add_subplot(gs[0, 1])
    pr = prof[prof["var"] != "age"].reset_index(drop=True)
    yy = np.arange(len(pr))[::-1]
    for g, dy in [("short", 0.22), ("post", 0), ("chronic", -0.22)]:
        ax.scatter(pr[g], yy + dy, s=34, color=COLORS[g], edgecolor="white", lw=0.8, zorder=3,
                   label=L["short"][g])
    for yi, (_, r) in zip(yy, pr.iterrows()):
        ax.hlines(yi, 0, 100, color=GRID, lw=0.5, zorder=1)
        star = "†" if r["p"] < 0.05 else ""
        ax.text(102, yi, f"{fmt_p(r['p'], L['dec'])}{star}", va="center", fontsize=7.5,
                color=TEXT if r["p"] < 0.05 else SUBTEXT, fontweight="bold" if r["p"] < 0.05 else None)
    age = prof.set_index("var").loc["age"]
    ax.set_yticks(yy); ax.set_yticklabels([L["vars"][v] for v in pr["var"]], color=TEXT, fontsize=8.5)
    ax.set_xlim(0, 100); ax.set_xlabel(L["xC"])
    ax.text(102, len(pr) - 0.2, "p", fontsize=8, fontweight="bold", color=TEXT)
    agetxt = (f"{L['vars']['age']}: " + " · ".join(f"{L['short'][g]} {age[g]:.0f}" for g in
              ["short", "post", "chronic"]) + f"  (p {fmt_p(age['p'], L['dec'])})")
    ax.text(-0.45, -0.1, agetxt + "\n" + L["pnote"] + "  † p < 0" + L["dec"] + "05\n" + L["rx_note"],
            fontsize=7.5, color=SUBTEXT, va="top", transform=ax.transAxes)
    ax.legend(loc="lower right", frameon=True, fontsize=8, facecolor="white", edgecolor=GRID)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(L["tC"], loc="left", fontsize=11, fontweight="bold", color=TEXT)

    # ---------------- D: out-of-range markers after 3 weeks (long-COVID survivors)
    ax = fig.add_subplot(gs[0, 2])
    m = mk[mk["post_n"] >= MIN_TESTED].copy().sort_values("post_pct").reset_index(drop=True)
    yy = np.arange(len(m))
    for yi, (_, r) in zip(yy, m.iterrows()):
        ax.hlines(yi, 0, 100, color=GRID, lw=0.5, zorder=1)
        ax.scatter(r["post_pct"], yi + 0.15, s=34, color=COLORS["post"], edgecolor="white", lw=0.8, zorder=3)
        lab = f"{int(r['post_n'])}"
        if r["chronic_n"] >= 3:
            ax.scatter(r["chronic_pct"], yi - 0.15, s=34, color=COLORS["chronic"], marker="D",
                       edgecolor="white", lw=0.8, zorder=3)
            lab += f" | {int(r['chronic_n'])}"
        ax.text(102, yi, lab, va="center", fontsize=7.5, color=SUBTEXT)
    names = m["marcador"] if lang == "PT" else m["marcador"].map(lambda v: MARKER_EN.get(v, v))
    ax.set_yticks(yy); ax.set_yticklabels(names, color=TEXT, fontsize=8.5)
    ax.set_xlim(0, 100); ax.set_xlabel(L["xD"])
    ax.legend(handles=[Line2D([], [], color=COLORS["post"], marker="o", ls="", label=L["winP"]),
                       Line2D([], [], color=COLORS["chronic"], marker="D", ls="", label=L["winC"])],
              loc="lower right", frameon=True, fontsize=8, facecolor="white", edgecolor=GRID)
    ax.text(102, len(m) - 0.2, "n", fontsize=8, fontweight="bold", color=TEXT)
    ax.text(-0.45, -0.1, L["dnote"], fontsize=7.5, color=SUBTEXT, va="top", transform=ax.transAxes)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(L["tD"], loc="left", fontsize=11, fontweight="bold", color=TEXT)

    nolab = int(((d["obito"] == 0) & (d["has_lab"] == 0)).sum())
    fig.text(0.06, 0.012, L["foot"].format(nolab=nolab), fontsize=8, color=SUBTEXT, va="bottom")
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

    mae, ea, eb, presc = load(a.data_dir)
    d, e, labs = build(mae, ea, eb, presc)
    prof = profile_table(d)
    mk = marker_table(e, d)

    # ---- coverage report: which sheets carry temporal information / long-COVID patients
    ids = set(mae["Registro"])
    print(f"Mother sheet: {len(mae)} admissions, {len(ids)} unique patients "
          f"({int(mae['readm'].sum())} admissions from re-admitted patients)")
    for name, df, dated in [(F_EXA, ea, True), (F_EXB, eb, True), (F_PRESC, presc, False)]:
        s = set(df["COD_PACIENTE"])
        lc = d.loc[(d["long_covid"] == 1) & d["Registro"].isin(s), "Registro"].nunique()
        print(f"  {name}: {len(df)} rows, {len(s & ids)}/{len(ids)} patients of mother sheet, "
              f"{len(s - ids)} outside it, dated={dated}, long-COVID patients present={lc}")
    print(f"Lab records per admission window: "
          f"{labs.assign(w=pd.cut(labs['t'], [-np.inf, 0, 21, 84, 365, np.inf])).groupby('w', observed=False).size().to_dict()}")
    print("\nClassification (LOS only):", d["grupo_los"].value_counts().to_dict())
    print("Classification (LOS + lab follow-up):", d["grupo"].value_counts().to_dict())
    print(pd.crosstab(d["grupo_los"], d["grupo"], margins=True))
    surv = d[d["obito"] == 0]
    print(f"\nSurvivors: {len(surv)}; with lab data: {int(surv['has_lab'].sum())}; "
          f"long COVID: {int(surv['long_covid'].sum())} ({100 * surv['long_covid'].mean():.1f}% of survivors)")
    print("\nProfile (survivors; % or median):")
    print(prof.round(3).to_string(index=False))
    print("\nOut-of-range markers in long-COVID survivors:")
    print(mk.round(1).to_string(index=False))

    keep = ["ep", "Idade", "obito", "los", "last_lab", "T", "post_discharge", "grupo_los", "grupo",
            "long_covid", "has_lab", "has_rx", "readm"]
    d[keep].to_csv(os.path.join(a.out_dir, "long_covid_classificacao.csv"), index=False)
    prof.to_csv(os.path.join(a.out_dir, "long_covid_perfil.csv"), index=False)
    mk.to_csv(os.path.join(a.out_dir, "long_covid_marcadores.csv"), index=False)
    for lang in ["PT", "EN"]:
        figure(d, prof, mk, lang, os.path.join(a.out_dir, f"Figure_long_covid_{lang}"))
    print("\nSaved figures and tables to", a.out_dir)


if __name__ == "__main__":
    main()
