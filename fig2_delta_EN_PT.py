"""
Figure 2 — Age (A) and hospital length of stay (B) by clinical outcome,
with the difference in medians (Death − Discharge) drawn inside each panel.
Generates two versions from the REAL cohort data (New_pacientes703.xlsx, n=703):
  Figure2_delta_EN.tiff (English) and Figure2_delta_PT.tiff (Portuguese).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
import statsmodels.formula.api as smf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "New_pacientes703.xlsx")
DPI = 600
SEED = 42
N_BOOT = 5000
IMPUTE_GROUP_MEAN = True   # rule described in Methods (no missing values in the current file)

TEXT, SUBTEXT, BORDER, PANEL = "#1F2328", "#57606A", "#D0D7DE", "#F6F8FA"
COLORS = {"overall": "#7C6FC0", "death": "#E07B39", "discharge": "#1D9E75"}
POS = {"overall": 2.0, "death": 1.0, "discharge": 0.0}

LANG = {
    "EN": dict(
        groups={"overall": "Overall", "death": "Death", "discharge": "Discharge"},
        xlabel={"age": "Age (years)", "los": "Hospital length of stay (days)"},
        unit={"age": "years", "los": "days"},
        unit1={"age": "year", "los": "day"},
        stats="Md = {med} (95% CI {lo}–{hi}); IQR {q1}–{q3}; x̄ = {mean} ± {sd}; max = {mx}",
        model="Negative binomial model: IRR = {irr} (95% CI {lo}–{hi}); {p}",
        mw="Mann–Whitney U test: p < 0.001; Shapiro–Wilk: W = 0.897; p < 0.001",
        pless="p < 0.001", peq="p = {p}",
        delta="ΔMd = {d} {u}",
        legend=["Median (bootstrap 95% CI)", "IQR (P25–P75)", "Mean", "Individual patients",
                "Difference in medians (Death − Discharge)"],
        dec=".",
    ),
    "PT": dict(
        groups={"overall": "Geral", "death": "Óbito", "discharge": "Alta"},
        xlabel={"age": "Idade (anos)", "los": "Tempo de internação (dias)"},
        unit={"age": "anos", "los": "dias"},
        unit1={"age": "ano", "los": "dia"},
        stats="Md = {med} (IC 95% {lo}–{hi}); IIQ {q1}–{q3}; x̄ = {mean} ± {sd}; máx. = {mx}",
        model="Modelo binomial negativo: RTI = {irr} (IC 95% {lo}–{hi}); {p}",
        mw="Teste de Mann-Whitney: p < 0,001; Shapiro-Wilk: W = 0,897; p < 0,001",
        pless="p < 0,001", peq="p = {p}",
        delta="ΔMd = {d} {u}",
        legend=["Mediana (IC 95% bootstrap)", "IIQ (P25–P75)", "Média", "Pacientes individuais",
                "Diferença entre medianas (Óbito − Alta)"],
        dec=",",
    ),
}

# ── Data ────────────────────────────────────────────────────────────────────
df = pd.read_excel(DATA_PATH, sheet_name="Sheet1")
df = df.rename(columns={"Óbito": "death", "Idade": "age", "Dias_permanência": "los"})
df = df[["death", "age", "los"]].copy()
if IMPUTE_GROUP_MEAN:
    for col in ["age", "los"]:
        df[col] = df[col].fillna(df.groupby("death")[col].transform("mean"))
else:
    df = df.dropna()


def get(var, key):
    if key == "overall":
        s = df[var]
    else:
        s = df.loc[df["death"] == (1 if key == "death" else 0), var]
    return s.dropna().to_numpy(dtype=float)


def compute_stats(var):
    rng = np.random.default_rng(SEED)          # same bootstrap in both languages
    out = {}
    for key in ["overall", "death", "discharge"]:
        x = get(var, key)
        q1, med, q3 = np.percentile(x, [25, 50, 75])
        boots = np.median(rng.choice(x, size=(N_BOOT, len(x)), replace=True), axis=1)
        out[key] = dict(x=x, q1=q1, med=med, q3=q3, ci=np.percentile(boots, [2.5, 97.5]),
                        mean=x.mean(), sd=x.std(ddof=1))
    d = df[["death", var]].copy()
    d[var] = d[var].round().clip(lower=0)
    fit = smf.negativebinomial(f"{var} ~ death", data=d).fit(disp=False)
    ci = fit.conf_int().loc["death"]
    nb = (np.exp(fit.params["death"]), np.exp(ci[0]), np.exp(ci[1]), fit.pvalues["death"])
    return out, nb


def num(v, nd, L):
    return f"{v:.{nd}f}".replace(".", L["dec"])


def panel(ax, var, st, nb, L, jitter_seed):
    rng = np.random.default_rng(jitter_seed)   # identical point positions in EN and PT
    for key in ["overall", "death", "discharge"]:
        s, cor, yc = st[key], COLORS[key], POS[key]
        x = s["x"]
        grid = np.linspace(x.min(), x.max(), 400)
        k = gaussian_kde(x, bw_method=0.35)(grid)
        k = k / k.max() * 0.36
        ax.fill_between(grid, yc + 0.03, yc + 0.03 + k, color=cor, alpha=0.28, lw=0)
        ax.plot(grid, yc + 0.03 + k, color=cor, lw=1.2, alpha=0.8)

        by, bw = yc - 0.08, 0.08
        q1, q3, med = s["q1"], s["q3"], s["med"]
        iqr = q3 - q1
        w_lo, w_hi = x[x >= q1 - 1.5 * iqr].min(), x[x <= q3 + 1.5 * iqr].max()
        ax.add_patch(mpatches.Rectangle((q1, by - bw), q3 - q1, 2 * bw,
                                        facecolor=cor, alpha=0.18, edgecolor=cor, lw=1.2))
        ax.plot([w_lo, q1], [by, by], color=cor, lw=1.2)
        ax.plot([q3, w_hi], [by, by], color=cor, lw=1.2)
        for w in (w_lo, w_hi):
            ax.plot([w, w], [by - bw / 2, by + bw / 2], color=cor, lw=1.2)
        outl = x[(x < w_lo) | (x > w_hi)]
        ax.scatter(outl, np.full_like(outl, by), s=12, color=cor, alpha=0.35, lw=0)
        ax.plot(s["ci"], [by, by], color=cor, lw=2.2)
        ax.plot(med, by, marker="D", ms=7, mfc="white", mec=cor, mew=1.8, zorder=5)
        ax.plot([s["mean"]] * 2, [by - bw * 0.6, by + bw * 0.6], color="#444444", lw=1.2, ls="--")

        jy = yc - 0.25 + rng.uniform(-0.04, 0.04, len(x))
        ax.scatter(x, jy, s=5, color=cor, alpha=0.25, lw=0)

        ax.text(-0.02, yc + 0.25, L["groups"][key], transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", color=cor, fontsize=12, fontweight="bold")
        ax.text(-0.02, yc + 0.22, f"n = {len(x)}", transform=ax.get_yaxis_transform(),
                ha="right", va="top", color=SUBTEXT, fontsize=10)
        ax.text(0.99, by - 0.30,
                L["stats"].format(med=f"{med:.0f}", lo=f"{s['ci'][0]:.0f}", hi=f"{s['ci'][1]:.0f}",
                                  q1=f"{q1:.0f}", q3=f"{q3:.0f}", mean=num(s["mean"], 1, L),
                                  sd=num(s["sd"], 1, L), mx=f"{x.max():.0f}"),
                transform=ax.get_yaxis_transform(), color=cor, fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85))

    # ── Δ median (Death − Discharge) drawn between the two groups ──────────
    m_dea, m_dis = st["death"]["med"], st["discharge"]["med"]
    dmd = m_dea - m_dis
    y_arrow = 0.50
    for m, y0 in [(m_dis, POS["discharge"] - 0.08), (m_dea, POS["death"] - 0.08)]:
        ax.plot([m, m], [y0, y_arrow], color="#444444", lw=0.9, ls=(0, (3, 2)), zorder=4)
    ax.annotate("", xy=(m_dea, y_arrow), xytext=(m_dis, y_arrow),
                arrowprops=dict(arrowstyle="<|-|>", color="#333333", lw=1.4,
                                shrinkA=0, shrinkB=0, mutation_scale=10), zorder=6)
    u = L["unit1"][var] if abs(dmd) == 1 else L["unit"][var]
    label = L["delta"].format(d=f"{dmd:+.0f}", u=u)
    bbox = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#333333", lw=0.8)
    if abs(dmd) >= 15:   # wide gap: label centred over the arrow
        ax.text((m_dea + m_dis) / 2, y_arrow + 0.05, label, ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color="#333333", bbox=bbox, zorder=7)
    else:                # narrow gap: label beside the arrow
        ax.text(max(m_dea, m_dis) + 2.0, y_arrow, label, ha="left", va="center",
                fontsize=9.5, fontweight="bold", color="#333333", bbox=bbox, zorder=7)

    irr, lo, hi, p = nb
    ptxt = L["pless"] if p < 0.001 else L["peq"].format(p=num(p, 3, L))
    if var == "age":   # panel A: Mann-Whitney (Shapiro-Wilk rejects normality)
        title = L["mw"]
    else:
        title = L["model"].format(irr=num(irr, 2, L), lo=num(lo, 2, L), hi=num(hi, 2, L), p=ptxt)
    ax.set_title(title, loc="right", fontsize=9, color=TEXT)

    xmax = max(s["x"].max() for s in st.values())
    ax.set_xlim(-2, np.ceil((xmax + 5) / 10) * 10)
    ax.set_ylim(-0.62, 2.7)
    ax.set_yticks([])
    ax.set_facecolor(PANEL)
    ax.xaxis.grid(True, color=BORDER, lw=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel(L["xlabel"][var], fontsize=11, color=SUBTEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)


STATS = {v: compute_stats(v) for v in ["age", "los"]}

for code, L in LANG.items():
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(10, 11), facecolor="white")
    panel(axA, "age", *STATS["age"], L, jitter_seed=1)
    panel(axB, "los", *STATS["los"], L, jitter_seed=2)
    for ax, letter in [(axA, "A)"), (axB, "B)")]:
        ax.text(-0.12, 1.04, letter, transform=ax.transAxes, fontsize=16, fontweight="bold")
    lg = L["legend"]
    handles = [Line2D([0], [0], marker="D", color="none", mfc="white", mec=TEXT, ms=7, label=lg[0]),
               mpatches.Patch(facecolor="none", edgecolor=TEXT, label=lg[1]),
               Line2D([0], [0], color="#444444", ls="--", label=lg[2]),
               Line2D([0], [0], marker="o", color="none", mfc="grey", ms=4, alpha=0.5, label=lg[3]),
               Line2D([0], [0], color="#333333", lw=1.4, marker=">", ms=5, label=lg[4])]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = os.path.join(BASE_DIR, f"Figure2_delta_{code}.tiff")
    plt.savefig(out, dpi=DPI, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("Saved:", out)
