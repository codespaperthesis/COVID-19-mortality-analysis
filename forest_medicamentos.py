"""Forest plot — medicamentos prescritos (subamostra harmonizada, n = 508).
Valores transcritos da figura original; exportação em 600 dpi."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# nome, n, %, OR bruto, IC inf, IC sup, sig, OR ajustado, IC inf, IC sup, sig
ROWS = [
 ("Morfina",175,"34,4",2.97,2.03,4.34,"***",5.68,2.87,11.25,"***"),
 ("Midazolam",274,"53,9",7.37,4.89,11.13,"***",5.51,1.97,15.40,"*"),
 ("Bromoprida",303,"59,6",8.13,5.23,12.65,"***",4.57,2.33,8.95,"***"),
 ("Vancomicina",140,"27,6",4.22,2.79,6.39,"***",2.48,1.03,5.98,"*"),
 ("Insulina",262,"51,6",7.44,4.96,11.17,"***",2.39,0.99,5.77,""),
 ("Meropenem",143,"28,1",5.36,3.51,8.18,"***",2.19,0.85,5.65,""),
 ("AAS (ácido acetilsalicílico)",79,"15,6",2.17,1.33,3.53,"**",1.80,0.79,4.09,""),
 ("Norepinefrina/epinefrina",230,"45,3",6.52,4.41,9.64,"***",1.48,0.66,3.36,""),
 ("Heparina",230,"45,3",5.58,3.80,8.19,"***",1.48,0.79,2.79,""),
 ("Furosemida",232,"45,7",5.00,3.42,7.32,"***",1.42,0.69,2.93,""),
 ("Solução de glicose",274,"53,9",5.06,3.42,7.49,"***",1.35,0.59,3.08,""),
 ("Amoxicilina (ou + clavulanato)",108,"21,3",0.59,0.38,0.93,"*",1.35,0.70,2.59,""),
 ("Metil/Prednisona/Prednisolona",204,"40,2",1.90,1.32,2.72,"***",1.27,0.70,2.30,""),
 ("Omeprazol",336,"66,1",6.74,4.24,10.72,"***",1.25,0.58,2.69,""),
 ("Ceftriaxona",221,"43,5",2.44,1.70,3.50,"***",1.23,0.71,2.11,""),
 ("Ondansetrona",294,"57,9",1.66,1.16,2.39,"**",1.19,0.64,2.23,""),
 ("Dexametasona",215,"42,3",2.50,1.74,3.59,"***",1.08,0.60,1.95,""),
 ("Enoxaparina",238,"46,9",1.72,1.21,2.46,"**",0.97,0.50,1.87,""),
 ("Atorvastatina",41,"8,1",2.06,1.08,3.94,"*",0.93,0.31,2.81,""),
 ("Sertralina",40,"7,9",0.91,0.47,1.76,"",0.91,0.35,2.36,""),
 ("Sulfametoxazol + Trimetoprima",25,"4,9",0.76,0.33,1.76,"",0.90,0.26,3.18,""),
 ("Gabapentina",57,"11,2",1.61,0.93,2.80,"",0.85,0.32,2.23,""),
 ("Anlodipino",89,"17,5",1.36,0.86,2.15,"",0.85,0.42,1.69,""),
 ("Amitriptilina",49,"9,6",1.13,0.63,2.05,"",0.67,0.24,1.88,""),
 ("Fentanila",260,"51,2",5.75,3.89,8.50,"***",0.62,0.21,1.85,""),
 ("Quetiapina",112,"22,0",1.81,1.19,2.77,"**",0.62,0.31,1.25,""),
 ("Clonazepam",74,"14,6",0.71,0.42,1.18,"",0.58,0.27,1.25,""),
 ("Paracetamol",212,"41,7",1.13,0.79,1.61,"",0.51,0.28,0.94,"*"),
 ("Diazepam",58,"11,4",0.69,0.39,1.23,"",0.40,0.17,0.95,"*"),
 ("Simeticona",75,"14,8",0.65,0.39,1.08,"",0.30,0.14,0.64,"**"),
 ("Lorazepam",80,"15,7",0.85,0.52,1.38,"",0.29,0.13,0.66,"**"),
 ("Metadona",138,"27,2",2.64,1.77,3.94,"***",0.24,0.09,0.64,"**"),
 ("Tramadol",136,"26,8",0.99,0.66,1.47,"",0.22,0.11,0.44,"***"),
 ("Dipirona",490,"96,5",1.15,0.44,3.02,"",0.16,0.04,0.57,"**"),
]
N_TOTAL = 508
RISK, PROT, REF = "#E07B39", "#1D9E75", "#C9A227"
TEXT, SUB, GREY, SHADE = "#2F3437", "#8B949E", "#57606A", "#EEF1F5"
plt.rcParams["font.family"] = "DejaVu Sans"

fmt = lambda v: f"{v:.2f}".replace(".", ",")
n = len(ROWS)
fig = plt.figure(figsize=(10.4, 10.9), facecolor="white")
# column layout in figure fractions
top, bottom = 0.845, 0.075
axL = fig.add_axes([0.0, bottom, 0.335, top - bottom])     # labels
axC = fig.add_axes([0.34, bottom, 0.17, top - bottom])   # crude plot
axCt = fig.add_axes([0.515, bottom, 0.15, top - bottom])   # crude text
axA = fig.add_axes([0.67, bottom, 0.17, top - bottom])  # adjusted plot
axAt = fig.add_axes([0.845, bottom, 0.155, top - bottom]) # adjusted text
ys = list(range(n))[::-1]
for ax in (axL, axC, axCt, axA, axAt):
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
for ax in (axL, axCt, axAt):
    ax.set_xlim(0, 1); ax.set_xticks([])

def draw(ax, i, y, orv, lo, hi, sig):
    col = PROT if orv < 1 else RISK
    ax.plot([lo, hi], [y, y], color=col, lw=1.8, solid_capstyle="round", zorder=3)
    if sig:
        ax.plot(orv, y, marker="D", ms=7, color=col, mec=col, zorder=4)
    else:
        ax.plot(orv, y, marker="o", ms=7, mfc="white", mec=col, mew=1.8, zorder=4)

for ax in (axC, axA):
    ax.set_xscale("log"); ax.set_xlim(1e-2, 1e2)
    for k, y in enumerate(ys):
        if k % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color=SHADE, lw=0, zorder=0)
    ax.axvline(1, color=REF, ls="--", lw=1.6, zorder=2)
    ax.spines["bottom"].set_visible(True); ax.spines["bottom"].set_color(GREY)
    ax.tick_params(axis="x", labelsize=8.5, colors=GREY)
    ax.set_xlabel("Razão de Chances (escala log)", fontsize=9.5, color=TEXT)

for (name, nn, pct, c, cl, ch, cs, a, al, ah, as_), y in zip(ROWS, ys):
    axL.text(0.02, y, name, va="center", ha="left", fontsize=10, color=TEXT)
    axL.text(0.985, y, f"n = {nn} ({pct}%)", va="center", ha="right", fontsize=7.5,
             color=SUB, style="italic")
    draw(axC, 0, y, c, cl, ch, cs)
    draw(axA, 0, y, a, al, ah, as_)
    axCt.text(0.05, y, f"{fmt(c)} ({fmt(cl)}–{fmt(ch)}){cs}", va="center", fontsize=8.6, color=TEXT)
    axAt.text(0.05, y, f"{fmt(a)} ({fmt(al)}–{fmt(ah)}){as_}", va="center", fontsize=8.6, color=TEXT)

fig.text(0.425, top + 0.012, "OR bruto (IC 95%)", ha="center", fontsize=11.5, fontweight="bold", color=TEXT)
fig.text(0.755, top + 0.012, "OR ajustado (IC 95%)", ha="center", fontsize=11.5, fontweight="bold", color=TEXT)
fig.text(0.012, 0.965, "Forest Plot — Razão de Chances Bruta e Ajustada", fontsize=17,
         fontweight="bold", color=TEXT, va="bottom")
fig.text(0.012, 0.952, f"Medicamentos prescritos (subamostra harmonizada) | n = {N_TOTAL}",
         fontsize=10.5, color=SUB, va="top")
handles = [mpatches.Patch(color=PROT, label="Fator protetor (OR < 1)"),
           mpatches.Patch(color=RISK, label="Fator de risco (OR > 1)"),
           Line2D([0], [0], marker="D", color="none", mfc="#3A3F44", mec="#3A3F44", ms=8, label="Losango = p < 0,05"),
           Line2D([0], [0], marker="o", color="none", mfc="white", mec="#3A3F44", mew=1.8, ms=8, label="Círculo aberto = p ≥ 0,05"),
           Line2D([0], [0], color=REF, ls="--", lw=1.6, label="Linha de referência (OR = 1)")]
leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.008, 0.925), ncol=5,
                 fontsize=8.8, frameon=True, handlelength=1.4, columnspacing=1.0, borderpad=0.5)
leg.get_frame().set_facecolor(SHADE); leg.get_frame().set_edgecolor("none")
fig.text(0.012, 0.02, "*** p<0,001   ** p<0,01   * p<0,05  |  OR = razão de chances; IC = intervalo de confiança",
         fontsize=7.5, color=SUB, style="italic")

for ext, kw in [("tiff", {"pil_kwargs": {"compression": "tiff_lzw"}}), ("png", {})]:
    fig.savefig(f"forest_medicamentos_600dpi.{ext}", dpi=600, facecolor="white", **kw)
print("ok")
