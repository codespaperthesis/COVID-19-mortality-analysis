"""
World map of the countries where the studies included in the systematic review were conducted.
Source: sheet "Incluidos" (titles) matched to the "País" column of sheet "Planilha1"
in Check_list_systematic_review_-_Price.xlsx. A multi-country study counts once for each country.
Generates Map_included_studies_PT.(png|tiff), Map_included_studies_EN.(png|tiff)
and included_studies_by_country.csv (study-country table used in the figure).
"""
import os
import unicodedata
import urllib.request
import pandas as pd
import geopandas as gpd
from shapely.geometry import box as bbox
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Check_list_systematic_review_-_Price.xlsx")
WORLD_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
             "geojson/ne_50m_admin_0_countries.geojson")
WORLD_CACHE = os.path.join(BASE_DIR, ".cache", "ne_50m_admin_0_countries.geojson")
DPI = 600
EUROPE_BOX = bbox(-1.0e6, 3.7e6, 2.6e6, 6.4e6)   # Robinson metres: Iberia/British Isles -> Italy/Germany

# Country column is blank for this study; country taken from the title
COUNTRY_OVERRIDE = {"The Current and Future Burden of Long COVID in the United States": "Estados Unidos"}

# keyword in the normalised "País" text -> ISO3 (England/Scotland/Wales -> United Kingdom)
KEYWORDS = [
    ("eua", "USA"), ("estados unidos", "USA"), ("reino unido", "GBR"), ("inglaterra", "GBR"),
    ("escocia", "GBR"), ("israel", "ISR"), ("franca", "FRA"), ("australia", "AUS"),
    ("japao", "JPN"), ("italia", "ITA"), ("belgica", "BEL"), ("malasia", "MYS"),
    ("mexico", "MEX"), ("alemanha", "DEU"), ("russia", "RUS"), ("taiwan", "TWN"),
    ("espanha", "ESP"), ("brasil", "BRA"),
]

NAMES = {
    "USA": ("Estados Unidos", "United States"), "GBR": ("Reino Unido", "United Kingdom"),
    "DEU": ("Alemanha", "Germany"), "AUS": ("Austrália", "Australia"), "FRA": ("França", "France"),
    "RUS": ("Rússia", "Russia"), "BEL": ("Bélgica", "Belgium"), "BRA": ("Brasil", "Brazil"),
    "ESP": ("Espanha", "Spain"), "ISR": ("Israel", "Israel"), "ITA": ("Itália", "Italy"),
    "JPN": ("Japão", "Japan"), "MEX": ("México", "Mexico"), "MYS": ("Malásia", "Malaysia"),
    "TWN": ("Taiwan", "Taiwan"),
}

# Sequential blue ramp (light -> dark), one class per bin; countries without studies in neutral gray
BINS = [(1, 1), (2, 3), (4, 9), (10, None)]
BIN_COLORS = ["#9ec5f4", "#5598e7", "#256abf", "#104281"]
NO_DATA, LAND_EDGE, OCEAN = "#E6E6E3", "#FFFFFF", "#FFFFFF"
TEXT, SUBTEXT, GRID = "#1F2328", "#57606A", "#D0D7DE"

LANG = {
    "PT": dict(legend_title="Nº de estudos incluídos", none="Sem estudos incluídos",
               bar_title="Estudos por país", europe="Europa ocidental", xlabel="Nº de estudos",
               note="n = {n} estudos incluídos em {k} países; estudos multinacionais contados em cada país."),
    "EN": dict(legend_title="No. of included studies", none="No included studies",
               bar_title="Studies by country", europe="Western Europe", xlabel="No. of studies",
               note="n = {n} included studies from {k} countries; multinational studies counted in each country."),
}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return " ".join(s.replace("\xa0", " ").split())


def to_iso(country_text):
    t = norm(country_text)
    codes = []
    for kw, iso in KEYWORDS:
        if kw in t and iso not in codes:
            codes.append(iso)
    return codes


def study_countries():
    """One row per (included study, country)."""
    p1 = pd.read_excel(DATA_PATH, sheet_name="Planilha1")
    p1 = p1.dropna(subset=[p1.columns[0]])
    country_of = {norm(t): c for t, c in zip(p1.iloc[:, 0], p1.iloc[:, 1])}
    for t, c in COUNTRY_OVERRIDE.items():
        country_of[norm(t)] = c

    inc = pd.read_excel(DATA_PATH, sheet_name="Incluidos", header=None).iloc[:, 0].dropna()
    rows = []
    for i, title in enumerate(inc, 1):
        title = " ".join(str(title).replace("\xa0", " ").split())
        raw = country_of.get(norm(title))
        codes = to_iso(raw) if raw is not None else []
        if not codes:
            raise ValueError(f"No country found for included study #{i}: {title} (País = {raw!r})")
        for iso in codes:
            rows.append(dict(study_id=i, title=title, country_raw=" ".join(str(raw).split()),
                             iso3=iso, country_pt=NAMES[iso][0], country_en=NAMES[iso][1]))
    return pd.DataFrame(rows)


def country_counts():
    sc = study_countries()
    return (sc.groupby(["iso3", "country_pt", "country_en"]).size().rename("n_studies")
            .reset_index().sort_values(["n_studies", "country_en"], ascending=[False, True]))


def load_world():
    if not os.path.exists(WORLD_CACHE):
        os.makedirs(os.path.dirname(WORLD_CACHE), exist_ok=True)
        urllib.request.urlretrieve(WORLD_URL, WORLD_CACHE)
    w = gpd.read_file(WORLD_CACHE)
    # Natural Earth codes France/Norway as -99 in ISO_A3; ADM0_A3 is always filled
    w["iso3"] = w["ADM0_A3"]
    w = w[w["iso3"] != "ATA"]
    return w.to_crs("ESRI:54030")   # Robinson


def label_countries(ax, gdf):
    """Study count on each country; small countries also get a dot so they remain visible."""
    for _, r in gdf[gdf["n_studies"].notna()].iterrows():
        p = r.geometry.representative_point()
        if r.geometry.area < 1.5e11:
            ax.scatter(p.x, p.y, s=30, color=BIN_COLORS[bin_index(r["n_studies"])],
                       edgecolor="white", linewidth=1.0, zorder=5)
        ax.annotate(f"{int(r['n_studies'])}", (p.x, p.y), xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7, color=TEXT, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=6)


def bin_index(n):
    for i, (lo, hi) in enumerate(BINS):
        if n >= lo and (hi is None or n <= hi):
            return i


def bin_label(lo, hi):
    return f"{lo}" if lo == hi else (f"≥ {lo}" if hi is None else f"{lo}–{hi}")


def draw(lang, world, counts, n_studies):
    L = LANG[lang]
    name_col = "country_pt" if lang == "PT" else "country_en"
    w = world.merge(counts[["iso3", "n_studies"]], on="iso3", how="left")
    w["color"] = [BIN_COLORS[bin_index(n)] if pd.notna(n) else NO_DATA for n in w["n_studies"]]

    fig = plt.figure(figsize=(13, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.6, 1], wspace=0.04)
    ax = fig.add_subplot(gs[0])
    w.plot(ax=ax, color=w["color"], edgecolor=LAND_EDGE, linewidth=0.3)
    ax.set_facecolor(OCEAN)
    ax.set_axis_off()

    # Europe is drawn in an inset; on the world map only the countries outside it get labels
    in_europe = w.geometry.representative_point().within(EUROPE_BOX)
    label_countries(ax, w[~in_europe])

    ix = ax.inset_axes([0.0, 0.03, 0.25, 0.36])
    eu = w.cx[EUROPE_BOX.bounds[0]:EUROPE_BOX.bounds[2], EUROPE_BOX.bounds[1]:EUROPE_BOX.bounds[3]]
    eu.plot(ax=ix, color=eu["color"], edgecolor=LAND_EDGE, linewidth=0.4)
    ix.set_xlim(EUROPE_BOX.bounds[0], EUROPE_BOX.bounds[2])
    ix.set_ylim(EUROPE_BOX.bounds[1], EUROPE_BOX.bounds[3])
    label_countries(ix, w[in_europe])
    ix.set_xticks([]); ix.set_yticks([])
    ix.set_xlabel(""); ix.set_ylabel("")
    for sp in ix.spines.values():
        sp.set_color(SUBTEXT); sp.set_linewidth(0.6)
    ix.set_facecolor(OCEAN)
    ix.set_title(L["europe"], fontsize=8, color=SUBTEXT, loc="left", pad=3)
    bx_, by_ = EUROPE_BOX.exterior.xy
    ax.plot(bx_, by_, color=SUBTEXT, lw=0.6, ls="--")

    handles = [Patch(fc=c, ec="none", label=bin_label(lo, hi)) for (lo, hi), c in zip(BINS, BIN_COLORS)]
    handles.append(Patch(fc=NO_DATA, ec="none", label=L["none"]))
    ax.legend(handles=handles, title=L["legend_title"], loc="lower left", bbox_to_anchor=(0.6, 0.0), frameon=False,
              fontsize=8, title_fontsize=8.5, labelcolor=TEXT, handlelength=1.4, borderaxespad=0.2)

    # Ranked bar chart (identifies every country, including the small ones)
    bx = fig.add_subplot(gs[1])
    c = counts.iloc[::-1]
    colors = [BIN_COLORS[bin_index(n)] for n in c["n_studies"]]
    bx.barh(c[name_col], c["n_studies"], color=colors, height=0.68)
    for y, n in enumerate(c["n_studies"]):
        bx.text(n + 0.25, y, str(n), va="center", fontsize=7.5, color=TEXT)
    bx.set_title(L["bar_title"], fontsize=9.5, color=TEXT, loc="left", weight="bold")
    bx.set_xlabel(L["xlabel"], fontsize=8, color=SUBTEXT)
    bx.set_xlim(0, c["n_studies"].max() * 1.15)
    bx.tick_params(axis="y", labelsize=8, colors=TEXT, length=0)
    bx.tick_params(axis="x", labelsize=7.5, colors=SUBTEXT)
    bx.xaxis.grid(True, color=GRID, lw=0.6)
    bx.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        bx.spines[s].set_visible(False)
    bx.spines["bottom"].set_color(GRID)

    fig.text(0.125, 0.06, L["note"].format(n=n_studies, k=len(counts)), fontsize=8, color=SUBTEXT)
    for ext in ("png", "tiff"):
        out = os.path.join(BASE_DIR, f"Map_included_studies_{lang}.{ext}")
        kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tiff" else {}
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white", **kw)
        print("saved", out)
    plt.close(fig)


if __name__ == "__main__":
    sc = study_countries()
    sc.to_csv(os.path.join(BASE_DIR, "included_studies_by_country.csv"), index=False, encoding="utf-8-sig")
    counts = country_counts()
    print(counts.to_string(index=False))
    world = load_world()
    missing = set(counts["iso3"]) - set(world["iso3"])
    if missing:
        raise ValueError(f"Countries not found in the base map: {missing}")
    for lang in LANG:
        draw(lang, world, counts, sc["study_id"].nunique())
