#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera os arquivos intermediarios usados por painel_hemato_gasometria.py:
  hemato_gaso_stats.pkl, hemato_gaso_raw.pkl, hemato_gaso_refs.pkl

Fontes (na mesma pasta deste script):
  ExamesB_filtradaa.xlsx        - resultados brutos (subamostra n=508)
  valores_de_ref_pesquisa.xlsx  - aba "hemograma" (adultos 18-65 anos, M/F)
                                  e aba "geral" (gasometria arterial/venosa)

Mediana por paciente unico; comparacao Alta vs Obito por Mann-Whitney U.
"""
import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HEMO = {"HEMOGRAMA COMPLETO (T+E)", "HEMOGRAMA COMPLETO"}
ART = {"GASOMETRIA", "GASOMETRIA ARTERIAL"}          # "GASOMETRIA" sem rotulo = arterial
VEN = {"GASOMETRIA VENOSA", "GASOMETRIA VENOSA CENTRAL"}

HEMO_DESC = {
    "Plaquetas": ["PLQ - PLAQUETAS"],
    "Glóbulos brancos": ["GB - GLOBULOS BRANCOS"],
    "Glóbulos vermelhos": ["GV - GLOBULOS VERMELHOS"],
    "Hematócrito": ["HT - HEMATOCRITO"],
    "RDW": ["RDW", "RDW - VariaÃ§Ã£o da distribuiÃ§Ã£o de eritrÃ³citos"],
    "Monócitos": ["MONOCITOS"],
    "Neutrófilos segment.": ["SEGMENTADOS", "NEUTROFILOS SEGMENTADOS"],
    "Linfócitos": ["LINFOCITOS"],
    "Basófilos": ["BASOFILOS"],
    "Eosinófilos": ["EOSINOFILOS"],
    "CHCM": ["CHCM", "CHCM - Conc. de Hemoglob. Corpuscular mÃ©dia"],
    "HCM": ["HCM", "HCM - Hemoglobina corpuscular mÃ©dia"],
}
GASO_DESC = {
    "PCO2": ["PCO2"], "BE": ["BE"], "HCO3": ["HCO3"],
    "FiO2": ["FiO2", "FiO2 (informado no pedido)"], "P50c": ["p50c"],
    "SaO2": ["SATURACAO DE O2"], "SvO2": ["SATURACAO DE O2"], "PO2": ["PO2"],
    "pH": ["pH"], "Temperatura": ["TEMPERATURA"],
}

# Elemento na aba "hemograma" (faixa 018ANOS 01DIAS - 065ANOS); RDW igual para M e F -> geral
HEMO_REF_ELEM = {
    "Plaquetas": "PLQ - PLAQUETAS",
    "Glóbulos brancos": "GB - GLOBULOS BRANCOS",
    "Glóbulos vermelhos": "GV - GLOBULOS VERMELHOS",
    "Hematócrito": "HT - HEMATOCRITO",
    "Monócitos": "MONOCITOS",
    "Linfócitos": "LINFOCITOS",
    "Basófilos": "BASOFILOS",
    "Eosinófilos": "EOSINOFILOS",
    "CHCM": "CHCM - Conc. de Hemoglob. Corpuscular média",
    "HCM": "HCM - Hemoglobina corpuscular média",
}
# Aba "geral": gasometria arterial e venosa (VN em texto livre; BE "0 - ±2" = -2 a +2)
GASO_REFS = {
    "pH_art": (7.32, 7.45), "PO2_art": (83, 108), "PCO2_art": (32, 48),
    "HCO3_art": (21, 28), "BE_art": (-2, 2), "SaO2_art": (95, 98),
    "pH_ven": (7.32, 7.43), "PO2_ven": (35, 40), "PCO2_ven": (38, 50),
    "HCO3_ven": (22, 29), "BE_ven": (-2, 2), "SvO2_ven": (60, 75),
}


def parse_vn(txt):
    lo, hi = txt.replace("VN:", "").strip().split("-")
    return float(lo.replace(",", ".")), float(hi.replace(",", "."))


def build_refs():
    ref = pd.read_excel(os.path.join(BASE_DIR, "valores_de_ref_pesquisa.xlsx"), sheet_name="hemograma")
    ref = ref[(ref["IDADE INICIAL"] == "018ANOS 01DIAS") & (ref["IDADE FINAL"] == "065ANOS")]
    refs = {}
    for var, elem in HEMO_REF_ELEM.items():
        rows = ref[ref["ELEMENTO"] == elem].drop_duplicates(["SEXO", "REFERENCIA"])
        refs[var] = [(sexo, *parse_vn(rows.loc[rows["SEXO"] == sexo, "REFERENCIA"].iloc[0]))
                     for sexo in ("M", "F")]
    rdw = ref[ref["ELEMENTO"] == "RDW - Variação da distribuição de eritrócitos"]["REFERENCIA"].iloc[0]
    refs["RDW"] = [("T", *parse_vn(rdw))]
    for var, (lo, hi) in GASO_REFS.items():
        refs[var] = [("T", lo, hi)]
    return refs


def main():
    df = pd.read_excel(os.path.join(BASE_DIR, "ExamesB_filtradaa.xlsx"))
    df["v"] = pd.to_numeric(df["RESULTADO"].astype(str).str.strip().str.replace(",", ".", regex=False),
                            errors="coerce")
    df = df.dropna(subset=["v", "Obito"])

    spec = {var: (HEMO, desc) for var, desc in HEMO_DESC.items()}
    for var, desc in GASO_DESC.items():
        if var != "SvO2":
            spec[var + "_art"] = (ART, desc)
        if var != "SaO2":
            spec[var + "_ven"] = (VEN, desc)

    stats, raw = {}, {}
    for var, (exames, desc) in spec.items():
        sub = df[df["EXAME"].isin(exames) & df["DESCRICAO"].isin(desc)]
        pm = sub.groupby(["COD_PACIENTE", "Obito"])["v"].median().reset_index()
        alta = pm.loc[pm["Obito"] == 0, "v"].to_numpy()
        obito = pm.loc[pm["Obito"] == 1, "v"].to_numpy()
        p = mannwhitneyu(alta, obito).pvalue if len(alta) and len(obito) else np.nan
        raw[var] = {"alta": alta, "obito": obito}
        stats[var] = dict(med_alta=np.median(alta), q1_alta=np.percentile(alta, 25),
                          q3_alta=np.percentile(alta, 75), med_obito=np.median(obito),
                          q1_obito=np.percentile(obito, 25), q3_obito=np.percentile(obito, 75),
                          p=p, n_alta=len(alta), n_obito=len(obito))

    for name, obj in [("stats", stats), ("raw", raw), ("refs", build_refs())]:
        with open(os.path.join(BASE_DIR, f"hemato_gaso_{name}.pkl"), "wb") as f:
            pickle.dump(obj, f)
    print("saved")


if __name__ == "__main__":
    main()
