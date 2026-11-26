#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera os arquivos de dados usados por 12_paineis_laboratoriais_por_dominio.py
a partir da planilha bruta de exames (ExamesB_filtradaa.xlsx).

Saidas (em --outdir, NAO versionar: contem dados por paciente):
    patient_median_data.pkl        STATS[var] = {p, n_alta, n_obito, med/q1/q3 por grupo}
    patient_median_raw_arrays.pkl  RAW[var]   = {"alta": array, "obito": array}
    ref_lookup.pkl                 REFS[var]  = [(rotulo, (lo, hi), unidade), ...]

Metodo (igual aos Quadros 4 e 5 da tese): mediana individual de cada paciente
ao longo de toda a internacao; comparacao Alta vs. Obito pelo teste de
Mann-Whitney U (bilateral). Referencias clinicas adultas dos Quadros 4 e 5.

Uso:
    python 13_prepara_dados_laboratoriais.py --xlsx ExamesB_filtradaa.xlsx --outdir dados_lab
"""
import argparse
import os
import pickle
import re
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu

df = None  # preenchido em main()

HEMO = ['HEMOGRAMA COMPLETO', 'HEMOGRAMA COMPLETO (T+E)']
GA = ['GASOMETRIA', 'GASOMETRIA ARTERIAL']
GV = ['GASOMETRIA VENOSA', 'GASOMETRIA VENOSA CENTRAL']

MAP = {
    # ---------------- Quadro 5
    'RPCU': [('RPCU*', None)],
    'Creatinina': [('CREATININA', 'VALOR')],
    'Ureia': [('UREIA', 'VALOR')],
    'D-dímero': [("QUANTIFICACAO DE DIMEROS 'D' DE FIBRINA", 'VALOR')],
    'Fibrinogênio': [('FIBRINOGENIO', 'VALOR')],
    'TTPA': [('TTPA (TEMPO TROMBOPL PARC/E ATIVADO)', 'RATIO')],
    'TP/INR': [('TP (TEMPO DE PROTROMBINA)', 'INR')],
    'Hb1Ac': [('HEMOGLOBINA GLICOSILADA', 'Resultado')],
    'Glicemia/jejum': [('GLICEMIA', 'VALOR')],
    'TGO': [('TGO / AST - TRANSAMINASE GLUTAMICA OXALACETICA', 'VALOR')],
    'TGP': [('TGP / ALT - ALANINA AMINOTRANSFERASE', 'VALOR')],
    'Bilirrubina': [('BILIRRUBINAS', 'BILIRRUBINA TOTAL'), ('BILIRRUBINA TOTAL', 'VALOR')],
    'Fosfatase alcalina': [('FOSFATASE ALCALINA', 'VALOR')],
    'Gama GT': [('GAMA GT', 'VALOR')],
    'Albumina': [('ALBUMINA - SANGUE', 'VALOR')],
    'Troponina I': [('TROPONINA I', 'VALOR'), ('TROPONINA I', 'RESULTADO')],
    'CPK': [('CPK - CREATINA QUINASE', 'VALOR')],
    'CK-MB': [('CK-MB - CREATINE PHOSPHOKINASE FRACAO 2', 'VALOR')],
    'Colesterol total': [('LIPIDOGRAMA COM JEJUM', 'Colesterol'), ('LIPIDOGRAMA SEM JEJUM', 'Colesterol')],
    'Triglicérides': [('LIPIDOGRAMA COM JEJUM', 'Triglic*'), ('LIPIDOGRAMA SEM JEJUM', 'Triglic*'),
                      ('TRIGLICERIDES COM JEJUM', 'VALOR'), ('TRIGLICERIDES SEM JEJUM', 'VALOR')],
    'LDL colesterol': [('LIPIDOGRAMA COM JEJUM', 'LDL colesterol'), ('LIPIDOGRAMA SEM JEJUM', 'LDL colesterol')],
    'HDL colesterol': [('LIPIDOGRAMA COM JEJUM', 'HDL colesterol'), ('LIPIDOGRAMA SEM JEJUM', 'HDL colesterol')],
    'Ferritina': [('FERRITINA', 'Resultado')],
    'Proteína C reativa': [('PROTEINA C REATIVA', 'VALOR')],
    'Lactato': [('LACTATO', 'VALOR'), ('LACTATO - SANGUE', 'VALOR')],
    'LDH': [('LDH - LACTATO DESIDROGENASE', 'VALOR')],
    'IL-6': [('INTERLEUCINA 6 (IL-6)', 'VALOR')],
    'Cálcio total': [('CALCIO TOTAL', 'VALOR')],
    'Cálcio ionizado': [('CALCIO IONICO', 'VALOR'), ('SODIO, POTASSIO, CALCIO IONICO', 'CALCIO IONICO')],
    'Sódio': [('SODIO', 'VALOR'), ('SODIO', 'Sodio')],
    'Potássio': [('POTASSIO', 'VALOR'), ('SODIO, POTASSIO, CALCIO IONICO', 'POTASSIO')],
    'Cloro': [('CLORO - SANGUE', 'VALOR'), ('SODIO, POTASSIO, CALCIO IONICO, CLORO', 'CLORO')],
    'Magnésio': [('MAGNESIO (SANGUE)', 'VALOR')],
    'LCR-Lactato': [('LACTATO - LIQUOR', 'VALOR')],
    'Procalcitonina': [('PROCALCITONINA*', None)],
    'Ácido úrico': [('ACIDO URICO', 'VALOR')],
    # ---------------- Quadro 4
    'Plaquetas': [(h, 'PLQ - PLAQUETAS') for h in HEMO],
    'Glóbulos brancos': [(h, 'GB - GLOBULOS BRANCOS') for h in HEMO],
    'Glóbulos vermelhos': [(h, 'GV - GLOBULOS VERMELHOS') for h in HEMO],
    'Hematócrito': [(h, 'HT - HEMATOCRITO') for h in HEMO],
    'RDW': [(HEMO[0], 'RDW*'), (HEMO[1], 'RDW')],
    'Monócitos': [(h, 'MONOCITOS') for h in HEMO],
    'Neutrófilos segment.': [(HEMO[0], 'NEUTROFILOS SEGMENTADOS'), (HEMO[1], 'SEGMENTADOS')],
    'Linfócitos': [(h, 'LINFOCITOS') for h in HEMO],
    'Basófilos': [(h, 'BASOFILOS') for h in HEMO],
    'Eosinófilos': [(h, 'EOSINOFILOS') for h in HEMO],
    'CHCM': [(HEMO[0], 'CHCM*'), (HEMO[1], 'CHCM')],
    'HCM': [(HEMO[0], 'HCM - *'), (HEMO[1], 'HCM')],
}
for tag, src in (('art', GA), ('ven', GV)):
    MAP[f'PCO2|{tag}'] = [(e, 'PCO2') for e in src]
    MAP[f'BE|{tag}'] = [(e, 'BE') for e in src]
    MAP[f'HCO3|{tag}'] = [(e, 'HCO3') for e in src]
    MAP[f'FiO2|{tag}'] = [(e, 'FiO2*') for e in src]
    MAP[f'P50c|{tag}'] = [(e, 'p50c') for e in src]
    MAP[f'{"SaO2" if tag == "art" else "SvO2"}|{tag}'] = [(e, 'SATURACAO DE O2') for e in src]
    MAP[f'PO2|{tag}'] = [(e, 'PO2') for e in src]
    MAP[f'pH|{tag}'] = [(e, 'pH') for e in src]
    MAP[f'Temperatura|{tag}'] = [(e, 'TEMPERATURA') for e in src]

num_re = re.compile(r'-?\d+(?:[.,]\d+)?')


def to_num(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v).strip()
    m = num_re.search(s)
    if not m:
        return np.nan
    return float(m.group(0).replace(',', '.'))


def match(series, pat):
    if pat is None:
        return pd.Series(True, index=series.index)
    if pat.endswith('*'):
        return series.str.startswith(pat[:-1], na=False)
    return series == pat


def compute(var, mode='median'):
    parts = []
    for ex, de in MAP[var]:
        m = match(df.EXAME, ex) & match(df.DESCRICAO, de)
        parts.append(df[m])
    sub = pd.concat(parts) if parts else df.iloc[:0]
    sub = sub.assign(v=sub.RESULTADO.map(to_num).astype(float)).dropna(subset=['v'])
    per = sub.groupby(['COD_PACIENTE', 'Obito']).v.median().reset_index()
    a = per[per.Obito == 0].v.values
    o = per[per.Obito == 1].v.values
    p = mannwhitneyu(a, o, alternative='two-sided').pvalue if len(a) >= 2 and len(o) >= 2 else np.nan
    q = lambda x, k: np.percentile(x, k) if len(x) else np.nan
    return dict(p=p, n_alta=len(a), n_obito=len(o), med_alta=q(a, 50), q1_alta=q(a, 25), q3_alta=q(a, 75),
                med_obito=q(o, 50), q1_obito=q(o, 25), q3_obito=q(o, 75)), {'alta': a, 'obito': o}




def _r(lo, hi, rot="Geral", un=""):
    return (rot, (lo, hi), un)


def _mf(m, f, un=""):
    return [_r(m[0], m[1], "Masculino", un), _r(f[0], f[1], "Feminino", un)]


REFS = {
    # Quadro 4
    "Plaquetas": _mf((166, 389), (203, 445)), "Glóbulos brancos": _mf((3.79, 10.33), (4.05, 11.84)),
    "Glóbulos vermelhos": _mf((4.57, 5.98), (4.01, 5.29)), "Hematócrito": _mf((39.6, 51.8), (35.4, 46.3)),
    "RDW": [_r(11.5, 14.7)], "Monócitos": _mf((0.24, 0.73), (0.2, 0.65)),
    "Linfócitos": _mf((1.07, 3.12), (1.17, 3.45)), "Basófilos": _mf((0.02, 0.11), (0.02, 0.11)),
    "Eosinófilos": _mf((0.03, 0.47), (0.03, 0.47)), "CHCM": _mf((33, 37.2), (32.6, 36.5)),
    "HCM": _mf((27.6, 33.2), (27, 32.9)),
    "PCO2|art": [_r(32, 48)], "BE|art": [_r(-2, 2)], "HCO3|art": [_r(21, 28)], "SaO2|art": [_r(95, 98)],
    "PO2|art": [_r(83, 108)], "pH|art": [_r(7.32, 7.45)],
    "BE|ven": [_r(-2, 2)], "PCO2|ven": [_r(38, 50)], "PO2|ven": [_r(35, 40)], "HCO3|ven": [_r(22, 29)],
    "SvO2|ven": [_r(60, 75)], "pH|ven": [_r(7.32, 7.43)],
    # Quadro 5
    "RPCU": [_r(None, 300)], "Creatinina": _mf((0.7, 1.3), (0.55, 1.02)), "Ureia": [_r(19, 49)],
    "D-dímero": [_r(None, 0.5)], "Fibrinogênio": [_r(200, 393)], "TTPA": [_r(None, 1.26)],
    "TP/INR": [_r(None, 1.3)], "Hb1Ac": [_r(4.3, 6.1)], "Glicemia/jejum": [_r(70, 99)],
    "TGO": [_r(None, 38)], "TGP": [_r(10, 49)], "Bilirrubina": [_r(0.3, 1.2)], "Albumina": [_r(3.2, 4.8)],
    "Troponina I": _mf((None, 53.53), (None, 38.64)), "CPK": _mf((46, 171), (34, 145)), "CK-MB": [_r(None, 25)],
    "Colesterol total": [_r(None, 190, "2a1d - 150a")], "Triglicérides": [_r(None, 150, "20-150a")],
    "LDL colesterol": [_r(None, 130)], "HDL colesterol": [_r(40, None, "20a1d-150a")],
    "Ferritina": _mf((22, 322), (10, 291)), "Proteína C reativa": [_r(None, 1)], "Lactato": [_r(0.5, 2)],
    "LDH": [_r(120, 246)], "Cálcio total": [_r(8.3, 10.6)], "Cálcio ionizado": [_r(1.15, 1.33)],
    "Sódio": [_r(136, 145)], "Potássio": [_r(3.5, 5.1)], "Ácido úrico": _mf((3.7, 9.2), (3.1, 7.8)),
}


def main():
    global df
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--outdir", default="dados_lab")
    args = ap.parse_args()
    df = pd.read_excel(args.xlsx)
    df = df[df.Obito.notna()]
    os.makedirs(args.outdir, exist_ok=True)
    S, R = {}, {}
    for var in MAP:
        S[var], R[var] = compute(var)
    for name, obj in (("patient_median_data.pkl", S), ("patient_median_raw_arrays.pkl", R), ("ref_lookup.pkl", REFS)):
        with open(os.path.join(args.outdir, name), "wb") as f:
            pickle.dump(obj, f)
    print(f"{'Variável':22s} {'n Alta':>6s} {'n Óbito':>7s}  Mediana Alta (IIQ)        Mediana Óbito (IIQ)       p")
    for var, s in S.items():
        print(f"{var:22s} {s['n_alta']:6d} {s['n_obito']:7d}  {s['med_alta']:.4g} ({s['q1_alta']:.4g}-{s['q3_alta']:.4g})"
              f"   {s['med_obito']:.4g} ({s['q1_obito']:.4g}-{s['q3_obito']:.4g})   p={s['p']:.4f}")


if __name__ == "__main__":
    main()
