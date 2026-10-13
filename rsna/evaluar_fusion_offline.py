#!/usr/bin/env python
"""Evaluación pareada rigurosa de la fusión offline de modelos (Experimento 0A / 0B).

Analiza:
1. Modelos OOF de Owen sobre sus 4,349 estudios (exploración de ponderaciones por familia sin fuga).
2. Fusión de Owen 15-modelos con modelos especializados de Joako sobre los mismos 415 estudios de val_int.
3. Validación bootstrap pareada (1,000 réplicas, IC 95%) por combinación y por patología.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]


def calcular_macro_auc(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> tuple[float, dict[str, float]]:
    """Calcula el macro ROC-AUC y por patología evaluando solo celdas válidas (weight > 0)."""
    aucs = {}
    for j, col in enumerate(LABELS):
        mask = weights[:, j] > 0
        y_col = y_true[mask, j]
        p_col = y_pred[mask, j]
        if len(np.unique(y_col)) > 1:
            aucs[col] = float(roc_auc_score(y_col, p_col))
        else:
            aucs[col] = float("nan")
    valid_aucs = [v for v in aucs.values() if not np.isnan(v)]
    macro = float(np.mean(valid_aucs)) if valid_aucs else 0.0
    return macro, aucs


def bootstrap_pareado(
    y_true: np.ndarray,
    p_cand: np.ndarray,
    p_base: np.ndarray,
    weights: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42
) -> dict:
    """Calcula bootstrap pareado a nivel estudio (filas completas) para delta de macro AUC."""
    n_samples = len(y_true)
    rng = np.random.RandomState(seed)
    deltas = []

    macro_cand_pt, _ = calcular_macro_auc(y_true, p_cand, weights)
    macro_base_pt, _ = calcular_macro_auc(y_true, p_base, weights)
    delta_puntual = macro_cand_pt - macro_base_pt

    for _ in range(n_boot):
        idx = rng.randint(0, n_samples, size=n_samples)
        y_b = y_true[idx]
        w_b = weights[idx]
        pc_b = p_cand[idx]
        pb_b = p_base[idx]

        m_c, _ = calcular_macro_auc(y_b, pc_b, w_b)
        m_b, _ = calcular_macro_auc(y_b, pb_b, w_b)
        deltas.append(m_c - m_b)

    deltas = np.array(deltas)
    ic_inf = float(np.percentile(deltas, 2.5))
    ic_sup = float(np.percentile(deltas, 97.5))
    p_pos = float(np.mean(deltas > 0.0) * 100.0)

    return {
        "delta_puntual": float(delta_puntual),
        "delta_medio_boot": float(np.mean(deltas)),
        "ic_95_inf": ic_inf,
        "ic_95_sup": ic_sup,
        "p_pos": p_pos,
        "significativo": bool(ic_inf > 0.0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/lustre/proyectos/p082/runs_joako/eval_fusion_offline"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TRABAJO 1: EVALUACIÓN RIGUROSA DE FUSIÓN OFFLINE (0A / 0B)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # PARTE A: ANÁLISIS DE PONDERACIONES OOF EN LOS 4,349 ESTUDIOS DE OWEN
    # -------------------------------------------------------------------------
    print("\n--- [PARTE A] Evaluación OOF de las 5 Familias de Owen (4,349 estudios) ---")
    oof_dir = Path("/lustre/proyectos/p082/runs/2084709")
    labels_v1_path = Path("/lustre/proyectos/p082/preproc/etiquetas_v1_2079203.parquet")
    labels_v1 = pd.read_parquet(labels_v1_path).set_index("StudyInstanceUID")

    df_r34_oof = pd.read_parquet(oof_dir / "oof_r34.parquet").set_index("StudyInstanceUID")
    df_eff_oof = pd.read_parquet(oof_dir / "oof_eff.parquet").set_index("StudyInstanceUID")
    df_cnxhib_oof = pd.read_parquet(oof_dir / "oof_cnxhib.parquet").set_index("StudyInstanceUID")
    df_r34hib_oof = pd.read_parquet(oof_dir / "oof_r34hib.parquet").set_index("StudyInstanceUID")
    df_cnx_oof = pd.read_parquet(oof_dir / "oof_cnx.parquet").set_index("StudyInstanceUID")

    oof_uids = df_eff_oof.index.tolist()
    assert len(oof_uids) == 4349, f"Esperados 4349 estudios OOF, obtenidos {len(oof_uids)}"

    t_oof = labels_v1.loc[oof_uids, [f"t_{c}" for c in LABELS]].to_numpy(np.float32)
    w_oof = labels_v1.loc[oof_uids, [f"w_{c}" for c in LABELS]].to_numpy(np.float32)
    y_oof = (np.nan_to_num(t_oof) >= 0.5).astype(int)

    p_r34_oof = df_r34_oof.loc[oof_uids, LABELS].to_numpy()
    p_eff_oof = df_eff_oof.loc[oof_uids, LABELS].to_numpy()
    p_cnxhib_oof = df_cnxhib_oof.loc[oof_uids, LABELS].to_numpy()
    p_r34hib_oof = df_r34hib_oof.loc[oof_uids, LABELS].to_numpy()
    p_cnx_oof = df_cnx_oof.loc[oof_uids, LABELS].to_numpy()

    # Modelos individuales OOF
    familias_oof = {
        "ResNet34 (v1)": p_r34_oof,
        "EfficientNetV2-S (v1)": p_eff_oof,
        "ConvNeXt-Tiny (v2 hib)": p_cnxhib_oof,
        "ResNet34 (v2 hib)": p_r34hib_oof,
        "ConvNeXt-Tiny (v1 base)": p_cnx_oof,
    }

    res_oof_indiv = {}
    for name, p in familias_oof.items():
        macro, per_pat = calcular_macro_auc(y_oof, p, w_oof)
        res_oof_indiv[name] = {"macro": macro, "per_pat": per_pat}
        print(f"  {name:25s}: macro AUC = {macro:.4f}")

    # Owen 0.912 baseline (promedio uniforme 1/3)
    p_owen_oof_base = (p_r34_oof + p_eff_oof + p_cnxhib_oof) / 3.0
    macro_owen_oof, per_pat_owen_oof = calcular_macro_auc(y_oof, p_owen_oof_base, w_oof)
    print(f"  --> Owen Baseline (r34+eff+cnxhib 1/3): macro AUC = {macro_owen_oof:.4f}")

    # Búsqueda en malla gruesa de ponderaciones sobre los 4,349 estudios (paso 0.05)
    print("\nExplorando malla de ponderaciones (alpha*r34 + beta*eff + gamma*cnxhib) en 4,349 estudios:")
    grid_results = []
    for a in np.linspace(0.0, 1.0, 21):
        for b in np.linspace(0.0, 1.0 - a, int(np.round((1.0 - a) * 20)) + 1):
            c = round(1.0 - a - b, 4)
            if c < -1e-5:
                continue
            c = max(0.0, c)
            p_combo = a * p_r34_oof + b * p_eff_oof + c * p_cnxhib_oof
            m_combo, _ = calcular_macro_auc(y_oof, p_combo, w_oof)
            grid_results.append((m_combo, round(a, 2), round(b, 2), round(c, 2)))

    grid_results.sort(key=lambda x: -x[0])
    top_5_grid = grid_results[:5]
    print("  Top 5 combinaciones de ponderación en OOF:")
    for rank, (m, a, b, c) in enumerate(top_5_grid, 1):
        print(f"    #{rank}: R34={a:.2f}, Eff={b:.2f}, CnxHib={c:.2f} --> macro AUC = {m:.4f} (delta = {m - macro_owen_oof:+.4f})")

    # Exploración de 4 y 5 familias en OOF
    p_4fam = (p_r34_oof + p_eff_oof + p_cnxhib_oof + p_cnx_oof) / 4.0
    m_4fam, _ = calcular_macro_auc(y_oof, p_4fam, w_oof)
    p_5fam = (p_r34_oof + p_eff_oof + p_cnxhib_oof + p_cnx_oof + p_r34hib_oof) / 5.0
    m_5fam, _ = calcular_macro_auc(y_oof, p_5fam, w_oof)
    p_eff_cnxhib = (p_eff_oof + p_cnxhib_oof) / 2.0
    m_eff_cnxhib, _ = calcular_macro_auc(y_oof, p_eff_cnxhib, w_oof)

    print(f"  Variantes adicionales en OOF:")
    print(f"    EffNet + CnxHib (2 vías 50/50, sin R34): macro AUC = {m_eff_cnxhib:.4f} (delta = {m_eff_cnxhib - macro_owen_oof:+.4f})")
    print(f"    4 Familias (+ Cnx Base 25% c/u)        : macro AUC = {m_4fam:.4f} (delta = {m_4fam - macro_owen_oof:+.4f})")
    print(f"    5 Familias (Todas 20% c/u)             : macro AUC = {m_5fam:.4f} (delta = {m_5fam - macro_owen_oof:+.4f})")

    # -------------------------------------------------------------------------
    # PARTE B: EVALUACIÓN PAREADA EN LOS 415 ESTUDIOS DE VAL_INT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("--- [PARTE B] Evaluación Pareada en val_int (415 estudios) ---")
    print("=" * 80)

    # 1. Cargar UIDs y predicciones base de referencia
    val_int_base_csv = Path("/lustre/proyectos/p082/runs_joako/2081432/predicciones_val_int.csv")
    df_val_base = pd.read_csv(val_int_base_csv).set_index("StudyInstanceUID")
    uids_415 = df_val_base.index.tolist()
    assert len(uids_415) == 415, f"Esperados 415 estudios en val_int, obtenidos {len(uids_415)}"

    # Verificar que los 415 UIDs estén en etiquetas de referencia
    t_val = labels_v1.loc[uids_415, [f"t_{c}" for c in LABELS]].to_numpy(np.float32)
    w_val = labels_v1.loc[uids_415, [f"w_{c}" for c in LABELS]].to_numpy(np.float32)
    y_val = (np.nan_to_num(t_val) >= 0.5).astype(int)

    # Predicciones de referencia de modelos Joako
    # a) R34 Base y ConvNeXt Base
    p_r34_base_val = df_val_base[[f"p_resnet_{c}" for c in LABELS]].to_numpy()
    p_cnx_base_val = df_val_base[[f"p_convnext_{c}" for c in LABELS]].to_numpy()
    p_ens_50_50_val = 0.5 * p_r34_base_val + 0.5 * p_cnx_base_val

    # b) ConvNeXt Híbrido (2084609)
    df_cnx_hib_joako = pd.read_csv("/lustre/proyectos/p082/runs_joako/2084609/evaluacion/predicciones_val_int_experimento2.csv").set_index("StudyInstanceUID")
    p_cnx_hib_joako_val = df_cnx_hib_joako.loc[uids_415, [f"p_convnext_hib_{c}" for c in LABELS]].to_numpy()

    # c) Slice Attention ResNet34 (2084614, época 6)
    df_slice_att = pd.read_csv("/lustre/proyectos/p082/runs_joako/2084614/predicciones_val_int_best.csv").set_index("StudyInstanceUID")
    p_slice_att_val = df_slice_att.loc[uids_415, [f"p_{c}" for c in LABELS]].to_numpy()

    # d) EfficientNetV2-S Fold 0 (2084365, evaluado en 2084751)
    df_eff_joako = pd.read_csv("/lustre/proyectos/p082/runs_joako/eval_efficientnet_2084365/predicciones_val_int_efficientnet.csv").set_index("StudyInstanceUID")
    p_eff_fold0_val = df_eff_joako.loc[uids_415, [f"p_{c}" for c in LABELS]].to_numpy()

    # e) ResNet34 Híbrido (2084568)
    p_r34_hib_joako_val = df_cnx_hib_joako.loc[uids_415, [f"p_resnet_hib_{c}" for c in LABELS]].to_numpy()

    # f) Ensambles Joako previos
    p_trio_base_d = 0.25 * p_r34_base_val + 0.25 * p_slice_att_val + 0.50 * p_cnx_base_val
    p_cuadruple_abcd = 0.25 * p_r34_base_val + 0.25 * p_cnx_base_val + 0.25 * p_r34_hib_joako_val + 0.25 * p_cnx_hib_joako_val
    p_quintuple_abcde = 0.20 * p_r34_base_val + 0.20 * p_cnx_base_val + 0.20 * p_r34_hib_joako_val + 0.20 * p_cnx_hib_joako_val + 0.20 * p_eff_fold0_val

    # g) Owen 15-modelos en val_int (provenientes de los parquets OOF de 4,349)
    p_r34_owen_val = df_r34_oof.loc[uids_415, LABELS].to_numpy()
    p_eff_owen_val = df_eff_oof.loc[uids_415, LABELS].to_numpy()
    p_cnxhib_owen_val = df_cnxhib_oof.loc[uids_415, LABELS].to_numpy()
    p_owen_15_val = (p_r34_owen_val + p_eff_owen_val + p_cnxhib_owen_val) / 3.0

    # 2. Comprobar reproducibilidad de baselines
    print("\nVerificación de reproducibilidad de baselines conocidos:")
    baselines_check = [
        ("ResNet34 Base", p_r34_base_val, 0.8435),
        ("ConvNeXt Base", p_cnx_base_val, 0.8518),
        ("Ensamble 50/50", p_ens_50_50_val, 0.8574),
        ("Trío Base+D", p_trio_base_d, 0.8628),
        ("Ensamble ABCD", p_cuadruple_abcd, 0.8653),
        ("Quíntuple ABCDE", p_quintuple_abcde, 0.8670),
        ("EfficientNetV2-S Fold 0", p_eff_fold0_val, 0.8517),
        ("Owen 15 Modelos", p_owen_15_val, 0.8630),
    ]
    for bname, p_b, expected in baselines_check:
        m, _ = calcular_macro_auc(y_val, p_b, w_val)
        status = "OK" if abs(m - expected) < 0.0005 else f"DISCREPANCIA (obs={m:.4f}, exp={expected:.4f})"
        print(f"  {bname:25s}: obs = {m:.4f} | exp = {expected:.4f} [{status}]")

    # 3. Definición de Combinaciones Predefinidas de Fusión
    combinaciones = {
        # Baseline Owen
        "B0_Owen_15_Baseline": p_owen_15_val,
        # Fusión Uniforme (cada modelo un voto)
        "C1_Owen15_mas_SliceAtt_16m": (15.0 / 16.0) * p_owen_15_val + (1.0 / 16.0) * p_slice_att_val,
        "C2_Owen15_mas_SliceAtt_mas_CnxHib_17m": (15.0 / 17.0) * p_owen_15_val + (1.0 / 17.0) * p_slice_att_val + (1.0 / 17.0) * p_cnx_hib_joako_val,
        # Fusión Ponderada Conservadora
        "C3_Owen_80_SliceAtt_20": 0.80 * p_owen_15_val + 0.20 * p_slice_att_val,
        "C4_Owen_70_SliceAtt_15_CnxHib_15": 0.70 * p_owen_15_val + 0.15 * p_slice_att_val + 0.15 * p_cnx_hib_joako_val,
        "C5_Owen_60_SliceAtt_20_CnxHib_20": 0.60 * p_owen_15_val + 0.20 * p_slice_att_val + 0.20 * p_cnx_hib_joako_val,
        # Fusión Equitativa de Enfoques
        "C6_Owen_50_SliceAtt_50": 0.50 * p_owen_15_val + 0.50 * p_slice_att_val,
        "C7_Owen_50_ABCD_50": 0.50 * p_owen_15_val + 0.50 * p_cuadruple_abcd,
        "C8_Owen_50_ABCDE_50": 0.50 * p_owen_15_val + 0.50 * p_quintuple_abcde,
        "C9_Owen_75_ABCDE_25": 0.75 * p_owen_15_val + 0.25 * p_quintuple_abcde,
        # Fusión Ponderación Óptima OOF en Owen + Slice-Att
        "C10_OwenOpt_70_SliceAtt_15_CnxHib_15": 0.70 * (0.15 * p_r34_owen_val + 0.40 * p_eff_owen_val + 0.45 * p_cnxhib_owen_val) + 0.15 * p_slice_att_val + 0.15 * p_cnx_hib_joako_val,
    }

    print("\n" + "-" * 80)
    print("RESULTADOS DE TODAS LAS COMBINACIONES EN val_int (415 estudios)")
    print("-" * 80)
    print(f"{'Combinación':<40} {'Macro AUC':>10} {'Delta vs Owen':>14} {'IC 95% Bootstrap':>22} {'P(>0)':>8}")

    res_val_int = {}
    for cname, p_c in combinaciones.items():
        macro_c, per_pat_c = calcular_macro_auc(y_val, p_c, w_val)
        boot = bootstrap_pareado(y_val, p_c, p_owen_15_val, w_val, n_boot=1000, seed=42)
        res_val_int[cname] = {
            "macro_auc": macro_c,
            "per_pat": per_pat_c,
            "bootstrap_vs_owen": boot,
        }
        ic_str = f"[{boot['ic_95_inf']:+.4f}, {boot['ic_95_sup']:+.4f}]"
        sig_star = " ***" if boot["significativo"] else ""
        print(f"{cname:<40} {macro_c:>10.4f} {boot['delta_puntual']:>+14.4f} {ic_str:>22} {boot['p_pos']:>7.1f}%{sig_star}")

    # Bootstrap pareado del mejor contra Quíntuple ABCDE (0.8670)
    mejor_combo_name = max(combinaciones, key=lambda k: res_val_int[k]["macro_auc"])
    mejor_p = combinaciones[mejor_combo_name]
    boot_vs_abcde = bootstrap_pareado(y_val, mejor_p, p_quintuple_abcde, w_val, n_boot=1000, seed=42)
    boot_vs_abcd = bootstrap_pareado(y_val, mejor_p, p_cuadruple_abcd, w_val, n_boot=1000, seed=42)

    print("\n" + "=" * 80)
    print(f"ANÁLISIS DETALLADO DEL CANDIDATO GANADOR: {mejor_combo_name}")
    print("=" * 80)
    print(f"Macro AUC: {res_val_int[mejor_combo_name]['macro_auc']:.4f}")
    print(f"Delta vs Owen 15 Baseline: {res_val_int[mejor_combo_name]['bootstrap_vs_owen']['delta_puntual']:+.4f} (IC 95%: [{res_val_int[mejor_combo_name]['bootstrap_vs_owen']['ic_95_inf']:+.4f}, {res_val_int[mejor_combo_name]['bootstrap_vs_owen']['ic_95_sup']:+.4f}], P={res_val_int[mejor_combo_name]['bootstrap_vs_owen']['p_pos']:.1f}%)")
    print(f"Delta vs Joako ABCD (0.8653): {boot_vs_abcd['delta_puntual']:+.4f} (IC 95%: [{boot_vs_abcd['ic_95_inf']:+.4f}, {boot_vs_abcd['ic_95_sup']:+.4f}], P={boot_vs_abcd['p_pos']:.1f}%)")
    print(f"Delta vs Joako Quíntuple ABCDE (0.8670): {boot_vs_abcde['delta_puntual']:+.4f} (IC 95%: [{boot_vs_abcde['ic_95_inf']:+.4f}, {boot_vs_abcde['ic_95_sup']:+.4f}], P={boot_vs_abcde['p_pos']:.1f}%)")

    print("\nDesglose por Patología (Owen 15 vs Candidato Ganador):")
    print(f"{'Patología':<20} {'Owen 15':>10} {mejor_combo_name:>15} {'Delta':>10}")
    per_pat_owen = res_val_int["B0_Owen_15_Baseline"]["per_pat"]
    per_pat_best = res_val_int[mejor_combo_name]["per_pat"]
    ganancias = 0
    for pat in LABELS:
        d = per_pat_best[pat] - per_pat_owen[pat]
        if d >= 0:
            ganancias += 1
        print(f"{pat:<20} {per_pat_owen[pat]:>10.4f} {per_pat_best[pat]:>15.4f} {d:>+10.4f}")
    print(f"\nPatologías no degradadas: {ganancias}/12")

    # Guardar resumen JSON y tablas
    resumen_salida = {
        "parte_a_oof_4349": {
            "familias_individuales": {k: v["macro"] for k, v in res_oof_indiv.items()},
            "owen_baseline_1_3": macro_owen_oof,
            "top_5_grid_ponderacion": top_5_grid,
            "variantes_adicionales": {
                "eff_cnxhib_50_50": m_eff_cnxhib,
                "cuatro_familias": m_4fam,
                "cinco_familias": m_5fam,
            }
        },
        "parte_b_val_int_415": {
            k: {
                "macro_auc": v["macro_auc"],
                "delta_vs_owen": v["bootstrap_vs_owen"]["delta_puntual"],
                "ic_95_inf": v["bootstrap_vs_owen"]["ic_95_inf"],
                "ic_95_sup": v["bootstrap_vs_owen"]["ic_95_sup"],
                "p_pos": v["bootstrap_vs_owen"]["p_pos"],
                "per_pat": v["per_pat"],
            } for k, v in res_val_int.items()
        },
        "candidato_ganador": {
            "nombre": mejor_combo_name,
            "macro_auc": res_val_int[mejor_combo_name]["macro_auc"],
            "bootstrap_vs_owen": res_val_int[mejor_combo_name]["bootstrap_vs_owen"],
            "bootstrap_vs_abcd": boot_vs_abcd,
            "bootstrap_vs_abcde": boot_vs_abcde,
            "patologias_favorables": f"{ganancias}/12",
        }
    }

    out_json = args.out / "resumen_fusion_offline.json"
    with open(out_json, "w") as f:
        json.dump(resumen_salida, f, indent=2)
    print(f"\nResumen guardado exitosamente en: {out_json}")


if __name__ == "__main__":
    main()
