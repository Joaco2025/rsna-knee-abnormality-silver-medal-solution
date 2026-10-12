#!/usr/bin/env python
"""Evaluación comparativa pareada: ResNet34 vs ConvNeXt-Tiny vs Ensamble (50/50).

Evalúa los modelos sobre los mismos estudios de validación (val_int y val_oficial)
usando los tensores preprocesados existentes en modo solo lectura.
Calcula AUC macro, AUC por patología, bootstrap C.I. 95%, diferencias pareadas netas
y correlación inter-modelo para contrastar objetivamente contra el baseline.

Uso:
  python rsna/evaluar_ensamble.py \
      --data path/to/preproc/v1_train \
      --labels path/to/etiquetas.parquet \
      --ckpt-resnet path/to/resnet34/best_oficial.pt \
      --ckpt-convnext path/to/convnext/best_oficial.pt \
      --out path/to/output_dir
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modelo import LABELS, KneeNet, prepare  # noqa: E402


class Studies(Dataset):
    def __init__(self, table: pd.DataFrame, data: Path, crop: int):
        self.uids = table.StudyInstanceUID.tolist()
        self.t = np.nan_to_num(table[[f"t_{c}" for c in LABELS]].to_numpy(np.float32))
        self.w = table[[f"w_{c}" for c in LABELS]].to_numpy(np.float32)
        self.y = table[[f"y_{c}" for c in LABELS]].to_numpy(np.float32)
        self.data, self.crop = data, crop

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, i: int):
        arr = np.load(self.data / f"{self.uids[i]}.npy", mmap_mode="r")
        size = arr.shape[-1]
        r = c = (size - self.crop) // 2
        x = torch.from_numpy(np.ascontiguousarray(arr[:, :, r:r + self.crop, c:c + self.crop]))
        present = (x.flatten(1).amax(dim=1) > 0).float()
        return x, present, torch.tensor(self.t[i]), torch.tensor(self.w[i]), torch.tensor(self.y[i])


@torch.no_grad()
def inferir_split(model, loader, device, n_slices: int, amp: bool):
    model.eval()
    ps = []
    for x, present, _, _, _ in loader:
        x = prepare(x.to(device, non_blocking=True), n_slices, train=False)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(x, present.to(device))
        ps.append(logits.float().sigmoid().cpu().numpy())
    return np.concatenate(ps)


def calcular_bootstrap_pareado(y: np.ndarray, p_res: np.ndarray, p_cnv: np.ndarray, p_ens: np.ndarray,
                               w: np.ndarray | None, n_boot: int = 500, seed: int = 42) -> dict:
    """Calcula intervalos de confianza al 95% y probabilidad de superioridad pareada."""
    n = len(y)
    rng = np.random.RandomState(seed)
    deltas_res, deltas_cnv = [], []
    boot_res, boot_cnv, boot_ens = [], [], []

    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        y_b, p_res_b, p_cnv_b, p_ens_b = y[idx], p_res[idx], p_cnv[idx], p_ens[idx]
        w_b = w[idx] if w is not None else None

        a_res, a_cnv, a_ens = [], [], []
        for j in range(len(LABELS)):
            m = np.ones(n, bool) if w_b is None else w_b[:, j] > 0
            yy = y_b[m, j]
            if 0 < yy.sum() < len(yy):
                a_res.append(roc_auc_score(yy, p_res_b[m, j]))
                a_cnv.append(roc_auc_score(yy, p_cnv_b[m, j]))
                a_ens.append(roc_auc_score(yy, p_ens_b[m, j]))

        if a_res and a_cnv and a_ens:
            m_res, m_cnv, m_ens = float(np.mean(a_res)), float(np.mean(a_cnv)), float(np.mean(a_ens))
            boot_res.append(m_res)
            boot_cnv.append(m_cnv)
            boot_ens.append(m_ens)
            deltas_res.append(m_ens - m_res)
            deltas_cnv.append(m_ens - m_cnv)

    deltas_res = np.array(deltas_res)
    deltas_cnv = np.array(deltas_cnv)

    return {
        "n_iteraciones": len(deltas_res),
        "ci95_resnet34": [round(float(np.percentile(boot_res, 2.5)), 4), round(float(np.percentile(boot_res, 97.5)), 4)],
        "ci95_convnext": [round(float(np.percentile(boot_cnv, 2.5)), 4), round(float(np.percentile(boot_cnv, 97.5)), 4)],
        "ci95_ensamble": [round(float(np.percentile(boot_ens, 2.5)), 4), round(float(np.percentile(boot_ens, 97.5)), 4)],
        "delta_vs_resnet_ci95": [round(float(np.percentile(deltas_res, 2.5)), 4), round(float(np.percentile(deltas_res, 97.5)), 4)],
        "prob_mejora_vs_resnet": round(float((deltas_res > 0).mean()), 4),
        "delta_vs_convnext_ci95": [round(float(np.percentile(deltas_cnv, 2.5)), 4), round(float(np.percentile(deltas_cnv, 97.5)), 4)],
        "prob_mejora_vs_convnext": round(float((deltas_cnv > 0).mean()), 4),
    }


def calcular_metricas(y: np.ndarray, p1: np.ndarray, p2: np.ndarray, pens: np.ndarray,
                       w: np.ndarray | None, split_name: str) -> dict:
    detalles = []
    aucs_m1, aucs_m2, aucs_ens = [], [], []

    for j, c in enumerate(LABELS):
        m = np.ones(len(y), bool) if w is None else w[:, j] > 0
        total_evaluable = int(m.sum())
        yy = y[m, j]
        pos = int(yy.sum())
        neg = total_evaluable - pos

        suficiente = pos >= 5 and neg >= 5
        evaluable_auc = pos > 0 and neg > 0

        # Correlación entre predicciones de ambos modelos
        if total_evaluable > 1:
            p1_sub, p2_sub = p1[m, j], p2[m, j]
            std1, std2 = float(np.std(p1_sub)), float(np.std(p2_sub))
            if std1 > 1e-7 and std2 > 1e-7:
                corr_pearson = float(np.corrcoef(p1_sub, p2_sub)[0, 1])
            else:
                corr_pearson = float("nan")
        else:
            corr_pearson = float("nan")

        if evaluable_auc:
            auc1 = float(roc_auc_score(yy, p1[m, j]))
            auc2 = float(roc_auc_score(yy, p2[m, j]))
            auc_e = float(roc_auc_score(yy, pens[m, j]))
            delta_res = auc_e - auc1
            delta_max = auc_e - max(auc1, auc2)
            aucs_m1.append(auc1)
            aucs_m2.append(auc2)
            aucs_ens.append(auc_e)
        else:
            auc1, auc2, auc_e, delta_res, delta_max = float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

        estado = "OK"
        if not evaluable_auc:
            estado = "SIN_CLASE"
        elif not suficiente:
            estado = "BAJO_SOPORTE"

        detalles.append({
            "hallazgo": c,
            "total_evaluable": total_evaluable,
            "positivos": pos,
            "negativos": neg,
            "prevalencia": round(pos / max(total_evaluable, 1), 4),
            "estado_soporte": estado,
            "auc_resnet34": round(auc1, 4) if not np.isnan(auc1) else None,
            "auc_convnext": round(auc2, 4) if not np.isnan(auc2) else None,
            "auc_ensamble": round(auc_e, 4) if not np.isnan(auc_e) else None,
            "delta_vs_resnet34": round(delta_res, 4) if not np.isnan(delta_res) else None,
            "delta_vs_max": round(delta_max, 4) if not np.isnan(delta_max) else None,
            "correlacion_r": round(corr_pearson, 4) if not np.isnan(corr_pearson) else None,
        })

    macro_m1 = float(np.nanmean(aucs_m1)) if aucs_m1 else float("nan")
    macro_m2 = float(np.nanmean(aucs_m2)) if aucs_m2 else float("nan")
    macro_ens = float(np.nanmean(aucs_ens)) if aucs_ens else float("nan")
    delta_macro_res = macro_ens - macro_m1 if not np.isnan(macro_ens) else float("nan")
    delta_macro_max = macro_ens - max(macro_m1, macro_m2) if not np.isnan(macro_ens) else float("nan")

    # Bootstrap pareado
    boot = calcular_bootstrap_pareado(y, p1, p2, pens, w)

    return {
        "split": split_name,
        "n_estudios": len(y),
        "macro_auc_resnet34": round(macro_m1, 4),
        "macro_auc_convnext": round(macro_m2, 4),
        "macro_auc_ensamble": round(macro_ens, 4),
        "delta_macro_vs_resnet34": round(delta_macro_res, 4),
        "delta_macro_vs_max": round(delta_macro_max, 4),
        "bootstrap": boot,
        "hallazgos": detalles,
    }


def imprimir_tabla_metricas(res: dict, titulo: str, advertencia: str | None = None):
    print(f"\n{'=' * 105}")
    print(f" {titulo} (N = {res['n_estudios']} estudios)")
    if advertencia:
        print(f" [!] NOTA METODOLÓGICA: {advertencia}")
    print(f"{'=' * 105}")
    print(f"{'Hallazgo':<18}{'Pos/Tot':>10}{'Prev%':>7}{'ResNet34':>10}{'ConvNeXt':>10}{'Ensamble':>10}{'Δ vs Res':>9}{'Corr(r)':>9}{'Soporte':>13}")
    print(f"{'-' * 105}")
    for d in res["hallazgos"]:
        p_tot = f"{d['positivos']}/{d['total_evaluable']}"
        prev = f"{d['prevalencia']*100:.1f}%"
        a1 = f"{d['auc_resnet34']:.4f}" if d['auc_resnet34'] is not None else "N/A"
        a2 = f"{d['auc_convnext']:.4f}" if d['auc_convnext'] is not None else "N/A"
        ae = f"{d['auc_ensamble']:.4f}" if d['auc_ensamble'] is not None else "N/A"
        dr = f"{d['delta_vs_resnet34']:+.4f}" if d['delta_vs_resnet34'] is not None else "N/A"
        cr = f"{d['correlacion_r']:.3f}" if d['correlacion_r'] is not None else "N/A"
        print(f"{d['hallazgo']:<18}{p_tot:>10}{prev:>7}{a1:>10}{a2:>10}{ae:>10}{dr:>9}{cr:>9}{d['estado_soporte']:>13}")
    print(f"{'-' * 105}")
    m1 = f"{res['macro_auc_resnet34']:.4f}"
    m2 = f"{res['macro_auc_convnext']:.4f}"
    me = f"{res['macro_auc_ensamble']:.4f}"
    dmr = f"{res['delta_macro_vs_resnet34']:+.4f}"
    print(f"{'MACRO AUC':<18}{'':>10}{'':>7}{m1:>10}{m2:>10}{me:>10}{dmr:>9}{'':>9}{'':>13}")
    b = res["bootstrap"]
    print(f"  -> Bootstrap 95% CI: ResNet34={b['ci95_resnet34']} | ConvNeXt={b['ci95_convnext']} | Ensamble={b['ci95_ensamble']}")
    print(f"  -> Δ(Ensamble - ResNet34) 95% CI: {b['delta_vs_resnet_ci95']} | P(Δ > 0) = {b['prob_mejora_vs_resnet']*100:.1f}%")
    print(f"{'=' * 105}\n")


def cargar_modelo(path: Path, device: torch.device):
    state = torch.load(path, map_location="cpu")
    bb = state["backbone"]
    assert bb in ("resnet34", "convnext_tiny", "efficientnet_v2_s"), f"Backbone no contemplado: {bb}"
    assert state.get("labels") == LABELS, f"Etiquetas permutadas en {path.name}"
    model = KneeNet(bb, None)
    model.load_state_dict(state["model"])
    model = model.to(device).eval().to(memory_format=torch.channels_last)
    return model, state


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="Ruta al directorio de tensores preprocesados")
    ap.add_argument("--labels", type=Path, required=True, help="Ruta al archivo parquet de etiquetas")
    ap.add_argument("--ckpt-resnet", type=Path, required=True, help="Ruta al checkpoint de ResNet-34 (.pt)")
    ap.add_argument("--ckpt-convnext", type=Path, required=True, help="Ruta al checkpoint de ConvNeXt-Tiny (.pt)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--weight-resnet", type=float, default=0.5)
    ap.add_argument("--weight-convnext", type=float, default=0.5)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (device.type == "cuda") and (not args.no_amp)

    print(f"=== EVALUACIÓN COMPARATIVA PAREADA ===")
    print(f"Dispositivo: {device} (AMP: {amp})")
    print(f"Datos: {args.data}")
    print(f"Etiquetas: {args.labels}")
    print(f"Salida: {args.out}")
    print(f"Ponderación ensamble: ResNet34={args.weight_resnet:.2f}, ConvNeXt={args.weight_convnext:.2f}")

    # Cargar modelos y verificar compatibilidad
    m_res, st_res = cargar_modelo(args.ckpt_resnet, device)
    m_cnv, st_cnv = cargar_modelo(args.ckpt_convnext, device)

    crop = st_res["crop"]
    assert st_cnv["crop"] == crop, f"Recortes incompatibles: {crop} vs {st_cnv['crop']}"
    assert st_res["slices"] == st_cnv["slices"], f"Cortes incompatibles: {st_res['slices']} vs {st_cnv['slices']}"
    slices = st_res["slices"]

    print(f"Modelo 1: {args.ckpt_resnet.name} ({st_res['backbone']}, época {st_res.get('epoch')}, cortes={slices}, recorte={crop})")
    print(f"Modelo 2: {args.ckpt_convnext.name} ({st_cnv['backbone']}, época {st_cnv.get('epoch')}, cortes={slices}, recorte={crop})")

    # Cargar y particionar tabla
    table = pd.read_parquet(args.labels)
    mask_exists = [(args.data / f"{u}.npy").exists() for u in table.StudyInstanceUID]
    table = table[mask_exists].reset_index(drop=True)

    official = table[table.oficial].reset_index(drop=True)
    rest = table[~table.oficial].reset_index(drop=True)
    fold = rest.StudyInstanceUID.map(lambda u: int(hashlib.md5(u.encode()).hexdigest(), 16) % 10)
    val_int = rest[fold == 0].reset_index(drop=True)

    print(f"\nParticiones de validación:")
    print(f"  val_int (Fold 0, NUNCA visto en entrenamiento de pesos): {len(val_int)} estudios")
    print(f"  val_oficial (58 estudios Gold, usados en calibración de pseudo-etiquetas): {len(official)} estudios")

    # Evaluaciones
    t0 = time.time()
    resultados_globales = {
        "metadata": {
            "fecha": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "device": str(device),
            "ckpt_resnet34": str(args.ckpt_resnet),
            "ckpt_convnext": str(args.ckpt_convnext),
            "peso_resnet34": args.weight_resnet,
            "peso_convnext": args.weight_convnext,
            "crop": crop,
            "slices": slices,
        }
    }

    w_total = args.weight_resnet + args.weight_convnext
    w1 = args.weight_resnet / w_total
    w2 = args.weight_convnext / w_total

    # --- 1. val_int ---
    loader_int = DataLoader(Studies(val_int, args.data, crop), batch_size=args.batch,
                            num_workers=args.workers, pin_memory=(device.type == "cuda"))
    print("\nInferiendo sobre val_int...")
    p1_int = inferir_split(m_res, loader_int, device, slices, amp)
    p2_int = inferir_split(m_cnv, loader_int, device, slices, amp)
    pens_int = w1 * p1_int + w2 * p2_int

    # Ground truth y máscara para val_int
    t_int = np.nan_to_num(val_int[[f"t_{c}" for c in LABELS]].to_numpy(np.float32))
    w_mask_int = val_int[[f"w_{c}" for c in LABELS]].to_numpy(np.float32)
    y_int = (t_int >= 0.5).astype(int)

    res_int = calcular_metricas(y_int, p1_int, p2_int, pens_int, w_mask_int, "val_int")
    resultados_globales["val_int"] = res_int
    imprimir_tabla_metricas(res_int, "VALIDACIÓN INTERNA (val_int, Fold 0)",
                            advertencia="Pseudo-etiquetas derivadas de reportes con Ollama (t >= 0.5 con w > 0). No sustituye el test oficial.")

    # Guardar predicciones val_int
    df_p_int = pd.DataFrame({"StudyInstanceUID": val_int.StudyInstanceUID})
    for j, c in enumerate(LABELS):
        df_p_int[f"target_{c}"] = y_int[:, j]
        df_p_int[f"weight_{c}"] = w_mask_int[:, j]
        df_p_int[f"p_resnet_{c}"] = p1_int[:, j]
        df_p_int[f"p_convnext_{c}"] = p2_int[:, j]
        df_p_int[f"p_ens_{c}"] = pens_int[:, j]
    df_p_int.to_csv(args.out / "predicciones_val_int.csv", index=False)

    # --- 2. val_oficial ---
    loader_off = DataLoader(Studies(official, args.data, crop), batch_size=args.batch,
                            num_workers=args.workers, pin_memory=(device.type == "cuda"))
    print("Inferiendo sobre val_oficial...")
    p1_off = inferir_split(m_res, loader_off, device, slices, amp)
    p2_off = inferir_split(m_cnv, loader_off, device, slices, amp)
    pens_off = w1 * p1_off + w2 * p2_off

    y_off = official[[f"y_{c}" for c in LABELS]].to_numpy(np.float32).astype(int)
    res_off = calcular_metricas(y_off, p1_off, p2_off, pens_off, None, "val_oficial")
    resultados_globales["val_oficial"] = res_off
    imprimir_tabla_metricas(res_off, "VALIDACIÓN OFICIAL (58 estudios Gold)",
                            advertencia="N=58 muy pequeño y usado en la calibración inicial. Útil solo como chequeo de coherencia histórica.")

    # Guardar predicciones val_oficial
    df_p_off = pd.DataFrame({"StudyInstanceUID": official.StudyInstanceUID})
    for j, c in enumerate(LABELS):
        df_p_off[f"gold_{c}"] = y_off[:, j]
        df_p_off[f"p_resnet_{c}"] = p1_off[:, j]
        df_p_off[f"p_convnext_{c}"] = p2_off[:, j]
        df_p_off[f"p_ens_{c}"] = pens_off[:, j]
    df_p_off.to_csv(args.out / "predicciones_val_oficial.csv", index=False)

    # Guardar resumen JSON
    (args.out / "resumen_evaluacion.json").write_text(json.dumps(resultados_globales, indent=2))

    total_s = time.time() - t0
    print(f"Evaluación completada en {total_s:.1f} s.")
    print(f"Archivos guardados en: {args.out}")


if __name__ == "__main__":
    main()
