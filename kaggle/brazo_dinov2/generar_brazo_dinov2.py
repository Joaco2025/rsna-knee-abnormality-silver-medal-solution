#!/usr/bin/env python
"""Genera script_dinov2.py y las celdas para pegar del brazo DINOv2 a partir de los notebooks de referencia.

Nada de la lógica de DINOv2 se reescribe: el script se arma con el TEXTO de nodos de primer nivel de las celdas
3-14 del notebook 0.940 (referencias/bend-the-knee-to-speedy-raptors-the-original.ipynb), elegidos con `ast`
y copiados por líneas, sin pasar por ast.unparse. Lo único que se añade es el pegamento de `PEGAMENTO`
(argumentos, GPU, ids de test, informe) y UNA sustitución de ruta declarada en SUSTITUCIONES.

Uso (desde cualquier directorio, con cualquier python 3.9+):
    python generar_brazo_dinov2.py            # escribe script_dinov2.py y las dos celdas
    python generar_brazo_dinov2.py --comprobar  # no escribe: sale con 1 si los archivos no coinciden con lo generado
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NB_940 = REPO / "referencias" / "bend-the-knee-to-speedy-raptors-the-original.ipynb"
PLANTILLA = HERE / "celda_brazo_dinov2.py"
SCRIPT = HERE / "script_dinov2.py"
CELDA = HERE / "celda_para_pegar_0957.py"
CELDA_DIAG = HERE / "celda_para_pegar_0957_diag.py"
MARCA_SCRIPT = "# @@SCRIPT_DINOV2@@"
CABECERA_DIAG = (
    "# VARIANTE DIAGNÓSTICA: igual que celda_para_pegar_0957.py, con la primera línea añadida. Si el brazo se omite o falla,\n"
    "# esta celda lanza una excepción y el rerun termina en error. Solo para UN envío que responda \"¿cupo el brazo?\".\n"
    "# Nunca como final. Se regenera con generar_brazo_dinov2.py.\n"
    "BRAZO_DINOV2_EXIGIR = True\n"
)

# Qué se toma de cada celda del 0.940 (índices de celda del notebook). "todo" = todos los nodos de primer nivel que
# no sean `Expr`/`If`/`Try` (esos son llamadas y comprobaciones de arranque, no definiciones) y cuyos objetivos no
# estén en EXCLUIR_ASIGNACIONES. Con lista de nombres: solo esas funciones o clases, más las asignaciones de nombre.
IMPORTS = ("Import", "ImportFrom")
EXCLUIR_ASIGNACIONES = {"_STARTUP_T0", "_STARTUP_GPU_IMPORT_S", "DEVS", "_STARTUP_PATH_S", "ROOT", "N_GROUP", "CACHE_SLICES"}
SELECCION = {
    # (celda): (funciones/clases que se toman o None = todas, asignaciones: True/False)
    3: ({"_cuda_execution_probe", "rsna_sha", "rsna_json", "rsna_event", "rsna_finite", "rsna_rank01", "rsna_rankpct",
         "rsna_frame", "rsna_save_predictions", "rsna_strict_load", "rsna_deadline", "rsna_array", "rsna_release_pixels",
         "rsna_phase"}, True),
    4: ({"log", "_one_direct", "find_dinov2"}, True),
    5: (None, True),
    6: (None, True),
    7: (None, True),
    8: (None, True),
    9: (None, True),
    10: (None, True),
    11: (None, True),
    12: (None, True),
    13: (None, True),
    14: (None, True),
}
# Asignaciones de c04 que sí interesan (el resto, p. ej. ROOT = find_root(), N_GROUP = plan_cache(...), se excluye).
ASIGNACIONES_C04 = {"IMG"}      # DINOV2_SOURCE = find_dinov2('small') pasa al pegamento (se resuelve al ejecutar, no al importar)

# Única modificación de texto: el directorio de diagnósticos. En el notebook es /kaggle/working/diagnostics; aquí es
# una variable (DIAG_DIR, definida en el pegamento con ese mismo valor por defecto) para poder escribir bajo --out.
SUSTITUCIONES = [("'/kaggle/working/diagnostics'", "DIAG_DIR", 2)]

# Nombres que el notebook usa pero no define dentro de las celdas tomadas (el pegamento los define o no se alcanzan).
GLOBALES_ESPERADOS = {
    "DEVS": "lo define el pegamento (los dispositivos CUDA visibles que pasan la sonda de la celda 3)",
    "ROOT": "lo define el pegamento (carpeta de la competencia)",
    "TIME_BUDGET": "lo reajusta el pegamento (plazo recibido en --deadline)",
    "DIAG_DIR": "lo define el pegamento",
    "augment": "solo se llama con jitter=True; el notebook nunca lo activa (jitter es False en _run_shared_dino_group y en pop_next) y ni él define augment",
    "TARGETS": "definido en la celda 3 (se toma)",
}

PEGAMENTO = r'''

# =====================================================================================================
# PEGAMENTO (no extraído del notebook). Todo lo de arriba es texto del notebook 0.940; lo de abajo es nuevo.
# Hace lo que en el notebook hacían rsna_initialize, la comprobación de 2xT4 y la celda 15 (aquí sin borrar
# submission.csv ni exigir otros datasets), con argumentos de línea de órdenes.
# =====================================================================================================
DIAG_DIR = '/kaggle/working/diagnostics'


def _glue_args(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='Brazo DINOv2 x20 del notebook 0.940, como script independiente')
    ap.add_argument('--data', help='carpeta de la competencia (test.csv, test_series.csv, test_series/)')
    ap.add_argument('--out', help='carpeta de salida; el script cambia a ella y escribe dinov2_sub.csv e informe.json')
    ap.add_argument('--deadline', type=float, help='plazo duro, en segundos desde la época (time.time())')
    ap.add_argument('--comprobar', action='store_true', help='solo localiza pesos y modelo base y escribe un JSON; no infiere')
    return ap.parse_args(argv)


def _glue_comprobar(out):
    import importlib
    info = {'python': __import__('sys').version.split()[0], 'torch': str(torch.__version__)}
    for modulo in ('transformers', 'pydicom', 'numpy', 'pandas'):
        info[modulo] = str(importlib.import_module(modulo).__version__)
    info['dinov2_small'] = str(find_dinov2('small'))
    info['pesos'] = str(find_weights())
    info['miembros'] = len(json.loads((Path(info['pesos']) / 'manifest.json').read_text())['members'])
    info['gpus'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    Path(out, 'comprobacion.json').write_text(json.dumps(info, indent=1, sort_keys=True) + '\n')
    print('[dinov2] comprobacion OK', json.dumps(info, sort_keys=True), flush=True)


def _glue_main():
    global ROOT, DEVS, TIME_BUDGET, DIAG_DIR, DINOV2_SOURCE, _RSNA_TEST_IDS
    args = _glue_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)
    DIAG_DIR = str(out / 'diagnostics')
    Path(DIAG_DIR).mkdir(parents=True, exist_ok=True)
    if args.comprobar:
        _glue_comprobar(out)
        return 0
    if not (args.data and args.deadline):
        raise SystemExit('faltan --data y --deadline')
    ROOT = Path(args.data)
    DINOV2_SOURCE = find_dinov2('small')              # celda 4 del notebook: DINOV2_SOURCE = find_dinov2('small')
    test_ids = pd.read_csv(ROOT / 'test.csv', dtype={'StudyInstanceUID': str}).StudyInstanceUID.tolist()
    if not test_ids or len(set(test_ids)) != len(test_ids):
        raise RuntimeError('test.csv vacío o con UIDs repetidos')
    _RSNA_TEST_IDS = [str(u) for u in test_ids]
    DEVS = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count()) if _cuda_execution_probe(i)]
    if not DEVS:
        raise RuntimeError('ninguna GPU CUDA pasó la sonda de ejecución')
    TIME_BUDGET = float(args.deadline) - T0           # rsna_deadline compara contra T0 + TIME_BUDGET
    log(f'dinov2: {len(_RSNA_TEST_IDS)} estudios, {len(DEVS)} GPU, plazo en {args.deadline - time.time():.0f} s')
    # Lo que hace la celda 15 del notebook:
    rsna_phase('dinov2', 'START')
    main()
    log('done')
    _speed_dino_templates.clear()
    gc.collect()
    rsna_phase('dinov2', 'COMPLETE')
    if _DINOV2_MATCHED_MEMBERS != EXPECTED_DINOV2_MEMBERS:       # la compuerta de la celda 16
        raise RuntimeError('DINOv2 20/20 fingerprint gate did not pass')
    stage = out / '_pipeline_stage.csv'
    sub = _v37_validate_submission(stage, pd.read_csv(ROOT / 'test.csv'), 'dinov2 salida')
    os.replace(stage, out / 'dinov2_sub.csv')
    vacios = sorted({str(s) for e in _RSNA_AUDIT['events'] if e.get('kind') == 'empty_study_rows' for s in e.get('studies', [])})
    kinds = {}
    for e in _RSNA_AUDIT['events']:
        kinds[e['kind']] = kinds.get(e['kind'], 0) + 1
    informe = {'estudios': len(sub), 'miembros': _DINOV2_MATCHED_MEMBERS, 'gpus': len(DEVS),
               'segundos': round(time.time() - T0, 1), 'estudios_sin_series': vacios,
               'series_con_fallo_de_decodificacion': len(set(map(str, DECODE_FAILED))), 'eventos': kinds}
    (out / 'informe.json').write_text(json.dumps(informe, indent=1, sort_keys=True) + '\n')
    print('[dinov2] informe', json.dumps(informe, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(_glue_main())
'''


# ---------------------------------------------------------------------------------------------------
def sha(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def celdas_940() -> dict[int, str]:
    nb = json.loads(NB_940.read_text())
    return {i: "".join(c["source"]) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"}


def _nombres_objetivo(nodo) -> set[str]:
    objetivos = nodo.targets if isinstance(nodo, ast.Assign) else [nodo.target]
    return {t.id for t in objetivos if isinstance(t, ast.Name)}


def _solo_nombres(nodo) -> bool:
    objetivos = nodo.targets if isinstance(nodo, ast.Assign) else [nodo.target]
    return all(isinstance(t, ast.Name) for t in objetivos)


def tomar(celda: int, texto: str, funciones, asignaciones: bool) -> list[tuple[str, str]]:
    """Nodos tomados de una celda: lista de (descripción, texto exacto por líneas)."""
    lineas = texto.split("\n")
    out = []
    for nodo in ast.parse(texto).body:
        ini = min([nodo.lineno] + [d.lineno for d in getattr(nodo, "decorator_list", [])])
        fragmento = "\n".join(lineas[ini - 1:nodo.end_lineno])
        tipo = type(nodo).__name__
        if tipo in IMPORTS:
            out.append((f"import ({fragmento.strip()})", fragmento))
        elif isinstance(nodo, (ast.FunctionDef, ast.ClassDef)):
            if funciones is None or nodo.name in funciones:
                out.append((f"{'def' if isinstance(nodo, ast.FunctionDef) else 'class'} {nodo.name}", fragmento))
        elif isinstance(nodo, (ast.Assign, ast.AnnAssign)) and asignaciones and _solo_nombres(nodo):
            nombres = _nombres_objetivo(nodo)
            if nombres & EXCLUIR_ASIGNACIONES:
                continue
            if celda == 4 and not nombres <= ASIGNACIONES_C04:
                continue
            out.append((f"asignación {', '.join(sorted(nombres))}", fragmento))
    return out


def construir_script(verbose: bool = False) -> tuple[str, list[dict]]:
    celdas = celdas_940()
    partes, inventario = [], []
    for celda, (funciones, asignaciones) in SELECCION.items():
        cuerpo = tomar(celda, celdas[celda], funciones, asignaciones)
        inventario.append({"celda": celda, "sha256_celda": sha(celdas[celda]), "nodos": [d for d, _ in cuerpo]})
        partes.append(f"# ---- celda {celda} del notebook 0.940 " + "-" * 50)
        partes.extend(t for _, t in cuerpo)
    cuerpo = "\n".join(partes) + "\n"
    for viejo, nuevo, veces in SUSTITUCIONES:
        if cuerpo.count(viejo) != veces:
            raise AssertionError(f"sustitución {viejo!r}: {cuerpo.count(viejo)} apariciones, se esperaban {veces}")
        cuerpo = cuerpo.replace(viejo, nuevo)
    cabecera = (
        '"""Brazo DINOv2 x20 del notebook 0.940 (Bend the Knee to Speedy Raptors), como script independiente.\n\n'
        "GENERADO por generar_brazo_dinov2.py; no editar a mano. Las celdas 3-14 del notebook se copian por nodos de\n"
        "primer nivel (ast), sin reescribir su lógica; ver la lista en experiments/reportes/brazo_dinov2_plan.md.\n"
        f"Notebook de origen: {NB_940.name}, sha256 {sha(NB_940.read_text())}\n"
        "Corre en su propio proceso para no mezclar sus globales (T0, TIME_BUDGET, IMG, GROUP, SLOTS, ...) con los del 0.957.\n\n"
        "Uso: python script_dinov2.py --data DATA --out CARPETA --deadline EPOCH   (o --comprobar --out CARPETA)\n"
        'Salida: CARPETA/dinov2_sub.csv (StudyInstanceUID + 12 hallazgos en rango percentil) e informe.json."""\n'
        "from __future__ import annotations\n"
        "# La línea anterior es la primera del notebook (celda 3), que ya trae `from __future__ import annotations`.\n"
    )
    # `from __future__` debe ir al principio del archivo: la quitamos del cuerpo si aparece y dejamos la de la cabecera.
    cuerpo = cuerpo.replace("from __future__ import annotations\n", "", 1)
    script = cabecera + cuerpo + PEGAMENTO
    ast.parse(script)
    return script, inventario


def globales_sin_resolver(script: str) -> set[str]:
    """Nombres leídos que ni el módulo ni los builtins definen, ni son locales de la función que los lee."""
    arbol = ast.parse(script)
    definidos = set(dir(builtins))
    for nodo in ast.walk(arbol):
        if isinstance(nodo, (ast.FunctionDef, ast.ClassDef)):
            definidos.add(nodo.name)
        elif isinstance(nodo, ast.Global):
            definidos.update(nodo.names)
    for nodo in arbol.body:
        for n in ast.walk(nodo):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                definidos.update((a.asname or a.name).split(".")[0] for a in n.names)
        if isinstance(nodo, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.For, ast.With)):
            for n in ast.walk(nodo):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    definidos.add(n.id)
    sin = set()

    def propios(func):
        """Nombres que `func` liga en su propio ámbito (parámetros, asignaciones, defs, imports, except as)."""
        locales = {a.arg for a in ast.walk(func.args) if isinstance(a, ast.arg)}

        def visitar(n):
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                locales.add(n.id)
            elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                locales.add(n.name)
                return                                   # su interior es otro ámbito
            elif isinstance(n, ast.Lambda):
                return
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                locales.update((a.asname or a.name).split(".")[0] for a in n.names)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                locales.add(n.name)
            for hijo in ast.iter_child_nodes(n):
                visitar(hijo)

        for sentencia in func.body:
            visitar(sentencia)
        return locales

    def revisar(func, heredados):
        visibles = heredados | propios(func)

        def visitar(n, vis):
            if isinstance(n, ast.FunctionDef):
                revisar(n, vis)
                return
            if isinstance(n, ast.Lambda):
                vis = vis | {a.arg for a in n.args.args} | {m.id for m in ast.walk(n) if isinstance(m, ast.Name)
                                                              and isinstance(m.ctx, ast.Store)}
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in vis and n.id not in definidos:
                sin.add(n.id)
            for hijo in ast.iter_child_nodes(n):
                visitar(hijo, vis)

        for sentencia in func.body:
            visitar(sentencia, visibles)
        for d in func.args.defaults + [k for k in func.args.kw_defaults if k is not None]:
            visitar(d, heredados)

    for nodo in arbol.body:
        if isinstance(nodo, ast.FunctionDef):
            revisar(nodo, set())
        elif isinstance(nodo, ast.ClassDef):
            for sub in nodo.body:
                if isinstance(sub, ast.FunctionDef):
                    revisar(sub, set())
    return sin


def armar_celda(script: str, plantilla: str) -> str:
    if plantilla.count(MARCA_SCRIPT) != 1:
        raise AssertionError(f"la plantilla debe tener exactamente una línea {MARCA_SCRIPT}")
    if "'''" in script:
        raise AssertionError("el script contiene ''' y no se puede incrustar como r'''...'''")
    if script.endswith("\\"):
        raise AssertionError("el script termina en barra invertida")
    literal = (f"DV2_SCRIPT_SHA256 = {sha(script)!r}\n"
               "DV2_SCRIPT_TEXTO = r'''" + script + "'''")
    return plantilla.replace(MARCA_SCRIPT, literal)


def generar() -> dict[str, str]:
    script, inventario = construir_script()
    plantilla = PLANTILLA.read_text()
    celda = armar_celda(script, plantilla)
    return {str(SCRIPT): script, str(CELDA): celda, str(CELDA_DIAG): CABECERA_DIAG + celda,
            "_inventario": json.dumps(inventario, indent=1, ensure_ascii=False)}


def main(argv) -> int:
    salidas = generar()
    inventario = salidas.pop("_inventario")
    if "--comprobar" in argv:
        malos = [r for r, t in salidas.items() if not Path(r).is_file() or Path(r).read_text() != t]
        for r in malos:
            print("DIFIERE:", r)
        print("sincronizado" if not malos else f"{len(malos)} archivo(s) distintos de lo generado")
        return 1 if malos else 0
    for ruta, texto in salidas.items():
        Path(ruta).write_text(texto)
        print(f"{ruta}: {len(texto.splitlines())} líneas, sha256 {sha(texto)}")
    sin = globales_sin_resolver(salidas[str(SCRIPT)])
    print("globales sin resolver dentro de funciones:", sorted(sin))
    (HERE / "inventario_nodos.json").write_text(inventario + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
