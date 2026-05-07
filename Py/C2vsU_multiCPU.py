"""
Autor: Santiago Décima

DESCRIPCIÓN:
Este software está diseñado para el procesamiento, análisis y ajuste estadístico de datos experimentales de inversa 
cuadratica de capacitancia total frente al potencial aplicado (C_total⁻² vs U). El programa automatiza la 
extracción de parámetros físicos fundamentales de la interfase semiconductor tipo n/electrolito.

OPTIMIZACIÓN:
- Paralelización masiva mediante multiprocessing (Pool de procesos).
- Vectorización avanzada con NumPy (broadcasting 4D) para minimizar el uso de loops en Python.
- Precomputación de bases funcionales (sqrt y cosh) para evitar cálculos redundantes.
- Gestión eficiente de memoria mediante chunking en el espacio de búsqueda.
"""

import numpy as np
import multiprocessing as mp
import os
import sys
import time
import datetime
import csv
from config import ARCHIVOS, LABELS, Cfg

# ==========================================
# ESTRUCTURAS DE DATOS Y SOBRECARGA
# ==========================================
class ParamOverride:
    def __init__(self):
        self.value = 0.0
        self.fixed = False
        self.active = False

class FileOverrides:
    def __init__(self):
        self.P = [ParamOverride() for _ in range(5)]
        self.margen = -1.0
        self.N = -1

g_overrides = [FileOverrides() for _ in range(len(ARCHIVOS))]
g_global = FileOverrides()

def parse_args():
    global g_global
    for arg in sys.argv[1:]:
        if arg.startswith("--margen_iter="):
            g_global.margen = float(arg.split("=")[1])
        elif arg.startswith("--max_iter="):
            g_global.N = int(arg.split("=")[1])
        elif "--" in arg:
            file_idx = -1
            pos = arg.find("--")
            if pos > 0:
                try: file_idx = int(arg[:pos]) - 1
                except: file_idx = -1
            
            param_part = arg[pos:]
            if param_part.startswith("--"):
                p_char = param_part[2]
                p_idx = -1
                if p_char == 'A': p_idx = 0
                elif p_char == 'b': p_idx = 1
                elif p_char == 'C': p_idx = 2
                elif p_char == 'd': p_idx = 3
                elif p_char == 'E': p_idx = 4
                
                if p_idx != -1:
                    if "=" in param_part:
                        eq_pos = param_part.find("=")
                        fixed = param_part[eq_pos+1] == '='
                        val = float(param_part[eq_pos + (2 if fixed else 1):])
                        target = g_overrides[file_idx] if (0 <= file_idx < len(ARCHIVOS)) else g_global
                        target.P[p_idx].value = val
                        target.P[p_idx].fixed = fixed
                        target.P[p_idx].active = True

def cargar_datos(path):
    try:
        if not os.path.exists(path):
            return None, None
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            header = f.readline()
            sep = '\t'
            if ';' in header: sep = ';'
            elif ',' in header: sep = ','
            h_parts = header.strip().split(sep)
            ix, iy = 0, 1
            for i, col in enumerate(h_parts):
                col_lower = col.lower()
                # Detección flexible de X (Potencial)
                if any(k in col_lower for k in ['potential', 'voltage', 'v_dc', 'potencial', 'voltaje']):
                    ix = i
                # Detección flexible de Y (C^-2 o Impedancia)
                if any(k in col_lower for k in ['wz', 'ωz', 'c-2', 'c^-2', 'c**-2', '1/c^2', 'capacitance', 'c-²', 'c⁻²']):
                    iy = i
            x_vals, y_vals = [], []
            for line in f:
                parts = line.strip().split(sep)
                if len(parts) > max(ix, iy):
                    try:
                        xv = float(parts[ix].replace(',', '.'))
                        yv = float(parts[iy].replace(',', '.'))
                        x_vals.append(xv)
                        y_vals.append(yv)
                    except: continue
            return np.array(x_vals, dtype=np.float32), np.array(y_vals, dtype=np.float32)
    except Exception:
        return None, None

def estimar_iniciales(x, y):
    n = len(x)
    sx, sy = np.sum(x), np.sum(y)
    sxy, sx2 = np.sum(x*y), np.sum(x*x)
    m = (n*sxy - sx*sy) / (n*sx2 - sx*sx)
    c = (sy - m*sx) / n
    A_init = np.sqrt(max(m, 1e-9))
    b_init = -c / max(m, 1e-9)
    id_max = np.argmax(y)
    d_init = x[id_max]
    Y1 = np.sqrt(max(y[id_max], 1e-9))
    Z1 = Y1 - A_init * np.sqrt(max(x[id_max] - b_init, 1e-9))
    ifar = 0 if id_max > n//2 else n-1
    Y2 = np.sqrt(max(y[ifar], 1e-9))
    Z2 = Y2 - A_init * np.sqrt(max(x[ifar] - b_init, 1e-9))
    f2 = 1.0 / np.cosh(np.clip(Cfg.n_fijo * (x[ifar] - d_init), -20, 20))
    dn = 1.0 - f2
    if abs(dn) < 1e-5: dn = 1e-5
    C_init = (Z1 - Z2) / dn
    E_init = Z1 - C_init
    if E_init < 100.0: E_init = 100.0
    return [A_init, b_init, C_init, d_init, E_init]

# ==========================================
# MOTOR DE CÁLCULO (Worker Process)
# ==========================================
def worker_search(args):
    """Procesa un rango de tareas con vectorización máxima"""
    task_indices, x_data, y_data, N, As, Ast, T1_base, T2_base, C_vals, E_vals = args
    best_mm = 1e30
    best_P = [0.0]*5
    
    y_target = y_data[np.newaxis, np.newaxis, np.newaxis, :]
    C_chunk_size = 50
    
    for iA, ib in task_indices:
        A = As[0] + iA * Ast[0]
        t1 = (A * T1_base[ib])[np.newaxis, np.newaxis, np.newaxis, :]
        
        for iC_start in range(0, N, C_chunk_size):
            iC_end = min(iC_start + C_chunk_size, N)
            C_sub = C_vals[iC_start:iC_end, np.newaxis, np.newaxis, np.newaxis]
            
            Y_noE = t1 + C_sub * T2_base[np.newaxis, :, np.newaxis, :]
            
            # Cálculo de Minimax Error
            error = np.max(np.abs((Y_noE + E_vals[np.newaxis, np.newaxis, :, np.newaxis])**2 - y_target), axis=-1)
            
            min_val = np.min(error)
            if min_val < best_mm:
                best_mm = min_val
                idx = np.unravel_index(np.argmin(error), error.shape)
                best_P = [A, As[1] + ib * Ast[1], C_vals[iC_start + idx[0]], As[3] + idx[1] * Ast[3], E_vals[idx[2]]]
                
    return (best_mm, best_P)

def print_progress(current, total, prefix='', length=40):
    percent = (current / total)
    filled = int(length * percent)
    bar = '█' * filled + '-' * (length - filled)
    sys.stdout.write(f'\r   -> {prefix}: [{bar}] {percent*100:3.1f}%')
    sys.stdout.flush()

def solve_file(idx, path, label, ov, g_ov, ts, num_files, logger):
    logger.log(f"[{idx+1}/{num_files}] Procesando: {path}")
    x_data, y_data = cargar_datos(path)
    if x_data is None or len(x_data) == 0: 
        logger.log(f"   [ERROR] No se pudieron cargar los datos de {path}")
        return None

    logger.log(f"   -> {len(x_data)} puntos cargados.")
    
    margen = ov.margen if ov.margen > 0 else (g_ov.margen if g_ov.margen > 0 else Cfg.margen)
    N = ov.N if ov.N > 0 else (g_ov.N if g_ov.N > 0 else Cfg.N)
    
    logger.log(f"   -> Etapa 1: Búsqueda Global 5D (N={N}, margen={margen:f})")
    
    P_start = estimar_iniciales(x_data, y_data)
    As, Ast = np.zeros(5), np.zeros(5)
    is_fixed = [False]*5
    for k in range(5):
        po = ov.P[k] if ov.P[k].active else g_ov.P[k]
        center = po.value if po.active else P_start[k]
        is_fixed[k] = (po.active and po.fixed)
        if is_fixed[k]:
            As[k], Ast[k] = center, 0.0
        else:
            range_base = 1.0 if abs(center) < 1e-9 else abs(center)
            As[k] = center - (range_base * margen)
            Ast[k] = (range_base * 2.0 * margen) / max(N - 1, 1)

    logger.log("   Configuración de búsqueda:")
    p_names = ["A", "b", "C", "d", "E"]
    for k in range(5):
        logger.log(f"     - {p_names[k]}: [{As[k]:.6g} a {As[k] + (N-1)*Ast[k]:.6g}] paso={Ast[k]:.6g}")

    # Precomputación
    b_vals = As[1] + np.arange(N) * Ast[1]
    d_vals = As[3] + np.arange(N) * Ast[3]
    C_vals = (As[2] + np.arange(N) * Ast[2]).astype(np.float32)
    E_vals = (As[4] + np.arange(N) * Ast[4]).astype(np.float32)
    E_vals = np.maximum(E_vals, 1e-5)
    
    T1_base = np.sqrt(np.maximum(x_data - b_vals[:, np.newaxis], 1e-9)).astype(np.float32)
    T2_base = (1.0 / np.cosh(np.clip(Cfg.n_fijo * (x_data - d_vals[:, np.newaxis]), -80, 80))).astype(np.float32)

    num_cpus = mp.cpu_count()
    all_tasks = [(iA, ib) for iA in range(N) for ib in range(N)]
    num_tasks = len(all_tasks)
    chunk_size = max(num_tasks // (num_cpus * 4), 1)
    task_chunks = [all_tasks[i:i + chunk_size] for i in range(0, num_tasks, chunk_size)]
    
    worker_args = [(chunk, x_data, y_data, N, As, Ast, T1_base, T2_base, C_vals, E_vals) for chunk in task_chunks]
    
    best_mm = 1e30
    best_P = [0.0]*5
    
    start_time = time.time()
    completed = 0
    with mp.Pool(processes=num_cpus) as pool:
        it = pool.imap_unordered(worker_search, worker_args)
        for res_mm, res_P in it:
            completed += 1
            if completed % 10 == 0 or completed == len(task_chunks):
                print_progress(completed, len(task_chunks), prefix='Búsqueda', length=40)
            if res_mm < best_mm:
                best_mm, best_P = res_mm, res_P
    
    print() 
    total_file_time = time.time() - start_time
    logger.log(f"   Ajuste finalizado. Minimax Error: {best_mm:f}")
    
    A, b, C, d, E = best_P
    step_f = margen / max(N - 1, 1)
    eA, eb, eC, ed, eE = [(0.0 if is_fixed[k] else abs(best_P[k]*step_f)) for k in range(5)]
    
    # Propiedades físicas
    N_D = 2.0 / (Cfg.eps_sc * Cfg.eps_0 * Cfg.e_0 * A**2)
    U_fb = b - (Cfg.k_B * Cfg.T) / Cfg.e_0
    C_0 = (Cfg.k_B * Cfg.T) / (2.0 * Cfg.z**2 * Cfg.e_0**2 * Cfg.eps_ele * Cfg.eps_0 * C**2)
    x_H = max(Cfg.eps_ele * Cfg.eps_0 * E, 1e-15)
    M_0 = (C_0 / Cfg.N_A) * 1000.0
    
    Y_final = (A * np.sqrt(np.maximum(x_data - b, 1e-9)) + C / np.cosh(np.clip(Cfg.n_fijo*(x_data - d), -80, 80)) + E)**2
    rmse = np.sqrt(np.mean((Y_final - y_data)**2))

    # Errores para el log
    err_ND = abs(N_D * 2 * eA / max(abs(A), 1e-12))
    err_Ufb = eb + Cfg.err_rel_x * abs(b)
    err_C0 = abs(2 * C_0 * eC / max(abs(C), 1e-12))
    err_Uz = ed + Cfg.err_rel_x * abs(d)
    err_xH = abs(x_H * eE / max(abs(E), 1e-12)) * 1e7 # nm
    err_M0 = (err_C0 / Cfg.N_A) * 1000.0

    logger.log(f"\n==================================================")
    logger.log(f" RESULTADOS FINALES - {label}")
    logger.log(f"==================================================")
    logger.log(f"   Minimax Error: {best_mm:.6e}   RMSE: {rmse:.6e}")
    logger.log(f"   N_D = {N_D:.6e} ± {err_ND:.6e} cm⁻³")
    logger.log(f"  U_fb = {U_fb:.6e} ± {err_Ufb:.6e} V")
    logger.log(f"   C_0 = {C_0:.6e} ± {err_C0:.6e} cm⁻³")
    logger.log(f"   U_z = {d:.6e} ± {err_Uz:.6e} V")
    logger.log(f"   x_H = {x_H*1e7:.6e} ± {err_xH:.6e} nm")
    logger.log(f"   M_0 = {M_0:.6e} ± {err_M0:.6e} mol/L\n")

    out_dir = os.path.dirname(path)
    for suffix, data in [("params", best_P), ("props", [N_D, U_fb, C_0, d, x_H, M_0])]:
        fname = f"{out_dir}/{suffix}_cpu_{ts}.csv"
        with open(fname, "w") as f:
            f.write(f"# margen={margen}, N={N}\n")
            if suffix == "params":
                f.write("Parametro,Valor,±Error\n")
                for k in range(5): f.write(f"{p_names[k]},{best_P[k]:.6e},±{eA if k==0 else eb if k==1 else eC if k==2 else ed if k==3 else eE:.6e}\n")
            else:
                f.write("Propiedad,Valor,±Error\n")
                f.write(f"N_D,{N_D:.6e},±{err_ND:.6e}\n")
                f.write(f"U_fb,{U_fb:.6e},±{err_Ufb:.6e}\n")
                f.write(f"C_0,{C_0:.6e},±{err_C0:.6e}\n")
                f.write(f"U_z,{d:.6e},±{err_Uz:.6e}\n")
                f.write(f"x_H,{x_H*1e7:.6e},±{err_xH:.6e}\n")
                f.write(f"M_0,{M_0:.6e},±{err_M0:.6e}\n")

    return { 'path': path, 'label': label, 'best_P': list(best_P), 'time': total_file_time }

class Logger:
    def __init__(self, filename):
        self.filename = filename
        with open(self.filename, "w") as f: pass
    def log(self, msg):
        print(msg)
        with open(self.filename, "a") as f: f.write(msg + "\n")

def main():
    parse_args()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_f = f"log_multiCPU_{ts}.txt"
    logger = Logger(log_f)

    logger.log("**************************************************")
    logger.log(" INICIO - PROCESAMIENTO C⁻² vs U (CPU Global Search)")
    logger.log(" Autor: Santiago Décima")
    logger.log("**************************************************")
    
    logger.log(f"Archivos a analizar: {len(ARCHIVOS)}")
    m_glob = g_global.margen if g_global.margen > 0 else Cfg.margen
    n_glob = g_global.N if g_global.N > 0 else Cfg.N
    logger.log(f"Margen Global: {m_glob}   N Global: {n_glob}")
    
    ov_str = []
    p_names = ["A", "b", "C", "d", "E"]
    for k in range(5):
        if g_global.P[k].active:
            ov_str.append(f"{p_names[k]}={g_global.P[k].value}")
    if ov_str:
        logger.log(f"Overrides Globales: {' '.join(ov_str)}")
    logger.log("")

    valid_results = []
    total_start = time.time()
    for i in range(len(ARCHIVOS)):
        res = solve_file(i, ARCHIVOS[i], LABELS[i], g_overrides[i], g_global, ts, len(ARCHIVOS), logger)
        if res: valid_results.append(res)

    if valid_results:
        with open("datos_grafico.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Archivo", "Label", "A", "b", "C", "d", "E"])
            for r in valid_results: writer.writerow([r['path'], r['label']] + r['best_P'])
        
        logger.log(f"Generando gráficos finales...")
        os.system(f"python3 graficador_c2vsu.py ajuste_multiCPU_{ts}.png")
    
    logger.log(f"PROCESAMIENTO FINALIZADO. Tiempo total: {time.time()-total_start:.1f}s")

if __name__ == "__main__":
    main()
