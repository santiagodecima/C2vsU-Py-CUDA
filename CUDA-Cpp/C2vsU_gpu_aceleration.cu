/*
 * Autor: Santiago Décima
 *
 * DESCRIPCIÓN:
 * Versión GPU (CUDA C++) de C2vsU_multi-Ef.py optimizada (FP32).
 * Procesa datos experimentales de C⁻² vs U, realizando una búsqueda
 * en dos etapas: 5D Global -> 2D Refinamiento (C y E).
 *
 * COMPILACIÓN:
 *   nvcc -O3 C2vsU_gpu_aceleration.cu -o C2vsU_gpu_aceleration
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "config.h"

// ==========================================
// ESTRUCTURAS DE DATOS Y SOBRECARGA DE PARÁMETROS
// ==========================================
struct ParamOverride {
    float value = 0;
    bool fixed  = false; // true si es == (fijo), false si es = (centrado)
    bool active = false;
};

struct FileOverrides {
    ParamOverride P[5]; // A, b, C, d, E
    double margen = -1.0; 
    int N = -1;
};

static std::vector<FileOverrides> g_overrides;
static FileOverrides g_global;

static void parse_args(int argc, char** argv) {
    g_overrides.resize(ARCHIVOS.size());
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("--margen_iter=") == 0) {
            g_global.margen = std::stod(arg.substr(14));
        } else if (arg.find("--max_iter=") == 0) {
            g_global.N = std::stoi(arg.substr(11));
        } else {
            // Buscar si empieza con un número de archivo (ej: 1--A=...)
            int file_idx = -1;
            size_t pos = arg.find("--");
            if (pos != std::string::npos && pos > 0) {
                try {
                    file_idx = std::stoi(arg.substr(0, pos)) - 1;
                } catch (...) { file_idx = -1; }
            }
            
            std::string param_part = (file_idx == -1) ? arg : arg.substr(pos);
            if (param_part.find("--") == 0) {
                char pchar = param_part[2];
                int p_idx = -1;
                if (pchar == 'A') p_idx = 0;
                else if (pchar == 'b') p_idx = 1;
                else if (pchar == 'C') p_idx = 2;
                else if (pchar == 'd') p_idx = 3;
                else if (pchar == 'E') p_idx = 4;

                if (p_idx != -1) {
                    size_t eq_pos = param_part.find("=");
                    if (eq_pos != std::string::npos) {
                        bool fixed = (param_part[eq_pos + 1] == '=');
                        float val = std::stof(param_part.substr(eq_pos + (fixed ? 2 : 1)));
                        
                        if (file_idx >= 0 && (size_t)file_idx < ARCHIVOS.size()) {
                            g_overrides[file_idx].P[p_idx].value = val;
                            g_overrides[file_idx].P[p_idx].fixed = fixed;
                            g_overrides[file_idx].P[p_idx].active = true;
                        } else {
                            g_global.P[p_idx].value = val;
                            g_global.P[p_idx].fixed = fixed;
                            g_global.P[p_idx].active = true;
                        }
                    }
                }
            }
        }
    }
}

// ==========================================
// ESTRUCTURAS DE DATOS
// ==========================================
struct Props {
    double N_D, eN_D, U_fb, eU_fb, C_0, eC_0, U_z, eU_z, x_H, ex_H, M_0, eM_0;
};

struct Dataset {
    std::string path, label, dir;
    std::vector<float> x, y;
    float P[5] = {};   // A, b, C, d, E
    float mmDiff = 1e30f;
    Props  props  = {};
    double local_margen = 0;
    int local_N = 0;
    bool is_fixed[5] = {false, false, false, false, false};
};

static std::ofstream g_log;

template<typename T>
static void LOG(const T& msg) {
    std::cout << msg;
    if (g_log.is_open()) g_log << msg;
}
static void LOGf(const std::string& s) { LOG(s); }

static std::string timestamp() {
    auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::ostringstream ss;
    ss << std::put_time(std::localtime(&t), "%Y%m%d_%H%M%S");
    return ss.str();
}

static std::string dirof(const std::string& path) {
    auto p = path.find_last_of("/\\");
    return (p == std::string::npos) ? "." : path.substr(0, p);
}

static void progress_bar(double frac, const std::string& prefix = "Optimizando") {
    int w = 30, f = (int)(w * frac);
    std::cout << "\r-> " << prefix << ": [";
    for (int i = 0; i < w; ++i) std::cout << (i < f ? "█" : "-");
    std::cout << "] " << std::fixed << std::setprecision(1) << frac * 100.0 << "%  " << std::flush;
}

static float to_float(std::string s) {
    std::replace(s.begin(), s.end(), ',', '.');
    try { return std::stof(s); } catch (...) { return NAN; }
}

static bool cargar_datos(Dataset& ds) {
    std::ifstream f(ds.path);
    if (!f.is_open()) return false;
    std::string header, line, tok;
    if (!std::getline(f, header)) return false;
    char sep = '\t';
    if (header.find(';') != std::string::npos) sep = ';';
    int ix = 0, iy = 1, col = 0;
    std::istringstream hss(header);
    while (std::getline(hss, tok, sep)) {
        std::string tlow = tok;
        std::transform(tlow.begin(), tlow.end(), tlow.begin(), [](unsigned char c){ return std::tolower(c); });

        // Detección de Potencial (X)
        if (tlow.find("potential") != std::string::npos || 
            tlow.find("voltage")   != std::string::npos || 
            tlow.find("v_dc")      != std::string::npos || 
            tlow.find("potencial") != std::string::npos || 
            tlow.find("voltaje")   != std::string::npos) {
            ix = col;
        }
        
        // Detección de Capacitancia (Y)
        if (tlow.find("wz")          != std::string::npos || 
            tok.find("ωZ")           != std::string::npos || 
            tlow.find("c-2")         != std::string::npos || 
            tlow.find("c^-2")        != std::string::npos || 
            tlow.find("c**-2")       != std::string::npos || 
            tlow.find("1/c^2")       != std::string::npos || 
            tlow.find("capacitance") != std::string::npos || 
            tok.find("c-²")          != std::string::npos || 
            tok.find("c⁻²")          != std::string::npos) {
            iy = col;
        }
        col++;
    }
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::vector<std::string> parts;
        std::istringstream ls(line);
        while (std::getline(ls, tok, sep)) parts.push_back(tok);
        if ((int)parts.size() > std::max(ix, iy)) {
            float vx = to_float(parts[ix]), vy = to_float(parts[iy]);
            if (!std::isnan(vx) && !std::isnan(vy)) { ds.x.push_back(vx); ds.y.push_back(vy); }
        }
    }
    return !ds.x.empty();
}

static void estimar_iniciales(const Dataset& ds, float* P) {
    int n = ds.x.size();
    double sx = 0, sy = 0, sxy = 0, sx2 = 0;
    for (int i = 0; i < n; i++) { sx += ds.x[i]; sy += ds.y[i]; sxy += ds.x[i]*ds.y[i]; sx2 += ds.x[i]*ds.x[i]; }
    double m = (n*sxy - sx*sy) / (n*sx2 - sx*sx);
    double c = (sy - m*sx) / n;
    P[0] = (float)std::sqrt(std::max(m, 1e-9));
    P[1] = (float)(-c / std::max(m, 1e-9));
    float mx = -1e30f; int id = 0;
    for (int i = 2; i < n-2; i++) {
        float sm = (ds.y[i-2]+ds.y[i-1]+ds.y[i]+ds.y[i+1]+ds.y[i+2])/5.0f;
        if (sm > mx) { mx = sm; id = i; }
    }
    P[3] = ds.x[id];
    float Y1 = std::sqrt(std::max(ds.y[id], 1e-9f));
    float Z1 = Y1 - P[0]*std::sqrt(std::max(ds.x[id]-P[1], 1e-9f));
    int ifar  = (id > n/2) ? 0 : n-1;
    float Y2 = std::sqrt(std::max(ds.y[ifar], 1e-9f));
    float Z2 = Y2 - P[0]*std::sqrt(std::max(ds.x[ifar]-P[1], 1e-9f));
    float f2 = 1.0f / std::cosh(std::min(std::max((float)CFG.n_fijo*(ds.x[ifar]-P[3]), -20.0f), 20.0f));
    float dn = 1.0f - f2; if (std::abs(dn) < 1e-5f) dn = 1e-5f;
    P[2] = (Z1 - Z2) / dn;
    P[4] = Z1 - P[2];
    if (P[4] < 100.0f) P[4] = 100.0f;
}

// ==========================================
// CUDA KERNELS (FP32)
// ==========================================
__device__ float model_val(float x, float A, float b, float C, float d, float E, float nf) {
    float t1 = A * sqrtf(fmaxf(x - b, 1e-9f));
    float ch = nf * (x - d);
    ch = fmaxf(-80.0f, fminf(80.0f, ch));
    float t2 = C / coshf(ch);
    float Y  = t1 + t2 + E;
    return Y * Y;
}

__global__ void minimax_kernel_5d(
    const float* dx, const float* dy, int np,
    float As, float Ast, float bs, float bst,
    float Cs, float Cst, float ds_, float dst,
    float Es, float Est, int N, long long total,
    long long offset, float* bDiff, float* bPar, float nf)
{
    long long idx    = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x  * blockDim.x;
    float lmm = 1e30f, lb[5];
    for (long long i = idx; i < total; i += stride) {
        long long gi = i + offset;
        int iA = (int)(gi % N);
        int ib = (int)((gi / N) % N);
        int iC = (int)((gi / N / N) % N);
        int id = (int)((gi / N / N / N) % N);
        int iE = (int)((gi / N / N / N / N) % N);
        float A = As + iA*Ast, b = bs + ib*bst, C = Cs + iC*Cst, d = ds_ + id*dst, E = Es + iE*Est;
        if (E < 1e-5f) E = 1e-5f;
        float md = 0.0f;
        for (int p = 0; p < np; p++) {
            float diff = fabsf(model_val(dx[p], A, b, C, d, E, nf) - dy[p]);
            if (diff > md) md = diff;
        }
        if (md < lmm) { lmm = md; lb[0]=A; lb[1]=b; lb[2]=C; lb[3]=d; lb[4]=E; }
    }
    extern __shared__ float sh[];
    int tid = threadIdx.x, bs2 = blockDim.x;
    sh[tid] = lmm;
    for (int k = 0; k < 5; k++) sh[(k+1)*bs2 + tid] = lb[k];
    __syncthreads();
    for (int s = bs2/2; s > 0; s >>= 1) {
        if (tid < s && sh[tid+s] < sh[tid]) {
            sh[tid] = sh[tid+s];
            for (int k = 0; k < 5; k++) sh[(k+1)*bs2+tid] = sh[(k+1)*bs2+tid+s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        bDiff[blockIdx.x] = sh[0];
        for (int k = 0; k < 5; k++) bPar[blockIdx.x*5+k] = sh[(k+1)*bs2];
    }
}

// ==========================================
// AJUSTE GPU (Etapa 1 Única con Overrides)
// ==========================================
static void realizar_ajuste(Dataset& ds, const FileOverrides& ov) {
    // Determinar margen y N para este archivo
    double local_margen = (ov.margen > 0) ? ov.margen : ((g_global.margen > 0) ? g_global.margen : CFG.margen);
    int local_N = (ov.N > 0) ? ov.N : ((g_global.N > 0) ? g_global.N : CFG.N);
    ds.local_margen = local_margen;
    ds.local_N = local_N;

    LOGf("   -> Etapa 1: Búsqueda Global 5D (N=" + std::to_string(local_N) + ", margen=" + std::to_string(local_margen) + ")\n");
    
    float P_start[5]; estimar_iniciales(ds, P_start);
    float As[5], Ast[5];
    
    for (int k = 0; k < 5; k++) {
        ParamOverride po = ov.P[k].active ? ov.P[k] : g_global.P[k];
        float center = po.active ? po.value : P_start[k];
        ds.is_fixed[k] = (po.active && po.fixed);
        
        if (ds.is_fixed[k]) {
            As[k] = center;
            Ast[k] = 0.0f;
        } else {
            // Si el centro es 0, usamos un margen absoluto basado en 1.0 para permitir la búsqueda
            float range_base = (std::abs(center) < 1e-9f) ? 1.0f : std::abs(center);
            As[k]  = center - (range_base * (float)local_margen);
            Ast[k] = (range_base * (float)(2.0 * local_margen)) / (float)std::max(local_N - 1, 1);
        }
    }

    {
        std::string pnames[] = {"A", "b", "C", "d", "E"};
        std::ostringstream ss;
        ss << "   Configuración de búsqueda:\n";
        for(int k=0; k<5; k++) {
            ss << "     - " << pnames[k] << ": [" << As[k] << " a " << (As[k] + (local_N-1)*Ast[k]) << "] paso=" << Ast[k] << (ds.is_fixed[k] ? " (FIJO)" : "") << "\n";
        }
        LOGf(ss.str());
    }

    long long total1 = 1;
    for (int k = 0; k < 5; k++) {
        // Si el paso es 0 (fijo), solo hay 1 valor. Pero el kernel usa N iteraciones.
        // Sin embargo, el kernel hace: A = As + iA*Ast. Si Ast es 0, A es siempre As.
        // Entonces podemos usar local_N siempre, el resultado será el mismo pero desperdiciamos hilos.
        // Para simplificar, mantendremos local_N.
        total1 *= (long long)local_N;
    }

    int np = ds.x.size();
    float *dx, *dy; cudaMalloc(&dx, np*4); cudaMalloc(&dy, np*4);
    cudaMemcpy(dx, ds.x.data(), np*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, ds.y.data(), np*4, cudaMemcpyHostToDevice);
    
    const int TPB = 256, NB = 1024;
    float *bDiff, *bPar; cudaMalloc(&bDiff, NB*4); cudaMalloc(&bPar, NB*5*4);
    
    const int CHUNKS = 100;
    long long chunk1 = (total1 + CHUNKS - 1) / CHUNKS;
    float gmm1 = 1e30f;
    
    for (int c = 0; c < CHUNKS; c++) {
        long long off = c * chunk1; long long sz = std::min(chunk1, total1 - off);
        if (sz <= 0) break;
        minimax_kernel_5d<<<NB, TPB, TPB*6*4>>>(dx, dy, np, As[0], Ast[0], As[1], Ast[1], As[2], Ast[2], As[3], Ast[3], As[4], Ast[4], local_N, sz, off, bDiff, bPar, (float)CFG.n_fijo);
        cudaDeviceSynchronize();
        std::vector<float> hd(NB), hp(NB*5);
        cudaMemcpy(hd.data(), bDiff, NB*4, cudaMemcpyDeviceToHost); cudaMemcpy(hp.data(), bPar, NB*5*4, cudaMemcpyDeviceToHost);
        for (int i = 0; i < NB; i++) {
            if (hd[i] < gmm1) { gmm1 = hd[i]; for (int k = 0; k < 5; k++) ds.P[k] = hp[i*5+k]; }
        }
        progress_bar((double)(c+1)/CHUNKS, "E1-Global");
    }
    std::cout << "\n";
    ds.mmDiff = gmm1;
    LOGf("   Ajuste finalizado. Minimax Error: " + std::to_string(gmm1) + "\n");

    cudaFree(dx); cudaFree(dy); cudaFree(bDiff); cudaFree(bPar);
}

static Props calc_props(const Dataset& ds) {
    double A = ds.P[0], b = ds.P[1], C = ds.P[2], d = ds.P[3], E = ds.P[4];
    double step_factor = ds.local_margen / std::max(ds.local_N - 1, 1);
    
    auto get_err = [&](double val, int k) {
        return ds.is_fixed[k] ? 0.0 : std::abs(val * step_factor);
    };

    double eA = get_err(A, 0), eb = get_err(b, 1), eC = get_err(C, 2), ed = get_err(d, 3), eE = get_err(E, 4);
    Props p;
    p.N_D  = 2.0 / (CFG.eps_sc * CFG.eps_0 * CFG.e_0 * A*A);
    p.U_fb = b - (CFG.k_B * CFG.T) / CFG.e_0;
    p.C_0  = (CFG.k_B * CFG.T) / (2.0 * CFG.z*CFG.z * CFG.e_0*CFG.e_0 * CFG.eps_ele * CFG.eps_0 * C*C);
    p.U_z  = d; p.x_H  = CFG.eps_ele * CFG.eps_0 * E;
    if (p.x_H < 0) p.x_H = 1e-15;
    p.M_0  = (p.C_0 / CFG.N_A) * 1000.0;
    double ex = CFG.err_rel_x;
    p.eN_D  = std::abs(p.N_D  * 2.0 * eA / A); p.eU_fb = eb + ex * std::abs(b);
    p.eC_0  = std::abs(2.0 * p.C_0 * eC / C); p.eU_z  = ed + ex * std::abs(d);
    p.ex_H  = std::abs(p.x_H * eE / E); p.eM_0  = (p.eC_0 / CFG.N_A) * 1000.0;
    return p;
}

static void guardar_csv(const Dataset& ds, const std::string& ts) {
    std::string dir = ds.dir;
    std::ofstream fp(dir + "/params_gpu_" + ts + ".csv");
    fp << "# margen_iter=" << ds.local_margen << ", max_iter=" << ds.local_N << "\n";
    fp << "Parametro,Valor,±Error\n";
    std::string pnames[] = {"A (F⁻¹V⁻¹/²cm²)","b (V vs Ag/AgCl)","C (F⁻¹cm²)","d (V vs Ag/AgCl)","E (F⁻¹cm²)"};
    double step_factor = ds.local_margen / std::max(ds.local_N - 1, 1);
    for (int k = 0; k < 5; k++) {
        double step = ds.is_fixed[k] ? 0.0 : std::abs((double)ds.P[k]) * step_factor;
        fp << pnames[k] << "," << std::scientific << (double)ds.P[k] << ",±" << step << "\n";
    }
    fp.close();
    std::ofstream fq(dir + "/props_gpu_" + ts + ".csv");
    fq << "# margen_iter=" << ds.local_margen << ", max_iter=" << ds.local_N << "\n";
    fq << "Propiedad,Valor,±Error\n";
    const Props& p = ds.props;
    auto row = [&](const std::string& name, double v, double e){ fq << name << "," << std::scientific << v << ",±" << e << "\n"; };
    row("N_D (cm⁻³)", p.N_D, p.eN_D); row("U_fb (V)", p.U_fb, p.eU_fb); row("C_0 (cm⁻³)", p.C_0, p.eC_0); row("U_z (V)", p.U_z, p.eU_z); row("x_H (nm)", p.x_H*1e7, p.ex_H*1e7); row("M_0 (mol/L)", p.M_0, p.eM_0);
    fq.close();
}

static void exportar_datos_grafico(const std::vector<Dataset>& DV, const std::string& ts) {
    std::ofstream f("datos_grafico.csv");
    f << "Archivo,Label,A,b,C,d,E\n";
    for (const auto& ds : DV) f << ds.path << "," << ds.label << "," << ds.P[0] << "," << ds.P[1] << "," << ds.P[2] << "," << ds.P[3] << "," << ds.P[4] << "\n";
    f.close();
}

int main(int argc, char** argv) {
    parse_args(argc, argv);
    std::string ts = timestamp(); g_log.open("log_gpu_" + ts + ".txt");
    LOGf("\n**************************************************\n");
    LOGf(" INICIO - PROCESAMIENTO C⁻² vs U (GPU Global Search)\n");
    LOGf(" Autor: Santiago Décima\n");
    LOGf("**************************************************\n");
    {
        double m = (g_global.margen > 0) ? g_global.margen : CFG.margen;
        int n = (g_global.N > 0) ? g_global.N : CFG.N;
        std::ostringstream ss;
        ss << "Archivos a analizar: " << ARCHIVOS.size() << "\n"
           << "Margen Global: " << m << "   N Global: " << n << "\n";
        
        std::string pnames[] = {"A", "b", "C", "d", "E"};
        bool any_glob = false;
        for(int k=0; k<5; k++) {
            if(g_global.P[k].active) {
                if(!any_glob) { ss << "Overrides Globales: "; any_glob = true; }
                ss << pnames[k] << (g_global.P[k].fixed ? "==" : "=") << g_global.P[k].value << " ";
            }
        }
        if(any_glob) ss << "\n";
        LOGf(ss.str());
    }
    std::vector<Dataset> DV; int total_arch = 0;
    for (size_t fi = 0; fi < ARCHIVOS.size(); fi++) {
        Dataset ds; ds.path = ARCHIVOS[fi]; ds.label = (fi < LABELS.size()) ? LABELS[fi] : ARCHIVOS[fi]; ds.dir = dirof(ARCHIVOS[fi]);
        LOGf("\n[" + std::to_string(fi+1) + "/" + std::to_string(ARCHIVOS.size()) + "] Procesando: " + ds.path + "\n");
        if (!cargar_datos(ds)) { LOGf("[SKIP] Archivo no encontrado.\n"); continue; }
        LOGf("   -> " + std::to_string(ds.x.size()) + " puntos cargados.\n");
        realizar_ajuste(ds, g_overrides[fi]);
        double rmse = 0;
        for (size_t p = 0; p < ds.x.size(); p++) {
            float t1 = ds.P[0]*sqrtf(fmaxf(ds.x[p]-ds.P[1],1e-9f)); float t2 = ds.P[2]/coshf((float)CFG.n_fijo*(ds.x[p]-ds.P[3]));
            float res = ((t1+t2+ds.P[4])*(t1+t2+ds.P[4])) - ds.y[p]; rmse += (double)res*res;
        }
        rmse = std::sqrt(rmse / ds.x.size());
        ds.props = calc_props(ds); guardar_csv(ds, ts); DV.push_back(ds); total_arch++;
        {
            std::ostringstream ss;
            ss << "\n==================================================\n RESULTADOS FINALES - " << ds.label << "\n==================================================\n";
            ss << "   Minimax Error: " << std::scientific << (double)ds.mmDiff << "   RMSE: " << rmse << "\n";
            const Props& pr = ds.props;
            ss << std::setw(6) << "N_D"  << " = " << pr.N_D  << " ± " << pr.eN_D  << " cm⁻³\n";
            ss << std::setw(6) << "U_fb" << " = " << pr.U_fb << " ± " << pr.eU_fb << " V\n";
            ss << std::setw(6) << "C_0"  << " = " << pr.C_0  << " ± " << pr.eC_0  << " cm⁻³\n";
            ss << std::setw(6) << "U_z"  << " = " << pr.U_z  << " ± " << pr.eU_z  << " V\n";
            ss << std::setw(6) << "x_H"  << " = " << pr.x_H*1e7 << " ± " << pr.ex_H*1e7 << " nm\n";
            ss << std::setw(6) << "M_0"  << " = " << pr.M_0  << " ± " << pr.eM_0  << " mol/L\n";
            LOGf(ss.str());
        }
    }
    if (!DV.empty()) {
        exportar_datos_grafico(DV, ts); std::string cmd = "python3 graficador_c2vsu.py ajuste_multi_gpu_" + ts + ".png"; system(cmd.c_str());
    }
    LOGf("\nPROCESAMIENTO FINALIZADO. Archivos: " + std::to_string(total_arch) + "/" + std::to_string(ARCHIVOS.size()) + "\n");
    g_log.close(); return 0;
}
