#ifndef CONFIG_H
#define CONFIG_H

#include <vector>
#include <string>

// ==========================================
// ARCHIVOS A ANALIZAR
// ==========================================
static const std::vector<std::string> ARCHIVOS = {
    "../datos/05M/MS3.txt",
    "../datos/05Mc/MS4c.txt",
};

static const std::vector<std::string> LABELS = {
    "Vidrio/Ti/TiO$_{2}$ en 0.5 M HClO$_{4}$",
    "ChapaTi/Ti/TiO$_{2}$ en 0.5 M HClO$_{4}$",
};

// ==========================================
// CONFIGURACIÓN Y CONSTANTES FÍSICAS
// ==========================================
struct Cfg {
    double n_fijo    = 19.4706671;
    double eps_sc    = 58.0;
    double eps_ele   = 76.0;
    double eps_0     = 8.8541878176e-14;
    double e_0       = 1.602176565e-19;
    double k_B       = 1.380649e-23;
    double T         = 298.0;
    double z         = 1.0;
    double N_A       = 6.02214076e23;
    double err_rel_x = 0.0001;
    double err_rel_y = 0.02;
    double margen    = 9.0; 
    int    N         = 100; 
} CFG;

#endif // CONFIG_H
