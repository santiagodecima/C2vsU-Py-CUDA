# ==========================================
# ARCHIVOS A ANALIZAR
# ==========================================
ARCHIVOS = [
    "../datos/Ef08/MS08.txt",
    "../datos/Ef10/MS10.txt",
    "../datos/Ef12/MS12.txt",
    "../datos/Ef14/MS14.txt",
]

LABELS = [
    "Ef=1.15 V vs ENH",
    "Ef=1.35 V vs ENH",
    "Ef=1.55 V vs ENH",
    "Ef=1.75 V vs ENH",
]

# ==========================================
# CONFIGURACIÓN Y CONSTANTES FÍSICAS
# ==========================================
class Cfg:
    n_fijo    = 19.4706671
    eps_sc    = 58.0
    eps_ele   = 76.0
    eps_0     = 8.8541878176e-14
    e_0       = 1.602176565e-19
    k_B       = 1.380649e-23
    T         = 298.0
    z         = 1.0
    N_A       = 6.02214076e23
    err_rel_x = 0.0001
    err_rel_y = 0.02
    margen    = 9.0 
    N         = 20 
