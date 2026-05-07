import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import csv
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURACIÓN ESTÉTICA Y DE LEYENDAS
# ==========================================
CONF = {
    'fs_titulo': 20,    # Tamaño de fuente del título superior
    'fs_ejes': 17,      # Tamaño de fuente de los nombres de los ejes (X e Y)
    'fs_ticks': 17,     # Tamaño de fuente de los números en las escalas de los ejes
    'fs_offset': 17,    # Tamaño de fuente del factor de escala (ej: 1e8) arriba del eje Y
    'fs_leyenda': 12,   # Tamaño de fuente del cuadro de leyenda
    'dpi': 300,         # Resolución de la imagen (puntos por pulgada)
    'line_width': 2.5,  # Grosor de la línea del ajuste
    'marker_size': 17,  # Tamaño de los puntos experimentales
    # Templates de Leyenda (Modificar para cambiar el texto del recuadro)
    'label_exp': "Datos experimentales ({label})",
    'label_fit': "Ajuste Minimax ({label})",
    'titulo_combined': '(b) Ajuste Minimax: Sistemas ChapaTi/TiO$_{2}$ y Vidrio/Ti/TiO$_{2}$ en 0.5 M HClO$_{4}$',
    'titulo_individual': 'Ajuste Individual - {label}'
}

def cargar_datos_experimentales(path):
    """Carga los datos experimentales detectando columnas y formatos numéricos (coma/punto) de forma robusta."""
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            header = f.readline().lower()
            sep = '\t'
            if ';' in header: sep = ';'
            elif ',' in header: sep = ','
        
        # Leer como strings inicialmente para limpiar manualmente si es necesario
        df = pd.read_csv(path, sep=sep, dtype=str, skipinitialspace=True)
        
        # Palabras clave para X (Potencial) e Y (Capacitancia)
        keys_x = ['potential', 'voltage', 'v_dc', 'potencial', 'voltaje']
        keys_y = ['wz', 'ωz', 'c-2', 'c^-2', 'c**-2', '1/c^2', 'capacitance', 'c-²', 'c⁻²']
        
        col_x = next((c for c in df.columns if any(k in c.lower() for k in keys_x)), df.columns[0])
        col_y = next((c for c in df.columns if any(k in c.lower() for k in keys_y)), df.columns[1])
        
        def clean_numeric(series):
            # Reemplazar coma por punto y convertir a float
            return pd.to_numeric(series.str.replace(',', '.', regex=False), errors='coerce')

        x = clean_numeric(df[col_x]).values
        y = clean_numeric(df[col_y]).values
        
        mask = ~np.isnan(x) & ~np.isnan(y)
        if not np.any(mask):
            raise ValueError("No se encontraron datos numéricos válidos en las columnas seleccionadas.")
            
        return x[mask].astype(float), y[mask].astype(float)
        
    except Exception as e:
        print(f"Error cargando {path}: {e}")
        return None, None

def plot_dataset(ax, x_exp, y_exp, A, b, C, d, E, label, color_pair):
    n_fijo = 19.4706671
    fit_color, exp_color = color_pair
    
    # Datos Experimentales
    ax.scatter(x_exp, y_exp, color=exp_color, s=CONF['marker_size'], alpha=0.7, 
               label=CONF['label_exp'].format(label=label))
    
    # Ajuste
    x_fit = np.linspace(min(x_exp), max(x_exp), 300)
    term1 = A * np.sqrt(np.maximum(x_fit - b, 1e-9))
    term2 = C / np.cosh(n_fijo * (x_fit - d))
    y_fit = (term1 + term2 + E)**2
    
    ax.plot(x_fit, y_fit, color=fit_color, linewidth=CONF['line_width'], alpha=0.9,
            label=CONF['label_fit'].format(label=label))

def main():
    output_name = sys.argv[1] if len(sys.argv) > 1 else "grafico_ajuste_combined.png"
    
    color_pairs = [
        ('red', '#8B0000'), ('blue', '#00008B'), ('green', '#006400'),
        ('orange', '#FF8C00'), ('purple', '#4B0082'), ('cyan', '#008B8B'),
        ('magenta', '#8B008B'), ('brown', '#5D4037')
    ]
    
    datasets = []
    with open("datos_grafico.csv", mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            datasets.append(row)

    # --- GRÁFICO COMBINADO ---
    fig_comb, ax_comb = plt.subplots(figsize=(10, 7))
    
    for i, ds in enumerate(datasets):
        path = ds['Archivo']
        label = ds['Label']
        params = [float(ds[k]) for k in ['A', 'b', 'C', 'd', 'E']]
        
        x_exp, y_exp = cargar_datos_experimentales(path)
        if x_exp is None: continue
        
        # Graficar en el combinado
        plot_dataset(ax_comb, x_exp, y_exp, *params, label, color_pairs[i % len(color_pairs)])
        
        # --- GRÁFICO INDIVIDUAL ---
        fig_ind, ax_ind = plt.subplots(figsize=(10, 7))
        plot_dataset(ax_ind, x_exp, y_exp, *params, label, color_pairs[0]) # Usar rojo para individuales
        
        ax_ind.set_xlabel('Potencial U (V vs ENH)', fontsize=CONF['fs_ejes'])
        ax_ind.set_ylabel('C_{total}⁻² (cm⁴ F⁻²)', fontsize=CONF['fs_ejes'])
        ax_ind.set_title(CONF['titulo_individual'].format(label=label), fontsize=CONF['fs_titulo'])
        ax_ind.tick_params(labelsize=CONF['fs_ticks'])
        ax_ind.yaxis.get_offset_text().set_fontsize(CONF['fs_offset'])
        ax_ind.grid(True, linestyle='--', alpha=0.7)
        ax_ind.legend(fontsize=CONF['fs_leyenda'], loc='best')
        
        # Guardar en el directorio del archivo
        file_dir = os.path.dirname(path)
        ind_name = f"ajuste_{os.path.basename(path).replace('.txt', '')}.png"
        ind_path = os.path.join(file_dir, ind_name)
        fig_ind.tight_layout()
        fig_ind.savefig(ind_path, dpi=CONF['dpi'])
        plt.close(fig_ind)
        print(f"Gráfico individual guardado: {ind_path}")

    # Finalizar combinado
    ax_comb.set_xlabel('Potencial U (V vs ENH)', fontsize=CONF['fs_ejes'])
    ax_comb.set_ylabel(r'$C_{\mathrm{total}}^{-2}$ (cm$^4$ F$^{-2}$)', fontsize=CONF['fs_ejes'])
    ax_comb.set_title(CONF['titulo_combined'], fontsize=CONF['fs_titulo'])
    ax_comb.tick_params(labelsize=CONF['fs_ticks'])
    ax_comb.yaxis.get_offset_text().set_fontsize(CONF['fs_offset'])
    ax_comb.grid(True, linestyle='--', alpha=0.7)
    ax_comb.legend(fontsize=CONF['fs_leyenda'], loc='best', ncol=1)
    
    fig_comb.tight_layout()
    fig_comb.savefig(output_name, dpi=CONF['dpi'])
    plt.close(fig_comb)
    print(f"Gráfico combinado guardado: {output_name}")

if __name__ == "__main__":
    main()
