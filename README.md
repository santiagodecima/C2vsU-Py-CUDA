# Manual de Usuario - C2vsU GPU Aceleration (Optimizado)
# Autor: Santiago Décima

Este programa realiza el ajuste estadístico del modelo físico-químico C_total^-2 vs U con el criterio de optimización Minimax. Disponible tanto para CPU (Py) como utilizando aceleración por GPU (CUDA C++ Py).

## 1. Requisitos y Compilación
- **Hardware**: Placa de video NVIDIA con soporte CUDA.
- **Software**: CUDA Toolkit (11.0+), Python 3 con `matplotlib`, `pandas` y `numpy`.
- **Compilación**:
  ```bash
  nvcc -O3 C2vsU_gpu_aceleration.cu -o C2vsU_gpu_aceleration
  ```
- **Archivos y Versiones**:
   - **Versión GPU (Principal)**: `C2vsU_gpu_aceleration.cu`. Requiere NVIDIA GPU.
   - **Versión CPU (Alternativa)**: Ubicada en la carpeta `Py/`. Ejecuta `python3 C2vsU_multiCPU.py`. Ideal para sistemas sin GPU NVIDIA.
   - `graficador_c2vsu.py`: Script de gráficos (compatible con ambas versiones).

## 2. Configuración de Archivos
Los archivos a analizar y sus etiquetas se configuran en el archivo `config.h`, en CUDA-C++-Py, y `config.py`, en Py puro:
### config.h:
```cpp
static const std::vector<std::string> ARCHIVOS = {
    "./Ef08/MS08.txt",
    "./Ef10/MS10.txt",
};
static const std::vector<std::string> LABELS = {
    "Ef=1.15 V vs ENH",
    "Ef=1.35 V vs ENH",
};
```
###config.py:
```py
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
```
## 3. Comandos y Flags de Ejecución
El programa permite configurar la búsqueda global y refinamientos específicos mediante flags:

### Flags Globales
- `--margen_iter=valor`: Define el margen de variación (ej. `0.4` para ±40%, `9.0` para ±900%).
- `--max_iter=N`: Define la cantidad de puntos por dimensión en la grilla de búsqueda.

### Flags de Parámetros (A, b, C, d, E)
- `--P=valor`: Centra la búsqueda del parámetro `P` en el `valor` indicado (usa el margen_iter global).
- `--P==valor`: **Fija** el parámetro `P` al valor exacto (no varía, error reported como 0.0).

### Control por Índice de Archivo
Se puede aplicar una configuración solo a un archivo específico usando el prefijo `N--`:
- `1--A=20000`: Para el primer archivo, centra `A` en 20000.
- `2--C==500`: Para el segundo archivo, fija `C` en 500.

### Ejemplos de Uso
1. **Ejecución Estándar** (usa valores de `config.h`):
   ```bash
   ./C2vsU_gpu_aceleration
   ```

2. **Búsqueda Global Refinada**:
   ```bash
   ./C2vsU_gpu_aceleration --margen_iter=0.1 --max_iter=100
   ```

3. **Fijar valores globales**:
   ```bash
   ./C2vsU_gpu_aceleration --E==5000 --C==-90000
   ```

4. **Combinación compleja** (Refinar margen en todos, fijar `E` globalmente, y centrar `C` solo en el archivo 1):
   ```bash
   ./C2vsU_gpu_aceleration --margen_iter=0.5 --max_iter=100 --E==5000 1--C=135000
   ```

## 4. Resultados y Gráficos
Al finalizar, el programa genera:
- `log_gpu_TIMESTAMP.txt`: Registro detallado de la búsqueda y rangos.
- `params_gpu_TIMESTAMP.csv` y `props_gpu_TIMESTAMP.csv`: Resultados en cada carpeta.
- `grafico_ajuste_combined.png`: Gráfico con todos los datasets.
- `ajuste_ARCHIVO.png`: Gráficos individuales dentro de cada carpeta (`./Ef08/`, etc.).

---
