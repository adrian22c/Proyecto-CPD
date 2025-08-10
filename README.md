# Predicción Distribuida de Vida Útil Remanente (RUL)
## Análisis Comparativo de Dask vs Enfoques Tradicionales

Este proyecto implementa un pipeline científico completo para la predicción de vida útil remanente (RUL) en motores turbofan utilizando el dataset C-MAPSS de la NASA. El objetivo principal es evaluar la eficacia de Dask para el procesamiento distribuido en comparación con enfoques tradicionales usando Pandas.

## Descripción del Proyecto

### Características Principales

- **Preprocesamiento Avanzado**: Limpieza de datos, eliminación de variables constantes y selección de sensores basada en información mutua
- **Ingeniería de Características**: Generación de características de ventana móvil, tendencias y estadísticas acumulativas
- **Modelado Comparativo**: Evaluación de múltiples algoritmos (LightGBM, XGBoost, Random Forest, Ridge)
- **Análisis de Escalabilidad**: Comparación de rendimiento entre diferentes configuraciones de workers
- **Visualizaciones Científicas**: Gráficos comprehensivos para análisis de resultados
- **Reporte Automatizado**: Generación de documentación científica completa

### Dataset

El proyecto utiliza el dataset C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) de la NASA, específicamente el subconjunto FD001. Este dataset contiene:

- Datos de entrenamiento con trayectorias completas de degradación
- Datos de prueba con trayectorias parciales
- Valores reales de RUL para evaluación
- 21 sensores de monitoreo de condiciones

## Requisitos del Sistema

### Requisitos Mínimos

- **Python**: 3.8 o superior
- **RAM**: 8 GB mínimo, 16 GB recomendado
- **Procesador**: Multi-core recomendado para aprovechar paralelización
- **Almacenamiento**: 5 GB de espacio libre

### Entorno Recomendado

- **Google Colab**: Configuración optimizada incluida
- **Jupyter Notebook**: Compatible con entornos locales
- **Sistema Operativo**: Linux/Windows/MacOS

## Instalación y Configuración

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/adrian22c/Proyecto-CPD.git
cd Proyecto-CPD
```

### Paso 2: Instalar Dependencias

#### Opción A: Instalación Automática (Recomendada)

```bash
pip install lightgbm dask-ml xgboost seaborn
```

#### Opción B: Instalación Manual

```bash
pip install pandas numpy matplotlib seaborn
pip install dask[complete] dask-ml
pip install lightgbm xgboost
pip install scikit-learn joblib psutil
```

### Paso 3: Descargar Dataset

1. Acceder al repositorio de datos de la NASA: [NASA Prognostics Center](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
2. Descargar el archivo `6. Turbofan Engine Degradation Simulation`
3. Extraer los archivos en la carpeta del proyecto:
   - `train_FD001.txt`
   - `test_FD001.txt`
   - `RUL_FD001.txt`

### Paso 4: Configurar Estructura de Directorios

```
Proyecto-CPD/
├── CMAPSSData/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   ├── RUL_FD001.txt
│   ├── clean/
│   ├── features/
│   ├── models/
│   └── paper_results/
│       └── plots/
├── notebooks/
│   └── Pipline_Proyecto_CPD.ipynb
├── README.md
└── requirements.txt
```

## Ejecución del Experimento

### Método 1: Google Colab (Recomendado)

1. Subir el notebook `Pipline_Proyecto_CPD.ipynb` a Google Colab
2. Subir los archivos del dataset a Google Drive
3. Modificar la variable `BASE` en el código para apuntar a la ubicación correcta
4. Ejecutar todas las celdas secuencialmente

```python
# Modificar esta línea según su estructura de Google Drive
BASE = '/content/drive/MyDrive/CMAPSSData'
```

### Método 2: Entorno Local

1. Abrir Jupyter Notebook o JupyterLab
2. Navegar al archivo `Pipline_Proyecto_CPD.ipynb`
3. Modificar la ruta base según su configuración local
4. Ejecutar todas las celdas

```python
# Para entorno local
BASE = '/ruta/a/su/proyecto/CMAPSSData'
```

## Configuración Avanzada

### Parámetros de Configuración

Puede modificar los siguientes parámetros en la sección de configuración:

```python
# Configuración de experimento
WINDOWS = [3, 5, 10, 15]  # Ventanas para características temporales
CROSS_VALIDATION_SPLITS = 5  # Número de splits para validación cruzada

# Configuración de modelos
RANDOM_STATE = 42  # Semilla para reproducibilidad
N_JOBS = -1  # Número de cores a utilizar (-1 = todos)

# Configuración de escalabilidad
WORKER_CONFIGS = [
    (1, 1),   # 1 worker, 1 thread
    (2, 1),   # 2 workers, 1 thread cada uno
    (2, 2),   # 2 workers, 2 threads cada uno
    (4, 1),   # 4 workers, 1 thread cada uno
    (4, 2),   # 4 workers, 2 threads cada uno
]
```

### Configuración de Memoria

Para datasets grandes, ajuste la configuración de Dask:

```python
import dask
dask.config.set({'distributed.worker.memory.target': 0.8})
dask.config.set({'distributed.worker.memory.spill': 0.9})
```

## Estructura del Pipeline

### Etapa 1: Preprocesamiento
- Carga de datos con Dask DataFrames
- Identificación y eliminación de variables constantes
- Cálculo de RUL para datos de entrenamiento
- Limpieza y validación de datos

### Etapa 2: Ingeniería de Características
- Selección de sensores usando información mutua
- Generación de características de ventana móvil
- Cálculo de tendencias y estadísticas acumulativas
- Normalización de características

### Etapa 3: Análisis Exploratorio
- Distribución de RUL
- Análisis de correlaciones
- Evolución temporal de sensores
- Estadísticas descriptivas

### Etapa 4: Modelado y Evaluación
- Entrenamiento de múltiples algoritmos
- Búsqueda de hiperparámetros
- Validación cruzada temporal
- Evaluación con métricas estándar

### Etapa 5: Análisis Comparativo
- Comparación Dask vs Pandas
- Medición de tiempos de ejecución
- Análisis de uso de memoria
- Evaluación de escalabilidad

### Etapa 6: Generación de Resultados
- Creación de visualizaciones científicas
- Guardado de modelos entrenados
- Generación de reporte científico
- Exportación de resultados

## Resultados Esperados

### Archivos de Salida

Al completar la ejecución, se generarán los siguientes archivos:

```
paper_results/
├── experiment_log.txt                    # Log completo del experimento
├── scientific_report.md                  # Reporte científico
├── comprehensive_results.json            # Resultados numéricos
├── scalability_analysis.csv              # Análisis de escalabilidad
└── plots/
    ├── model_comparison_comprehensive.png
    ├── best_model_analysis.png
    ├── scalability_analysis.png
    ├── dask_vs_pandas_comparison.png
    ├── mutual_information_analysis.png
    ├── correlation_matrix_selected_sensors.png
    ├── sensor_temporal_evolution.png
    └── cross_validation_results.png

features/
├── train_features_advanced.parquet
├── test_features_advanced.parquet
├── mutual_information_analysis.csv
└── descriptive_statistics.csv

models/
├── best_model_ridge.pkl                  # Mejor modelo entrenado
└── scaler.pkl                           # Normalizador
```

### Métricas de Evaluación

- **RMSE (Root Mean Square Error)**: Métrica principal de evaluación
- **MAE (Mean Absolute Error)**: Error absoluto medio
- **R² Score**: Coeficiente de determinación
- **Tiempo de Entrenamiento**: Eficiencia computacional
- **Uso de Memoria**: Consumo de recursos

### Resultados Típicos

Con la configuración por defecto, puede esperar:

- **RMSE**: ~170-180 ciclos
- **R²**: Variable según configuración
- **Tiempo Total**: 2-4 horas (dependiendo del hardware)
- **Mejor Modelo**: Típicamente Ridge o LightGBM

## Solución de Problemas

### Problemas Comunes

#### Error de Memoria Insuficiente

```python
# Reducir el tamaño de la muestra
sample = train.sample(frac=0.1).compute()

# Usar menos particiones en Dask
train_dd = dd.from_pandas(train_pd, npartitions=2)
```

#### Error de Rutas de Archivos

```python
# Verificar que los archivos existan
import os
assert os.path.exists(f'{BASE}/train_FD001.txt'), "Archivo train_FD001.txt no encontrado"
```

#### Problemas con Dask

```python
# Reiniciar cliente Dask
client.restart()

# O crear nuevo cliente
client = Client(n_workers=4, threads_per_worker=2)
```

### Optimización de Rendimiento

#### Para Sistemas con Poca RAM

```python
# Configuración conservadora
dask.config.set({'distributed.worker.memory.target': 0.6})
WORKER_CONFIGS = [(1, 1), (2, 1)]  # Menos workers
```

#### Para Sistemas Potentes

```python
# Configuración agresiva
dask.config.set({'distributed.worker.memory.target': 0.9})
WORKER_CONFIGS = [(4, 2), (8, 2), (16, 1)]  # Más workers
```

## Personalización del Experimento

### Modificar Algoritmos de ML

```python
# Agregar nuevos modelos
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

models['SVR'] = SVR()
models['MLP'] = MLPRegressor(random_state=42)
```

### Cambiar Métricas de Evaluación

```python
from sklearn.metrics import mean_absolute_percentage_error

# Agregar nueva métrica
mape = mean_absolute_percentage_error(y_test, y_pred)
```

### Configurar Nuevas Características

```python
# Modificar ventanas temporales
WINDOWS = [5, 10, 20, 30]

# Agregar nuevas transformaciones
result[f'{sensor}_diff2'] = grouped.diff(2).fillna(0)
result[f'{sensor}_pct_change'] = grouped.pct_change().fillna(0)
```

## Validación y Reproducibilidad

### Semillas Aleatorias

El código utiliza semillas fijas para garantizar reproducibilidad:

```python
RANDOM_STATE = 42
np.random.seed(42)
```

### Validación de Resultados

Para validar que el experimento se ejecutó correctamente:

1. Verificar que todos los archivos de salida se generaron
2. Comprobar que el log no contiene errores críticos
3. Revisar las métricas en el reporte científico
4. Validar que las visualizaciones se crearon correctamente

### Control de Calidad

```python
# Verificar integridad de datos
assert not train_features.isnull().any().any(), "Datos faltantes en características"
assert len(results) > 0, "No se entrenaron modelos"
assert os.path.exists(f'{RESULTS_DIR}/scientific_report.md'), "Reporte no generado"
```

## Contribuciones y Desarrollo

### Estructura del Código

- **Funciones de preprocesamiento**: `load_and_preprocess_data()`
- **Ingeniería de características**: `advanced_feature_engineering()`
- **Modelado**: `comprehensive_modeling()`
- **Análisis**: `dask_vs_pandas_comparison()`, `scalability_analysis()`
- **Visualización**: `create_comprehensive_visualizations()`
- **Reporte**: `generate_scientific_report()`

### Extensiones Sugeridas

1. **Nuevos Algoritmos**: Implementar LSTM, CNN-LSTM
2. **Más Datasets**: Soporte para FD002, FD003, FD004
3. **Técnicas Avanzadas**: Ensemble methods, AutoML
4. **Optimización**: Hyperopt, Optuna para búsqueda de hiperparámetros
5. **Despliegue**: MLflow para seguimiento de experimentos

---

**Nota**: Este README está diseñado para facilitar la replicación completa del experimento. Asegúrese de seguir todos los pasos en orden y verificar los requisitos del sistema antes de la ejecución.
