# Priorización de clientes para campañas de depósitos a plazo

Proyecto integrador del curso **Data Mining Tools (CC209) – UPC, 2026**. Estado actual: **TP1 (semana 7)**.

## Problema

Un banco portugués vende depósitos a plazo por teléfono y solo el 11.7 % de las llamadas termina en suscripción. La pregunta del proyecto es:

> Con la información disponible **antes de llamar**, ¿se puede ordenar a los clientes de modo que llamando solo al 20 % con mayor puntaje se alcance a la mayoría de los que suscribirían?

- **Tipo de problema:** clasificación binaria con clase minoritaria, usada como ranking.
- **Unidad de análisis:** un contacto de campaña.
- **Criterios de utilidad:** captar al menos el 50 % de los suscriptores llamando al 20 % de los clientes, superar a una regla de negocio simple y no usar información posterior a la llamada.

## Datos

| | |
|---|---|
| Dataset | [Bank Marketing – UCI ML Repository, id 222](https://archive.ics.uci.edu/dataset/222/bank+marketing) (versión `bank-full`) |
| Autores | S. Moro, P. Rita, P. Cortez (2014). DOI [10.24432/C5K306](https://doi.org/10.24432/C5K306) |
| Licencia | CC BY 4.0 |
| Tamaño | 45 211 contactos, 16 variables de entrada, objetivo `y` (11.7 % "yes") |
| Periodo | Mayo 2008 – noviembre 2010, ordenado por fecha (sin año explícito) |

`data/raw/bank_marketing.csv` es la copia sin modificar que entrega `ucimlrepo`. Si se borra, `python -m src.data` la descarga de nuevo.

La variable `duration` se **excluye** del modelo: se conoce solo cuando la llamada terminó, es decir, cuando ya se sabe el resultado (data leakage).

## Resultados del TP1 (conjunto de validación)

| Modelo | PR-AUC | ROC-AUC | Captura en top 20 % |
|---|---|---|---|
| Baseline trivial (Dummy) | 0.117 | 0.500 | 20.7 % |
| Baseline regla de negocio | 0.245 | 0.616 | 37.3 % |
| Regresión logística | 0.416 | 0.773 | 56.2 % |
| Random Forest | 0.465 | 0.802 | 63.3 % |
| HistGradientBoosting (modelo preliminar) | 0.478 | 0.803 | 63.1 % |

En una prueba temporal (entrenar con el 80 % más antiguo de train + validation y evaluar en el 20 % más reciente) la ROC-AUC baja a ~0.68 y la captura a 32-37 %, casi lo mismo que la regla de negocio. Ese es el principal problema abierto para el TF1. El conjunto de test (20 %) **no se ha usado** todavía.

Detalle e interpretación en los notebooks.

## Estructura

```
TP/
├── streamlit_app.py                 aplicación (punto de entrada para Streamlit Cloud)
├── .streamlit/config.toml           tema y opciones de la app
├── data/raw/bank_marketing.csv      datos originales
├── notebooks/
│   ├── 01_eda.ipynb                 problema, dataset, EDA y calidad de datos
│   └── 02_preparacion_y_modelos.ipynb  split, pipeline, baselines, modelos, robustez, plan TF1
├── src/
│   ├── config.py                    rutas, semilla y listas de columnas
│   ├── data.py                      descarga, limpieza fija y split train/val/test
│   ├── features.py                  variables derivadas, ColumnTransformer y Pipeline
│   └── evaluate.py                  métricas de ranking y de umbral
├── models/modelo_preliminar.joblib  pipeline completo + umbral (lo carga la app)
├── reports/
│   ├── figures/                     gráficos exportados por los notebooks
│   ├── metricas_validacion.csv
│   └── metricas_temporal.csv
├── requirements.txt                 dependencias de la app (las usa Streamlit Cloud)
├── requirements-dev.txt             + dependencias de los notebooks
└── README.md
```

## Cómo ejecutar

Requiere Python 3.11 o superior (desarrollado con 3.13).

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

(en macOS / Linux: `source .venv/bin/activate`)

```bash
pip install -r requirements-dev.txt
```

### Notebooks

Ejecutar en orden, desde la carpeta raíz del proyecto:

```bash
python -m nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb notebooks/02_preparacion_y_modelos.ipynb
```

También se pueden abrir en Jupyter o VS Code y ejecutarlos celda por celda. Toda la aleatoriedad usa `SEED = 42` (`src/config.py`), así que los resultados se reproducen. El notebook 02 regenera `models/modelo_preliminar.joblib`.

### Aplicación

```bash
python -m streamlit run streamlit_app.py
```

Se abre en http://localhost:8501. Tiene tres pestañas:

- **Evaluar un cliente**: formulario con los datos previos a la llamada; devuelve el puntaje, la posición en el ranking y la recomendación (llamar si está en el 20 % superior).
- **Priorizar una lista (CSV)**: sube un CSV (formato del proyecto o el original de UCI) y descarga la lista ordenada. Incluye 200 clientes de ejemplo.
- **Sobre el modelo**: flujo, métricas de validación y limitaciones.

Flujo: **Usuario → formulario / CSV → preprocesamiento (`src/features.py`) → modelo (`models/*.joblib`) → puntaje y recomendación**. La app carga el mismo `Pipeline` guardado por el notebook, así que aplica exactamente las transformaciones del entrenamiento.

Uso del modelo desde Python:

```python
import joblib
art = joblib.load("models/modelo_preliminar.joblib")
proba = art["pipeline"].predict_proba(X_nuevo)[:, 1]   # X_nuevo con las columnas de art["columnas_entrada"]
llamar = proba >= art["umbral"]
```

## Despliegue en Streamlit Community Cloud

El repositorio ya está preparado; no hace falta cambiar código.

1. Subir el proyecto a un repositorio de GitHub (puede ser público o privado). Deben subirse `models/modelo_preliminar.joblib` y `data/raw/bank_marketing.csv`: la app los necesita y no están en `.gitignore`.
2. En https://share.streamlit.io → **Create app** → elegir el repositorio y la rama.
3. **Main file path:** `streamlit_app.py`.
4. En **Advanced settings → Python version** elegir **3.13** (pandas 3 no funciona con versiones anteriores a 3.11).
5. Deploy. Streamlit instala `requirements.txt` automáticamente; no se necesitan secretos.

Si se vuelve a entrenar el modelo con otra versión de scikit-learn, hay que actualizar la versión en `requirements.txt`: el `.joblib` solo se carga de forma confiable con la misma versión con la que se guardó.

**Plan de contingencia para la exposición:** si falla la conexión, ejecutar la app localmente con el comando de la sección anterior; no depende de ningún servicio externo.

## Matriz de decisiones de herramientas (TP1)

| Necesidad | Herramienta elegida | Alternativa considerada | Justificación |
|---|---|---|---|
| Obtención de datos | `ucimlrepo` + copia en `data/raw` | Descarga manual del zip | Descarga reproducible desde la fuente oficial; la copia local evita depender de que UCI esté en línea |
| EDA | pandas + matplotlib/seaborn | ydata-profiling | Un reporte automático genera decenas de gráficos sin pregunta detrás; preferimos gráficos elegidos para cada pregunta |
| Preparación | `Pipeline` + `ColumnTransformer` de scikit-learn | Transformar el DataFrame a mano | Ajusta las transformaciones solo con train, se reaplica igual a datos nuevos y se guarda junto al modelo |
| Separación | Hold-out estratificado 60/20/20 + `StratifiedKFold` (5) | Split temporal como esquema principal | El aleatorio permite comparar modelos con más datos de cada clase; el temporal se usa como prueba de robustez y pasará a ser el principal en el TF1 |
| Baseline | `DummyClassifier` + regla de negocio | Solo Dummy | El Dummy es demasiado fácil de superar; la regla representa lo que el banco haría sin modelo |
| Modelamiento | Regresión logística, Random Forest, HistGradientBoosting | XGBoost / LightGBM | Cubren lineal vs. árboles con una sola librería; boosting externo queda como opción del TF1 si el ajuste lo justifica |
| Métricas | PR-AUC, captura y lift en top 20 %, ROC-AUC | Accuracy | Con 11.7 % de positivos la accuracy es engañosa; la captura en el top 20 % es la que responde la pregunta del negocio |
| Persistencia | `joblib` | pickle / ONNX | Formato estándar para objetos de scikit-learn; suficiente para la app del TF1 |
| Experimentación (TF1) | Optuna + MLflow | GridSearchCV | Pendiente |
| Interpretabilidad (TF1) | SHAP + importancia por permutación | Solo coeficientes | Pendiente |
| Despliegue (TF1) | Streamlit | FastAPI | Pendiente: una interfaz es más útil para la demostración que un endpoint |

## Uso de herramientas de IA generativa

Se usó un asistente de IA (Claude, de Anthropic) como apoyo técnico para estructurar el repositorio, escribir el código de los módulos de `src/` y de los notebooks. El grupo revisó y validó los resultados y es responsable del análisis y las conclusiones.
