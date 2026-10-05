"""Aplicación Streamlit: priorización de clientes para la campaña de depósitos a plazo.

Ejecutar localmente desde la raíz del proyecto:
    streamlit run streamlit_app.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))  # el pipeline guardado referencia funciones de src.features

from src import config, data  # noqa: E402

MODEL_PATH = config.MODELS_DIR / "modelo_preliminar.joblib"

JOBS = ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
        "self-employed", "services", "student", "technician", "unemployed"]
ETIQUETAS_JOB = {
    "admin.": "Administrativo", "blue-collar": "Obrero", "entrepreneur": "Emprendedor",
    "housemaid": "Trabajador/a del hogar", "management": "Gerencia", "retired": "Jubilado/a",
    "self-employed": "Independiente", "services": "Servicios", "student": "Estudiante",
    "technician": "Técnico", "unemployed": "Desempleado/a",
}
ETIQUETAS_MES = dict(zip(config.MONTH_ORDER, ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio",
                                             "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]))
DESCONOCIDO = "(desconocido)"

st.set_page_config(page_title="Priorización de llamadas", page_icon="📞", layout="wide")


@st.cache_resource
def cargar_modelo():
    return joblib.load(MODEL_PATH)


@st.cache_data
def scores_referencia():
    """Puntajes del modelo sobre validation: sirven para ubicar a un cliente nuevo en el ranking."""
    df = data.clean(data.load_raw())
    _, val, _ = data.train_val_test_split(df)
    X_val, _ = data.split_xy(val)
    return np.sort(cargar_modelo()["pipeline"].predict_proba(X_val)[:, 1])


def percentil(score):
    ref = scores_referencia()
    return np.searchsorted(ref, score, side="right") / len(ref) * 100


def preparar_entrada(df, columnas):
    """Acepta tanto el formato del proyecto como el CSV original de UCI."""
    df = df.rename(columns={"day_of_week": "day"}).copy()
    df = df.replace({"unknown": np.nan, "": np.nan})
    faltan = [c for c in columnas if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas: {', '.join(faltan)}")
    return df[columnas]


art = cargar_modelo()
pipeline, umbral, columnas = art["pipeline"], art["umbral"], art["columnas_entrada"]

st.title("📞 Priorización de clientes para depósitos a plazo")
st.caption(
    "Proyecto integrador – Data Mining Tools (CC209), UPC. "
    f"Modelo: {art['modelo']} entrenado con el dataset Bank Marketing (UCI). "
    "Usa solo información disponible antes de llamar."
)

tab_cliente, tab_lista, tab_info = st.tabs(["Evaluar un cliente", "Priorizar una lista (CSV)", "Sobre el modelo"])

# --- Un cliente -------------------------------------------------------------------------
with tab_cliente:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Perfil")
        age = st.number_input("Edad", 18, 100, 40)
        job = st.selectbox("Ocupación", [DESCONOCIDO] + JOBS, index=5,
                           format_func=lambda j: ETIQUETAS_JOB.get(j, j))
        marital = st.selectbox("Estado civil", ["married", "single", "divorced"],
                               format_func={"married": "Casado/a", "single": "Soltero/a",
                                            "divorced": "Divorciado/a o viudo/a"}.get)
        education = st.selectbox("Educación", ["secondary", "tertiary", "primary", DESCONOCIDO],
                                 format_func=lambda e: {"secondary": "Secundaria", "tertiary": "Superior",
                                                        "primary": "Primaria"}.get(e, e))
    with c2:
        st.subheader("Situación financiera")
        balance = st.number_input("Saldo promedio anual (€)", -10000, 200000, 500, step=100)
        housing = st.radio("¿Préstamo hipotecario?", ["no", "yes"], horizontal=True,
                           format_func={"no": "No", "yes": "Sí"}.get)
        loan = st.radio("¿Préstamo personal?", ["no", "yes"], horizontal=True,
                        format_func={"no": "No", "yes": "Sí"}.get)
        default = st.radio("¿Crédito en mora?", ["no", "yes"], horizontal=True,
                           format_func={"no": "No", "yes": "Sí"}.get)
    with c3:
        st.subheader("Campaña")
        contact = st.selectbox("Medio de contacto", ["cellular", "telephone", DESCONOCIDO],
                               format_func=lambda c: {"cellular": "Celular", "telephone": "Teléfono fijo"}.get(c, c))
        month = st.selectbox("Mes de la llamada", config.MONTH_ORDER, index=4, format_func=ETIQUETAS_MES.get)
        day = st.number_input("Día del mes", 1, 31, 15)
        campaign = st.number_input("Llamadas en esta campaña (incluida esta)", 1, 70, 1)
        contactado = st.checkbox("Fue contactado en una campaña anterior")
        pdays = st.number_input("Días desde el último contacto anterior", 0, 999, 180, disabled=not contactado)
        previous = st.number_input("Contactos en campañas anteriores", 1, 300, 1, disabled=not contactado)
        poutcome = st.selectbox("Resultado de la campaña anterior", ["failure", "success", "other", DESCONOCIDO],
                                disabled=not contactado,
                                format_func=lambda p: {"failure": "No aceptó", "success": "Aceptó",
                                                       "other": "Otro"}.get(p, p))

    st.divider()

    def nulo(v):
        return np.nan if v == DESCONOCIDO else v

    cliente = pd.DataFrame([{
        "age": age, "job": nulo(job), "marital": marital, "education": nulo(education),
        "default": default, "balance": balance, "housing": housing, "loan": loan,
        "contact": nulo(contact), "day": day, "month": month, "campaign": campaign,
        "pdays": pdays if contactado else -1, "previous": previous if contactado else 0,
        "poutcome": nulo(poutcome) if contactado else np.nan,
    }])[columnas]
    score = float(pipeline.predict_proba(cliente)[:, 1][0])
    pct = percentil(score)

    m1, m2, m3 = st.columns(3)
    m1.metric("Puntaje del modelo", f"{score:.3f}")
    m2.metric("Posición en el ranking", f"Top {100 - pct:.0f} %",
              help="Comparado con los clientes del conjunto de validación.")
    m3.metric("Umbral de llamada", f"{umbral:.3f}", help="Puntaje que deja por encima al 20 % de los clientes.")
    if score >= umbral:
        st.success("**Recomendación: llamar.** El cliente está dentro del 20 % con mayor puntaje.")
    else:
        st.warning("**Recomendación: no priorizar.** El cliente queda fuera del 20 % con mayor puntaje.")
    st.caption("El puntaje sirve para ordenar clientes; todavía no está calibrado como probabilidad real.")

# --- Lista CSV --------------------------------------------------------------------------
with tab_lista:
    st.write(
        "Sube un CSV con una fila por cliente. Columnas requeridas: "
        + ", ".join(f"`{c}`" for c in columnas)
        + ". También se acepta el formato original de UCI (`day_of_week`, valores `unknown`); "
          "las columnas `duration` e `y` se ignoran si vienen."
    )
    archivo = st.file_uploader("Archivo CSV", type="csv")
    usar_ejemplo = st.toggle("Usar 200 clientes de ejemplo del conjunto de validación")

    lista = None
    if archivo is not None:
        lista = pd.read_csv(archivo)
    elif usar_ejemplo:
        _, val, _ = data.train_val_test_split(data.clean(data.load_raw()))
        lista = val.drop(columns=["y", "orden", "duration"]).head(200)

    if lista is not None:
        try:
            X = preparar_entrada(lista, columnas)
        except ValueError as e:
            st.error(str(e))
        else:
            resultado = lista.copy()
            resultado["puntaje"] = pipeline.predict_proba(X)[:, 1]
            resultado["llamar"] = np.where(resultado["puntaje"] >= umbral, "sí", "no")
            resultado = resultado.sort_values("puntaje", ascending=False)
            resultado.insert(0, "prioridad", range(1, len(resultado) + 1))
            primeras = ["prioridad", "puntaje", "llamar"]
            resultado = resultado[primeras + [c for c in resultado.columns if c not in primeras]]

            a, b = st.columns(2)
            a.metric("Clientes evaluados", f"{len(resultado):,}")
            b.metric("Recomendados para llamar", f"{(resultado['llamar'] == 'sí').sum():,}")
            st.dataframe(resultado, hide_index=True, width="stretch",
                         column_config={"puntaje": st.column_config.ProgressColumn(
                             "puntaje", min_value=0.0, max_value=1.0, format="%.3f")})
            st.download_button("Descargar lista priorizada", resultado.to_csv(index=False).encode("utf-8"),
                               "clientes_priorizados.csv", "text/csv")

# --- Información ------------------------------------------------------------------------
with tab_info:
    st.subheader("Cómo funciona")
    st.markdown(
        "**Usuario → formulario o CSV → preprocesamiento (`src/features.py`) → modelo → puntaje y recomendación**\n\n"
        "El preprocesamiento y el modelo forman un solo `Pipeline` de scikit-learn guardado en "
        "`models/modelo_preliminar.joblib`, de modo que la aplicación aplica exactamente las mismas "
        "transformaciones que se usaron al entrenar."
    )
    metricas = config.REPORTS_DIR / "metricas_validacion.csv"
    if metricas.exists():
        st.subheader("Desempeño en validación")
        tabla = pd.read_csv(metricas)
        st.dataframe(tabla[["modelo", "pr_auc", "roc_auc", "captura_top20", "lift_top20"]].round(3),
                     hide_index=True, width="stretch")
    st.subheader("Limitaciones")
    st.markdown(
        "- Entrenado con un solo banco portugués entre 2008 y 2010.\n"
        "- En una prueba temporal (entrenar con el pasado, evaluar en el periodo más reciente) el modelo "
        "capta solo 32-37 % de los suscriptores en el top 20 %, casi lo mismo que una regla simple.\n"
        "- El mes de la llamada pesa mucho y probablemente refleja el tipo de campaña, no una estacionalidad estable.\n"
        "- Las relaciones son asociaciones, no causas."
    )
