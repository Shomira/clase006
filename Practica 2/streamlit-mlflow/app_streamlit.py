import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn

st.set_page_config(
    page_title="Predicción Bank Marketing",
    layout="centered"
)
st.title("Predicción de Aprobación de Estudiantes 👩🏻‍🎓👨🏻‍🎓")

st.write(
    "Esta aplicación carga un Pipeline registrado en MLflow. "
    "Por eso se ingresan las columnas originales del dataset, no las variables encodeadas."
)

# Conexión al servidor MLflow
mlflow.set_tracking_uri("http://localhost:9090") #("http://127.0.0.1:5000")

# Cambie la versión si MLflow registra una nueva versión del modelo:
# models:/clase06/1, models:/clase06/2, etc.
MODEL_URI = "models:/estudiantes_arboles/10"

@st.cache_resource
def cargar_modelo():
    return mlflow.sklearn.load_model(MODEL_URI)

model = cargar_modelo()

st.sidebar.header("Configuración")
st.sidebar.write(f"Modelo cargado: `{MODEL_URI}`")
st.set_page_config(page_title="Predicción de Estudiantes", layout="centered")



st.subheader("Datos del Estudiante")


col1, col2 = st.columns(2)

with col1:
    carrera = st.selectbox("Carrera", ["Industrial", "Arquitectura", "Economia", "Computacion", "Medicina", "Derecho"])
    modalidad = st.selectbox("Modalidad", ["Presencial", "Virtual", "Hibrida"])
    beca = st.selectbox("Beca", ["Si", "No"])
    edad = st.number_input("Edad", min_value=18, max_value=40, step=1)

with col2:
    promedio = st.number_input("Promedio", min_value=0.0, max_value=10.0, step=0.1)
    asistencias = st.number_input("Asistencias (%)", min_value=0, max_value=100, step=1)

# Convertir datos a DataFrame para el modelo
datos = pd.DataFrame({
        "carrera": [carrera],
        "modalidad": [modalidad],
        "beca": [beca],
        "edad": [edad],
        "promedio": [promedio],
        "asistencias": [asistencias]
    })
# Botón para predecir
if st.button("Predecir"):
    prediccion = model.predict(datos)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(datos)[0]
        prob_no = proba[0]
        prob_si = proba[1]
    else:
        prob_no = None
        prob_si = None

    if prediccion == 1:
        st.success("El modelo predice que el estudiante aprobará. 🎓🤓")
    else:
        st.warning("El modelo predice que el estudiante NO aprobará 🥹")

    if prob_si is not None:
        st.write(f"Probabilidad de NO aprobar : {prob_no:.4f}")
        st.write(f"Probabilidad de SÍ aprobar: {prob_si:.4f}")
   
   