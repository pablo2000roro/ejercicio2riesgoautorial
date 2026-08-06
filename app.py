import os
import json
import joblib
import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(
    page_title="Riesgo actuarial",
    layout="centered"
)

st.title("Predicción de riesgo actuarial - Pablo Rodriguez - PTI-0620")


# -----------------------------
# Cargar modelo y metadatos
# -----------------------------
@st.cache_resource
def cargar_modelo():

    pkl = (
        "kmeans_riesgo_actuarial.pkl"
        if os.path.exists("kmeans_riesgo_actuarial.pkl")
        else "kmeans_riesgo_actuarial(2).pkl"
    )

    meta = (
        "model_metadata.json"
        if os.path.exists("model_metadata.json")
        else "model_metadata(2).json"
    )

    modelo = joblib.load(pkl)

    with open(meta, encoding="utf-8") as f:
        metadata = json.load(f)

    return modelo, metadata


# -----------------------------
# Cargar base
# -----------------------------
@st.cache_data
def cargar_base():

    csv = (
        "insurance.csv"
        if os.path.exists("insurance.csv")
        else "insurance(2).csv"
    )

    return pd.read_csv(csv)


modelo, metadata = cargar_modelo()
df = cargar_base()

# -----------------------------
# Información del modelo
# -----------------------------
st.caption(metadata.get("proyecto", "Proyecto"))

mapa = {
    int(k): v
    for k, v in metadata["kmeans"]["mapa_riesgo"].items()
}


# -----------------------------
# Formulario
# -----------------------------
with st.form("datos"):

    col1, col2 = st.columns(2)

    age = col1.number_input(
        "Edad",
        min_value=18,
        max_value=100,
        value=35
    )

    sex = col2.selectbox(
        "Sexo",
        sorted(df["sex"].unique())
    )

    bmi = col1.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=28.0
    )

    children = col2.number_input(
        "Hijos",
        min_value=0,
        max_value=10,
        value=1
    )

    smoker = col1.selectbox(
        "Fumador",
        sorted(df["smoker"].unique())
    )

    region = col2.selectbox(
        "Región",
        sorted(df["region"].unique())
    )

    charges = st.number_input(
        "Cargos médicos estimados",
        min_value=0.0,
        max_value=100000.0,
        value=12000.0
    )

    enviar = st.form_submit_button("Evaluar")


# -----------------------------
# Predicción
# -----------------------------
if enviar:

    cliente = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
        "charges": charges
    }])

    try:

        cluster = int(modelo.predict(cliente)[0])

        riesgo = mapa.get(cluster, "No definido")

        st.success(f"Riesgo actuarial: {riesgo}")

        col1, col2 = st.columns(2)

        col1.metric("Cluster", cluster)
        col2.metric("Nivel de riesgo", riesgo)

    except Exception as e:

        st.error("Error al realizar la predicción.")
        st.exception(e)
        st.stop()

    # -----------------------------
    # Explicación con Groq
    # -----------------------------
    api_key = (
        st.secrets.get("GROQ_API_KEY")
        or os.getenv("GROQ_API_KEY")
    )

    if api_key:

        prompt = f"""
Eres un actuario profesional.

Analiza el siguiente cliente.

Edad: {age}
Sexo: {sex}
BMI: {bmi}
Hijos: {children}
Fumador: {smoker}
Región: {region}
Cargos médicos: {charges}

Resultado del modelo:

Cluster: {cluster}
Nivel de riesgo: {riesgo}

Explica brevemente qué significa el resultado y proporciona tres recomendaciones prudentes para reducir el riesgo.
"""

        try:

            client = OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1"
            )

            respuesta = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": "Eres un asesor actuarial profesional."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            st.info(respuesta.choices[0].message.content)

        except Exception as e:

            st.warning("No fue posible obtener la explicación con Groq.")
            st.exception(e)

    else:

        st.warning(
            "Agrega la variable GROQ_API_KEY en los secretos de Streamlit."
        )


# -----------------------------
# Base de datos
# -----------------------------
st.divider()

st.subheader("Vista previa de la base de datos")

st.dataframe(
    df.head(20),
    use_container_width=True
)
