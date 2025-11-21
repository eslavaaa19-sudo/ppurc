import streamlit as st
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# ---------------- CONFIGURACIÓN DE LA APP ----------------
st.set_page_config(
    page_title="Predicción de Socavones",
    layout="wide",
    page_icon="🕳️"
)

# ---------------- ENCABEZADO ----------------
st.markdown("""
<div style='background-color:#003366; padding:18px; border-radius:6px;'>
<h2 style='color:white; text-align:center;'>🕳️ Sistema de Monitoreo y Predicción de Socavones</h2>
<p style='color:#cfd9e7; text-align:center;'>Análisis geotécnico con datos en línea</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------- CARGA DE BASE DE DATOS ----------------
url = "https://raw.githubusercontent.com/agumax/SocavonesDB/main/socavones.csv"

try:
    response = requests.get(url)
    df = pd.read_csv(pd.compat.StringIO(response.text))
    data_status = "🟢 Datos cargados desde la nube"
except:
    st.error("❌ Error: No se pudo cargar el archivo CSV en línea.")
    st.stop()

# ---------------- PREPARACIÓN DEL MODELO ----------------
label = LabelEncoder()
df["suelo"] = label.fit_transform(df["suelo"])

X = df[["suelo", "fugas_agua", "lluvia_mm", "vibraciones"]]
y = df["socavon"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

modelo = RandomForestClassifier(n_estimators=200, random_state=42)
modelo.fit(X_train, y_train)

# ---------------- LAYOUT 3 COLUMNAS ----------------
col1, col2, col3 = st.columns([1.25, 1.25, 1])

# ======================================================================
# COLUMNA 1 – DATOS
# ======================================================================
with col1:
    st.markdown("## 📡 Datos")
    st.markdown(f"**Estado de conexión:** {data_status}")
    st.markdown("**Fuente:** CSV público en GitHub")
    st.markdown("---")

    st.markdown("### 🔍 Vista preliminar")
    st.dataframe(df.head())

    st.markdown("---")

    st.markdown("### ⚙️ Modelo")
    st.write("""
    - Modelo: **Random Forest**
    - Árboles: **200**
    - Variables usadas:
        - Tipo de suelo
        - Fugas de agua
        - Lluvia (mm)
        - Vibraciones
    """)

# ======================================================================
# COLUMNA 2 – PREDICCIÓN
# ======================================================================
with col2:
    st.markdown("## 🧮 Predicción")

    suelo_input = st.selectbox("Tipo de suelo", label.classes_)
    fugas_input = st.selectbox("Fugas de agua", [0, 1])
    lluvia_input = st.slider("Lluvia actual (mm)", 0, 300, 50)
    vibraciones_input = st.slider("Vibraciones del terreno", 0.0, 5.0, 1.0)

    if st.button("🔵 Calcular riesgo"):
        valores = {
            "suelo": label.transform([suelo_input])[0],
            "fugas_agua": fugas_input,
            "lluvia_mm": lluvia_input,
            "vibraciones": vibraciones_input
        }

        nuevo_df = pd.DataFrame([valores])
        prob = modelo.predict_proba(nuevo_df)[0][1]

        st.markdown("---")
        st.markdown("### 🎯 Resultado")
        st.metric("Probabilidad estimada de socavón", f"{prob*100:.2f}%")

        # Indicador de riesgo
        if prob < 0.30:
            color = "🟢 Riesgo bajo"
        elif prob < 0.60:
            color = "🟡 Riesgo medio"
        else:
            color = "🔴 Riesgo alto"

        st.markdown(f"## {color}")

# ======================================================================
# COLUMNA 3 – ESTADÍSTICAS
# ======================================================================
with col3:
    st.markdown("## 📊 Estadísticas del dataset")

    st.markdown("### Casos de socavón")

    counts = df["socavon"].value_counts()
    fig, ax = plt.subplots()
    ax.bar(["No", "Sí"], counts)
    ax.set_title("Distribución de casos")
    st.pyplot(fig)

    st.markdown("---")

    st.markdown("### Relación lluvia / socavón")

    fig2, ax2 = plt.subplots()
    colores = df["socavon"].map({0: "blue", 1: "red"})
    ax2.scatter(df["lluvia_mm"], df["socavon"], c=colores)
    ax2.set_xlabel("Lluvia (mm)")
    ax2.set_ylabel("Socavón (0/1)")
    st.pyplot(fig2)

# ---------------- PIE DE PÁGINA ----------------
st.markdown("""
<br>
<div style='text-align:center; color:gray; font-size:12px;'>
Sistema de Predicción de Socavones · Streamlit · 2025
</div>
""", unsafe_allow_html=True)
