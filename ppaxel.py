import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Predicción de Socavones - López Portillo, Iztapalapa",
    layout="wide"
)

st.title("Predicción de Socavones en la Colonia López Portillo, Iztapalapa")
st.write("Análisis estadístico enfocado en evaluar el riesgo de formación de socavones en la colonia López Portillo, dentro de la alcaldía Iztapalapa.")

# ------------------------------
# 1. CARGAR EXCEL
# ------------------------------
ruta_excel = "Trabajo_Estadistica_Tabla.xlsx"
df = pd.read_excel(ruta_excel).dropna(how="all")

df = df.rename(columns={
    "Precipitación (mm)": "precipitacion",
    "Densidad de población (hab/km²)": "densidad",
    "Tráfico vehicular (vehículos/día)": "trafico",
    "Extracción de agua (m³/año)": "extraccion",
    "Profundidad de socavón (m)": "profundidad",
    "Nivel freático (m bajo tierra)": "freatico",
    "Humedad del suelo (%)": "humedad",
    "Socavones reportados": "socavones"
})

st.subheader("Datos utilizados (Base de la colonia López Portillo, Iztapalapa)")
st.dataframe(df, use_container_width=True)

# ------------------------------
# 2. GRÁFICAS AVANZADAS
# ------------------------------
st.header("Visualización de datos para López Portillo")

col1, col2 = st.columns(2)

# Gráfica 1: Lluvia vs Socavones
with col1:
    st.subheader("Relación entre precipitación y socavones")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="precipitacion", y="socavones", ax=ax)
    ax.set_xlabel("Precipitación (mm)")
    ax.set_ylabel("Socavones reportados")
    st.pyplot(fig)

# Gráfica 2: Mapa de Calor
with col2:
    st.subheader("Mapa de correlación de variables")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ------------------------------
# 3. MODELO SIMPLIFICADO
# ------------------------------
st.header("Entrenamiento del modelo predictivo para López Portillo")

features = ["precipitacion", "densidad", "trafico",
            "extraccion", "profundidad", "freatico", "humedad"]

df_modelo = df.dropna(subset=features + ["socavones"])

X = df_modelo[features]
y = df_modelo["socavones"]

# Estandarización
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Modelos
modelo_log = LogisticRegression()
modelo_arbol = DecisionTreeClassifier(max_depth=3)

modelo_log.fit(X_scaled, y)
modelo_arbol.fit(X, y)

st.success("Modelos entrenados exitosamente con datos de la colonia López Portillo.")

# ------------------------------
# 4. INTERFAZ DE PREDICCIÓN
# ------------------------------
st.header("Simulación de riesgo de socavón en López Portillo")

c1, c2, c3 = st.columns(3)

lluvia = c1.slider("Precipitación estimada (mm)", 0, 1500, 700)
densidad = c1.slider("Densidad poblacional (hab/km²)", 10000, 50000, 16000)
trafico = c1.slider("Tráfico vehicular (vehículos por día)", 10000, 60000, 25000)

extraccion = c2.slider("Extracción de agua subterránea (m³/año)", 100000, 900000, 500000)
profundidad = c2.slider("Profundidad estimada del hundimiento (m)", 0, 30, 5)
freatico = c2.slider("Nivel freático (m bajo tierra)", 20, 80, 50)

humedad = c3.slider("Humedad del suelo (%)", 10, 80, 40)

if st.button("Evaluar riesgo"):
    usuario = [[lluvia, densidad, trafico, extraccion,
                profundidad, freatico, humedad]]

    pred_log = modelo_log.predict(scaler.transform(usuario))[0]
    pred_tree = modelo_arbol.predict(usuario)[0]

    st.subheader("Resultados de la evaluación")

    colA, colB = st.columns(2)

    with colA:
        st.write("Modelo: Regresión Logística")
        if pred_log >= 1:
            st.error("Riesgo detectado de formación de socavón en López Portillo")
        else:
            st.success("Sin indicios significativos de riesgo")

    with colB:
        st.write("Modelo: Árbol de Decisión")
        if pred_tree >= 1:
            st.error("Riesgo detectado de formación de socavón en López Portillo")
        else:
            st.success("Sin indicios significativos de riesgo")