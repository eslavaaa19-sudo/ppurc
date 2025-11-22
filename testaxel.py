import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go

# ---------- CONFIG ----------
st.set_page_config(page_title="Dashboard Socavones - López Portillo", layout="wide", initial_sidebar_state="expanded")

# Ruta local de la imagen que subiste (se mostrará en la cabecera)
HEADER_IMAGE_PATH = "/mnt/data/6a9890b1-5dc9-4da8-9ed2-6f7074ab4d23.png"

# ---------- ESTILOS (dark / tarjetas) ----------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b0f1a;
        color: #dbe7ff;
    }
    .card {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border-radius: 10px;
        padding: 14px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.6);
        border: 1px solid rgba(255,255,255,0.03);
    }
    .kpi {
        font-size: 22px;
        font-weight: 700;
    }
    .kpi-sub {
        font-size: 13px;
        color: #9fb0ff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- CABECERA ----------
col1, col2 = st.columns([1, 3])
with col1:
    try:
        img = Image.open(HEADER_IMAGE_PATH)
        st.image(img, use_column_width=True)
    except Exception as e:
        st.write("")  # si no carga la imagen, se sigue
with col2:
    st.markdown("<h1 style='color:#e6f0ff'>Dashboard de Socavones — López Portillo, Iztapalapa</h1>", unsafe_allow_html=True)
    st.markdown("<div style='color:#9fb0ff'>Prototipo: visualización + modelo ML para estimar riesgo de socavones. Datos de ejemplo sintético.</div>", unsafe_allow_html=True)

st.write("")  # espacio

# ---------- DATOS DE EJEMPLO (sintético) ----------
# Creamos un dataset sintético con composición similar a lo que usarías:
np.random.seed(42)
n = 200
months = np.random.choice(pd.date_range("2024-01-01", periods=24, freq='M').strftime('%Y-%m'), size=n)
tipo_suelo = np.random.choice(['arcilla','arena','franco','limosa'], size=n, p=[0.35,0.3,0.2,0.15])
humedad_suelo = np.round(np.random.normal(12, 4, size=n).clip(1,40),1)
lluvia_mm = np.round(np.random.exponential(80, size=n)).astype(int)
nivel_freatico = np.round(np.random.normal(2.0, 0.8, size=n).clip(0.1,6),2)
carga_trafico = np.random.randint(0, 10, size=n)
antiguedad = np.random.randint(1, 60, size=n)
pendiente = np.round(np.random.exponential(1.5, size=n),2)

# Generar etiqueta sintética (probabilística)
score = (
    0.03*humedad_suelo + 
    0.002*lluvia_mm + 
    0.15*(tipo_suelo == 'arcilla').astype(int) + 
    0.08*(nivel_freatico) + 
    0.02*carga_trafico + 
    0.01*antiguedad + 
    0.05*pendiente
)
prob_socavon = 1 / (1 + np.exp(- (score - score.mean())))
socavon = (prob_socavon > np.quantile(prob_socavon, 0.7)).astype(int)  # ~30% casos positivos

df = pd.DataFrame({
    'mes': months,
    'tipo_suelo': tipo_suelo,
    'humedad_suelo': humedad_suelo,
    'lluvia_mm': lluvia_mm,
    'nivel_freatico': nivel_freatico,
    'carga_trafico': carga_trafico,
    'antiguedad': antiguedad,
    'pendiente': pendiente,
    'socavon': socavon
})

# ---------- SIDEBAR: Cargar CSV opcional + Entrenamiento ----------
st.sidebar.header("Datos & Modelo")
upload = st.sidebar.file_uploader("Sube tu CSV (opcional). Columna 'socavon' obligatoria", type=['csv'])
use_sample = st.sidebar.button("Usar datos de ejemplo")
if upload:
    try:
        df_upload = pd.read_csv(upload)
        st.sidebar.success("CSV cargado.")
        # re-asignar df si la estructura es ok
        if 'socavon' in df_upload.columns:
            df = df_upload.copy()
        else:
            st.sidebar.error("El CSV debe contener la columna 'socavon'.")
    except Exception as e:
        st.sidebar.error("Error leyendo CSV: " + str(e))

st.sidebar.markdown("---")
train_button = st.sidebar.button("Entrenar modelo (RandomForest)")
st.sidebar.markdown("Modelo entrena con todas las features numéricas + tipo_suelo (one-hot).")

# ---------- ENTRENAR MODELO ----------
model = None
if train_button:
    st.sidebar.info("Entrenando modelo...")
    X = df.drop(columns=['socavon','mes'], errors='ignore')
    y = df['socavon']
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
    preproc = ColumnTransformer([
        ('num', MinMaxScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])
    model = Pipeline([
        ('prep', preproc),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    st.sidebar.success(f"Modelo entrenado — exactitud: {acc*100:.1f}%")
    # guardar en session para uso en predicción
    st.session_state['model'] = model
    st.session_state['model_cols'] = X.columns.tolist()

# Si ya hay modelo en sesión (por haber entrenado antes)
if 'model' in st.session_state:
    model = st.session_state['model']

# ---------- KPIs (tarjetas) ----------
total_registros = len(df)
total_socavones = int(df['socavon'].sum())
porcentaje_socavon = total_socavones / total_registros if total_registros>0 else 0
prob_media = float(prob_socavon.mean())  # de datos sintéticos

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
with kpi_col1:
    st.markdown('<div class="card"><div class="kpi">{}<span style="font-size:14px"> registros</span></div><div class="kpi-sub">Total de observaciones</div></div>'.format(total_registros), unsafe_allow_html=True)
with kpi_col2:
    st.markdown('<div class="card"><div class="kpi">{:.1f}%</div><div class="kpi-sub">Promedio prob. socavón</div></div>'.format(porcentaje_socavon*100), unsafe_allow_html=True)
with kpi_col3:
    st.markdown('<div class="card"><div class="kpi">{}</div><div class="kpi-sub">Socavones detectados</div></div>'.format(total_socavones), unsafe_allow_html=True)
with kpi_col4:
    st.markdown('<div class="card"><div class="kpi">{:.1f}</div><div class="kpi-sub">Prob. media (score sintético)</div></div>'.format(prob_media), unsafe_allow_html=True)

st.write("")  # espacio

# ---------- GRÁFICAS PRINCIPALES ----------
# 1) Total por mes (línea)
st.markdown("<div class='card'><h3 style='margin-bottom:6px'>Socavones por mes</h3>", unsafe_allow_html=True)
soc_por_mes = df.groupby('mes').agg(total=('socavon','sum')).reset_index().sort_values('mes')
fig_line = px.line(soc_por_mes, x='mes', y='total', markers=True)
fig_line.update_layout(template='plotly_dark', margin=dict(l=20,r=20,t=20,b=20), height=320)
st.plotly_chart(fig_line, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

cols = st.columns([2,1])
with cols[0]:
    # 2) Barras: métricas agregadas (ej. humedad promedio por tipo de suelo)
    st.markdown("<div class='card'><h3 style='margin-bottom:6px'>Humedad media por tipo de suelo</h3>", unsafe_allow_html=True)
    hum_by_soil = df.groupby('tipo_suelo')['humedad_suelo'].mean().reset_index().sort_values('humedad_suelo', ascending=True)
    fig_bar = px.bar(hum_by_soil, x='humedad_suelo', y='tipo_suelo', orientation='h')
    fig_bar.update_layout(template='plotly_dark', margin=dict(l=20,r=20,t=20,b=20), height=320)
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with cols[1]:
    # 3) Dona: distribución de tipos de suelo
    st.markdown("<div class='card'><h3 style='margin-bottom:6px'>Distribución de tipos de suelo</h3>", unsafe_allow_html=True)
    counts = df['tipo_suelo'].value_counts().reset_index()
    counts.columns = ['tipo_suelo','count']
    fig_pie = px.pie(counts, values='count', names='tipo_suelo', hole=0.5)
    fig_pie.update_layout(template='plotly_dark', margin=dict(l=20,r=20,t=20,b=20), height=320, showlegend=True)
    st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")  # espacio

# ---------- TABLA RESUMEN + DETALLES ----------
st.markdown("<div class='card'><h3 style='margin-bottom:6px'>Resumen por categoría</h3>", unsafe_allow_html=True)
summary = df.groupby('tipo_suelo').agg(
    registros=('socavon','count'),
    socavones=('socavon','sum'),
    prob_media=('humedad_suelo','mean')  # aquí solo como ejemplo
).reset_index()
st.dataframe(summary.style.set_properties(**{'background-color': '#071026', 'color':'#dbe7ff'}), height=220)
st.markdown("</div>", unsafe_allow_html=True)

# ---------- PANEL DE PREDICCION MANUAL ----------
st.markdown("<div class='card'><h3 style='margin-bottom:6px'>Predicción manual</h3>", unsafe_allow_html=True)
with st.form("predict_form"):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        in_tipo = st.selectbox("Tipo de suelo", options=sorted(df['tipo_suelo'].unique().tolist()))
        in_humedad = st.number_input("Humedad del suelo", value=float(df['humedad_suelo'].median()))
        in_lluvia = st.number_input("Lluvia (mm)", value=int(df['lluvia_mm'].median()))
    with col_b:
        in_nivel = st.number_input("Nivel freático", value=float(df['nivel_freatico'].median()))
        in_trafico = st.number_input("Carga de tráfico (0-10)", min_value=0, max_value=20, value=int(df['carga_trafico'].median()))
    with col_c:
        in_ant = st.number_input("Antigüedad (años)", value=int(df['antiguedad'].median()))
        in_pend = st.number_input("Pendiente", value=float(df['pendiente'].median()))
    predict_btn = st.form_submit_button("Predecir riesgo")

if predict_btn:
    # Comprobar que hay modelo (o entrenar rápido si no lo hay)
    if model is None and 'model' in st.session_state:
        model = st.session_state['model']
    if model is None:
        # entrenamiento rápido con los datos actuales
        X = df.drop(columns=['socavon','mes'], errors='ignore')
        y = df['socavon']
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()
        preproc = ColumnTransformer([
            ('num', MinMaxScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ])
        model = Pipeline([
            ('prep', preproc),
            ('clf', RandomForestClassifier(n_estimators=150, random_state=1))
        ])
        model.fit(X, y)
        st.info("Modelo entrenado rápidamente con datos actuales (solo para demo).")
        st.session_state['model'] = model

    # crear df de entrada
    df_input = pd.DataFrame([{
        'tipo_suelo': in_tipo,
        'humedad_suelo': in_humedad,
        'lluvia_mm': in_lluvia,
        'nivel_freatico': in_nivel,
        'carga_trafico': in_trafico,
        'antiguedad': in_ant,
        'pendiente': in_pend
    }])
    prob = model.predict_proba(df_input)[0][1]
    st.markdown(f"<div style='font-size:20px; font-weight:700'>Probabilidad de socavón: {prob*100:.2f}%</div>", unsafe_allow_html=True)
    if prob > 0.6:
        st.error("⚠️ Riesgo ALTO")
    elif prob > 0.3:
        st.warning("⚠️ Riesgo MEDIO")
    else:
        st.success("✔️ Riesgo BAJO")

st.markdown("</div>", unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("<div style='padding-top:10px; color:#9fb0ff'>Prototipo — Datos sintéticos. Para uso real se requieren datos históricos, validación y revisión por ingenieros geotécnicos.</div>", unsafe_allow_html=True)
