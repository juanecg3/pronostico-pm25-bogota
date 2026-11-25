"""Prototipo Streamlit para consultar y simular pronósticos PM2.5 a 24 h."""
from datetime import date
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.modelo_pm25 import FEATURES, PROJECT, cargar_parametros_campeon, cargar_particiones, entrenar_campeon

st.set_page_config(page_title="Pronóstico PM2.5 | Kennedy", page_icon="🌫️", layout="wide")


@st.cache_resource
def cargar_sistema():
    datos, train, test = cargar_particiones()
    parametros = cargar_parametros_campeon()
    modelo = entrenar_campeon(train, parametros)
    return datos, train, test, modelo, parametros


def clasificar_referencia(valor):
    # Umbrales orientativos: el IBOCA oficial requiere NowCast ponderado de 12 h.
    if valor <= 12.0:
        return "Bajo (referencia de concentración)", "success"
    if valor <= 35.4:
        return "Moderado (referencia de concentración)", "info"
    if valor <= 55.4:
        return "Regular (referencia de concentración)", "warning"
    if valor <= 151.2:
        return "Alto (referencia de concentración)", "error"
    return "Peligroso (referencia de concentración)", "error"


def mostrar_referencia(valor):
    texto, tipo = clasificar_referencia(valor)
    aviso = (
        f"**{texto}: {valor:.1f} µg/m³.** Referencia visual, no IBOCA oficial: para PM₂.₅, "
        "el IBOCA usa una media NowCast ponderada de 12 horas."
    )
    getattr(st, tipo)(aviso)


try:
    datos, train, test, modelo, parametros = cargar_sistema()
except Exception as exc:
    st.error(f"No se pudo cargar el modelo o los datos: {exc}")
    st.stop()

st.title("Pronóstico experimental de PM₂.₅ · Estación Kennedy")
st.caption(f"Modelo a 24 h; entrenado con objetivos enero–mayo. Parámetros congelados: {parametros}.")
st.warning("Prototipo académico; no es una alerta oficial y no sustituye la información de la Secretaría Distrital de Ambiente.")
modo = st.sidebar.radio("Modo de uso", ["Consultar junio", "Simular valores manuales"])
tab_pred, tab_xai = st.tabs(["Pronóstico", "Explicabilidad"])

with tab_pred:
    if modo == "Consultar junio":
        opciones = test.reset_index(drop=True)
        etiquetas = opciones["fecha_hora"].dt.strftime("%Y-%m-%d %H:%M")
        i = st.sidebar.selectbox("Fecha/hora de origen (t)", range(len(opciones)), format_func=lambda j: etiquetas.iloc[j])
        fila = opciones.iloc[int(i)]
        pred = float(modelo.predict(fila[FEATURES].to_frame().T)[0])
        real = float(fila["target_24h"])
        actual = float(fila["pm25_original"])
        c1, c2, c3 = st.columns(3)
        c1.metric("PM₂.₅ observado en t", f"{actual:.1f} µg/m³")
        c2.metric("Pronóstico Random Forest · t+24 h", f"{pred:.1f} µg/m³")
        c3.metric("Real en t+24 h", f"{real:.1f} µg/m³", f"Error absoluto: {abs(real-pred):.1f} µg/m³")
        mostrar_referencia(pred)

        origen = fila["fecha_hora"]
        hist = datos.loc[
            datos["fecha_hora"].between(origen - pd.Timedelta(hours=24), origen),
            ["fecha_hora", "pm25_original"],
        ].dropna()
        futuro = origen + pd.Timedelta(hours=24)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(hist["fecha_hora"], hist["pm25_original"], color="#252525", marker=".", label="PM₂.₅ observado")
        ax.scatter([futuro], [pred], color="#D62728", s=65, label="Pronóstico t+24 h", zorder=3)
        ax.scatter([futuro], [real], color="#277C78", marker="x", s=70, label="Real t+24 h", zorder=3)
        ax.axvline(origen, color="#276FBF", ls="--", lw=1, label="Origen t")
        ax.set_ylabel("PM₂.₅ (µg/m³)"); ax.set_xlabel("Fecha y hora")
        ax.grid(axis="y", alpha=0.25); ax.legend(ncols=2, frameon=False)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    else:
        st.sidebar.subheader("Variables de entrada")
        fecha = st.sidebar.date_input("Fecha de referencia", value=date(2026, 6, 18), min_value=date(2026, 1, 1), max_value=date(2027, 12, 31))
        hora = st.sidebar.slider("Hora local", 0, 23, 8)
        actual = st.sidebar.slider("PM₂.₅ actual (µg/m³)", 0.0, 150.0, 18.0)
        lag1 = st.sidebar.slider("PM₂.₅ hace 1 hora", 0.0, 150.0, 17.5)
        lag24 = st.sidebar.slider("PM₂.₅ hace 24 horas", 0.0, 150.0, 20.0)
        roll3 = st.sidebar.slider("Promedio de las 3 últimas lecturas", 0.0, 150.0, 18.0)
        roll6 = st.sidebar.slider("Promedio de las 6 últimas lecturas", 0.0, 150.0, 19.0)
        X = pd.DataFrame([{
            "pm25_original": actual, "pm25_lag1": lag1, "pm25_lag24": lag24,
            "pm25_roll3": roll3, "pm25_roll6": roll6, "hora": hora,
            "dia_semana": fecha.weekday(),
        }], columns=FEATURES)
        pred = float(modelo.predict(X)[0])
        c1, c2 = st.columns(2)
        c1.metric("PM₂.₅ ingresado en t", f"{actual:.1f} µg/m³")
        c2.metric("Pronóstico Random Forest · t+24 h", f"{pred:.1f} µg/m³")
        mostrar_referencia(pred)
        nombres = ["t−24", "media 6 lecturas", "media 3 lecturas", "t−1", "t", "pronóstico t+24"]
        vals = [lag24, roll6, roll3, lag1, actual, pred]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(nombres, vals, marker="o", color="#276FBF")
        ax.scatter([nombres[-1]], [vals[-1]], color="#D62728", s=75, zorder=3)
        ax.set_ylabel("PM₂.₅ (µg/m³)"); ax.set_title("Entradas resumidas y pronóstico")
        ax.grid(axis="y", alpha=0.25); fig.tight_layout()
        st.pyplot(fig); plt.close(fig)
        st.caption("La figura ordena los valores ingresados; no representa mediciones horarias intermedias.")

with tab_xai:
    archivo_img = PROJECT / "experiments" / "importancia variables.png"
    archivo_json = PROJECT / "experiments" / "resultados explicabilidad.json"
    if archivo_img.exists() and archivo_json.exists():
        st.image(str(archivo_img), caption="Aumento del MAE al permutar cada variable")
        xai = json.loads(archivo_json.read_text(encoding="utf-8"))
        st.json(xai.get("ranking_importancia", []))
    else:
        st.info("Aún no hay resultados XAI. Ejecute `python src/explicabilidad.py` desde la carpeta del proyecto.")

st.caption("Las variables usan lecturas originales observadas, igual que el modelo de la evaluación final.")
