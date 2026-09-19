import streamlit as st
import numpy as np
import pandas as pd
import pickle

# 1. Configuración inicial (Debe ser obligatoriamente la primera línea de Streamlit)
st.set_page_config(page_title="Predicción Store Gamer", page_icon="🎮", layout="wide")

# 2. Inyección de CSS para la interfaz y el fondo
def set_gaming_bg():
    st.markdown(
         f"""
         <style>
         .stApp {{
             /* URL de ejemplo estilo Racing/F1. Cambie el link por su wallpaper de Gears of War si lo prefiere */
             background-image: url("https://images.unsplash.com/photo-1547394765-185e1e68f34e?q=80&w=2070");
             background-attachment: fixed;
             background-size: cover;
         }}
         /* Contenedores oscuros semitransparentes para garantizar la lectura */
         .stVerticalBlock, .stForm, [data-testid="stColumn"] {{
             background-color: rgba(15, 15, 15, 0.85) !important;
             border-radius: 10px;
             padding: 15px;
         }}
         h1, h2, h3, p, label, .st-emotion-cache-10trblm {{
             color: #FFFFFF !important;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
set_gaming_bg()

# 3. Carga del modelo (Caché para evitar bloqueos de I/O en cada recarga)
@st.cache_resource
def load_model():
    try:
        filename = 'modelo-tree.pkl'
        return pickle.load(open(filename, 'rb'))
    except FileNotFoundError:
        st.error("⚠️ Archivo 'modelo-tree.pkl' no encontrado. Verifique la ruta.")
        return None, None, None

modelo, min_max_scaler, variables = load_model()

# 4. Interfaz gráfica principal
st.title('🏎️ Predicción de Inversión en Videojuegos')
st.markdown("Ingrese el perfil del jugador para estimar su gasto.")

# Layout en columnas para mejor usabilidad
col1, col2 = st.columns(2)

with col1:
    Edad = st.slider('Edad', min_value=14, max_value=52, value=20, step=1)
    videojuego = st.selectbox('Videojuego', ["'Mass Effect'", "'Battlefield'", "'Fifa'", "'KOA: Reckoning'", "'Crysis'", "'Sim City'", "'Dead Space'", "'F1'"])
    Plataforma = st.selectbox('Plataforma', ["'Play Station'", "'Xbox'", "PC", "Otros"])

with col2:
    Sexo = st.selectbox('Sexo', ['Hombre', 'Mujer'])
    Consumidor_habitual = st.selectbox('Consumidor Habitual', ['True', 'False'])
    
    st.markdown("<br><br>", unsafe_allow_html=True) # Espaciado vertical
    # Botón explícito para aislar la ejecución
    ejecutar = st.button('🎮 Predecir Gasto', use_container_width=True)

# 5. Lógica de predicción (Solo corre si se presiona el botón)
if ejecutar and modelo is not None:
    # Construcción del dataframe inicial
    datos = [[Edad, videojuego, Plataforma, Sexo, Consumidor_habitual]]
    data = pd.DataFrame(datos, columns=['Edad', 'videojuego', 'Plataforma', 'Sexo', 'Consumidor_habitual'])
    
    # Preparación de variables (Dummies y reindexación)
    data_preparada = pd.get_dummies(data, columns=['videojuego', 'Plataforma', 'Sexo', 'Consumidor_habitual'], drop_first=False, dtype=int)
    data_preparada = data_preparada.reindex(columns=variables, fill_value=0)
    
    # Predicción (Omitimos el .transform de min_max_scaler por ser un árbol)
    Y_pred = modelo.predict(data_preparada)
    
    # Mostrar resultados con estilo de métrica de Streamlit
    st.markdown("---")
    st.subheader("💰 Resultado de la Predicción")
    st.metric(label="Gasto Estimado del Cliente", value=f"${Y_pred[0]:,.2f}")
    
    st.info("📌 **Nota técnica:** El modelo cuenta con un error porcentual (MAPE) del 5%.")
