import os
import io
import base64
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from openai import OpenAI
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas


st.set_page_config(page_title="Tablero Inteligente", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #ffe3ef 0%, #ffd6ea 45%, #fff0f7 100%);
}

h1, h2, h3 {
    color: #d63384;
}

section[data-testid="stSidebar"] {
    background: #fff0f7;
    border-right: 2px solid #f8b6d2;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {
    color: #b02a6b;
}

.stButton > button {
    background: linear-gradient(135deg, #ff4fa3, #ff7bbf);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    box-shadow: 0 8px 20px rgba(255, 79, 163, 0.25);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #ff3d98, #ff69b4);
    color: white;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.card {
    background: rgba(255,255,255,0.78);
    border: 2px solid #f6b2d0;
    border-radius: 20px;
    padding: 18px 20px;
    box-shadow: 0 10px 24px rgba(214, 51, 132, 0.10);
    margin-bottom: 16px;
}

.canvas-card {
    background: rgba(255,255,255,0.72);
    border: 2px solid #f6b2d0;
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 12px 30px rgba(214, 51, 132, 0.12);
    backdrop-filter: blur(10px);
}

.result-card {
    background: white;
    border: 2px solid #f3b3d1;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 10px 24px rgba(214, 51, 132, 0.10);
    margin-top: 16px;
}

.small-note {
    color: #9c4674;
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)


def canvas_to_pil(image_data):
    arr = np.array(image_data)
    return Image.fromarray(arr.astype("uint8"), "RGBA")


def pil_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@st.cache_resource
def load_digit_model():
    return tf.keras.models.load_model("model/handwritten.h5")


def predict_digit(image):
    model = load_digit_model()
    image = ImageOps.grayscale(image)
    img = image.resize((28, 28))
    img = np.array(img, dtype="float32") / 255.0
    img = img.reshape((1, 28, 28, 1))
    pred = model.predict(img, verbose=0)
    return int(np.argmax(pred[0]))


st.title("Tablero Inteligente")
st.markdown("""
<div class="card">
    <h3 style="margin-top:0;">Una sola app para todo</h3>
    <p class="small-note">
        Usa las pestañas para cambiar entre tablero libre, análisis de bocetos con IA y reconocimiento de dígitos.
    </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "Tablero libre",
    "Analizar boceto con IA",
    "Reconocer dígitos"
])

with tab1:
    st.subheader("Tablero para dibujo")

    left, right = st.columns([1, 2])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        canvas_width = st.slider("Ancho del tablero", 300, 700, 500, 50, key="free_w")
        canvas_height = st.slider("Alto del tablero", 200, 600, 300, 50, key="free_h")
        drawing_mode = st.selectbox(
            "Herramienta de dibujo",
            ("freedraw", "line", "rect", "circle", "transform", "polygon", "point"),
            key="free_mode"
        )
        stroke_width = st.slider("Ancho de línea", 1, 30, 15, key="free_stroke")
        stroke_color = st.color_picker("Color de trazo", "#ff2e88", key="free_color")
        bg_color = st.color_picker("Color de fondo", "#ffffff", key="free_bg")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="canvas-card">', unsafe_allow_html=True)
        st_canvas(
            fill_color="rgba(255, 105, 180, 0.18)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            height=canvas_height,
            width=canvas_width,
            drawing_mode=drawing_mode,
            key=f"free_canvas_{canvas_width}_{canvas_height}",
        )
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.subheader("Analizar boceto con IA")

    left, right = st.columns([1, 2])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("Dibuja un boceto y deja que la IA lo describa brevemente en español.")
        api_key_input = st.text_input("Ingresa tu API key", type="password")
        analyze_button = st.button("Analizar imagen", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="canvas-card">', unsafe_allow_html=True)
        ia_canvas = st_canvas(
            fill_color="rgba(255, 105, 180, 0.18)",
            stroke_width=5,
            stroke_color="#000000",
            background_color="#FFFFFF",
            height=300,
            width=420,
            drawing_mode="freedraw",
            key="ia_canvas",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    api_key = api_key_input or st.secrets.get("OPENAI_API_KEY", "")

    if analyze_button:
        if ia_canvas.image_data is None:
            st.warning("Primero dibuja algo en el canvas.")
        elif not api_key:
            st.warning("Ingresa tu API key.")
        else:
            try:
                client = OpenAI(api_key=api_key)
                input_image = canvas_to_pil(ia_canvas.image_data)
                base64_image = pil_to_base64(input_image)

                with st.spinner("Analizando..."):
                    response = client.responses.create(
                        model="gpt-4.1-mini",
                        input=[{
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "Describe en español y brevemente el dibujo."},
                                {"type": "input_image", "image_url": f"data:image/png;base64,{base64_image}"}
                            ]
                        }]
                    )

                result_text = getattr(response, "output_text", "").strip()

                st.markdown(
                    f"""
                    <div class="result-card">
                        <h3 style="margin-top:0; color:#d63384;">Resultado</h3>
                        <p style="margin-bottom:0;">{result_text if result_text else "No hubo respuesta."}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Ocurrió un error: {e}")

with tab3:
    st.subheader("Reconocimiento de dígitos escritos a mano")

    left, center, right = st.columns([1, 1.2, 1])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        digit_stroke = st.slider("Ancho de línea del dígito", 1, 30, 15, key="digit_stroke")
        predict_button = st.button("Predecir dígito", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with center:
        st.markdown('<div class="canvas-card">', unsafe_allow_html=True)
        digit_canvas = st_canvas(
            fill_color="rgba(255, 105, 180, 0.18)",
            stroke_width=digit_stroke,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=220,
            width=220,
            drawing_mode="freedraw",
            key="digit_canvas",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("Dibuja un solo dígito, preferiblemente centrado y grande.")
        st.markdown('</div>', unsafe_allow_html=True)

    if predict_button:
        if digit_canvas.image_data is None:
            st.warning("Por favor dibuja un dígito en el canvas.")
        else:
            try:
                digit_image = canvas_to_pil(digit_canvas.image_data)
                result = predict_digit(digit_image)

                st.markdown(
                    f"""
                    <div class="result-card">
                        <h2 style="margin:0; color:#d63384;">El dígito es: {result}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"No se pudo cargar el modelo o hacer la predicción: {e}")

with st.sidebar:
    st.title("Acerca de")
    st.write("Esta app reúne tres herramientas en una sola interfaz.")
    st.write("1. Tablero libre")
    st.write("2. Análisis de bocetos con IA")
    st.write("3. Reconocimiento de dígitos")
