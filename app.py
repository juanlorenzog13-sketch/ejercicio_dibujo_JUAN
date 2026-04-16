import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

st.set_page_config(page_title="Contador de Dedos", layout="centered")
st.title("Contador de Dedos")
st.write("Toma o sube una foto de una mano y la app intentará decir cuántos dedos están levantados.")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def count_fingers(hand_landmarks, hand_label, image_width, image_height):
    landmarks = hand_landmarks.landmark

    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Pulgar
    # Ajuste según mano izquierda o derecha
    if hand_label == "Right":
        if landmarks[tips_ids[0]].x < landmarks[tips_ids[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)
    else:
        if landmarks[tips_ids[0]].x > landmarks[tips_ids[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)

    # Índice, medio, anular, meñique
    for tip_id in tips_ids[1:]:
        if landmarks[tip_id].y < landmarks[tip_id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("O toma una foto")

image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file)
elif camera_image is not None:
    image = Image.open(camera_image)

if image is not None:
    image = image.convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:

        results = hands.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            annotated = image_np.copy()

            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_label = "Right"
                if results.multi_handedness:
                    hand_label = results.multi_handedness[idx].classification[0].label

                finger_count = count_fingers(
                    hand_landmarks,
                    hand_label,
                    annotated.shape[1],
                    annotated.shape[0]
                )

                mp_drawing.draw_landmarks(
                    annotated,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                st.image(annotated, caption="Mano detectada", use_container_width=True)
                st.success(f"Dedos levantados: {finger_count}")
        else:
            st.image(image, caption="Imagen subida", use_container_width=True)
            st.warning("No se detectó una mano.")
