import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import cv2
import tempfile
import os

# -----------------------------------
# Seiteneinstellungen
# -----------------------------------
st.set_page_config(page_title="🧪 Zellkern-Zähler – Stufe 1", layout="wide")
st.title("🧪 Interaktiver Zellkern-Zähler – Stufe 1")

# -----------------------------------
# Bild-Upload
# -----------------------------------
uploaded_file = st.file_uploader(
    "📁 Bild hochladen",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    # Bild laden und nach RGB konvertieren
    pil_image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(pil_image)

    # -----------------------------------
    # Zellkern-Erkennung (Basisversion)
    # -----------------------------------
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    # Otsu-Schwellenwert
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Konturen finden
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kern_count = len(contours)

    st.markdown(f"### 🧮 Gefundene Zellkerne: **{kern_count}**")

    # -----------------------------------
    # Bild temporär speichern, damit Canvas es laden kann
    # -----------------------------------
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    pil_image.save(tmp_file.name)
    tmp_path = tmp_file.name

    # -----------------------------------
    # Interaktive Korrektur
    # -----------------------------------
    st.markdown("✏️ **Interaktive Korrektur** — oben **GRÜN** = add, unten **ROT** = remove")

    # Canvas zum Hinzufügen (grün)
    canvas_add = st_canvas(
        fill_color="rgba(0,255,0,0.6)",
        stroke_width=10,
        background_image=Image.open(tmp_path),
        update_streamlit=True,
        height=pil_image.height,
        width=pil_image.width,
        drawing_mode="point",
        key="canvas_add"
    )

    # Canvas zum Entfernen (rot)
    canvas_remove = st_canvas(
        fill_color="rgba(255,0,0,0.6)",
        stroke_width=10,
        background_image=Image.open(tmp_path),
        update_streamlit=True,
        height=pil_image.height,
        width=pil_image.width,
        drawing_mode="point",
        key="canvas_remove"
    )

    # -----------------------------------
    # Feedback speichern
    # -----------------------------------
    if st.button("💾 Feedback speichern"):
        st.success("✅ Feedback gespeichert!")
        # Hier könnte man Punkte aus canvas_add.json_data und canvas_remove.json_data auswerten
        st.write("📌 Punkte zum Hinzufügen:", canvas_add.json_data)
        st.write("📌 Punkte zum Entfernen:", canvas_remove.json_data)

    # -----------------------------------
    # Aufräumen
    # -----------------------------------
    os.unlink(tmp_path)
