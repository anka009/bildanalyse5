import streamlit as st
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas
import cv2

# Seite konfigurieren
st.set_page_config(page_title="🧪 Interaktiver Lern-Zellkern-Zähler – Stufe 1", layout="wide")
st.title("🧪 Interaktiver Lern-Zellkern-Zähler – Stufe 1")

# Bild hochladen
uploaded_file = st.file_uploader(
    "📁 Bild hochladen",
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)

# Funktion: Zellkerne zählen (Dummy-Beispiel mit OpenCV)
def detect_nuclei(image: Image.Image):
    # Graustufen
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    # Schwelle setzen
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    # Konturen finden
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

if uploaded_file:
    # TIFF richtig öffnen und in RGB konvertieren
    pil_img = Image.open(uploaded_file)
    if pil_img.mode not in ("RGB", "RGBA"):
        pil_img = pil_img.convert("RGB")

    st.image(pil_img, caption="Hochgeladenes Bild", use_container_width=True)

    # Zellkerne zählen
    nuclei_count = detect_nuclei(pil_img)
    st.subheader(f"Gefundene Zellkerne: {nuclei_count}")

    st.markdown("""
    ### ✏️ Interaktive Korrektur (Grün = Hinzufügen, Rot = Löschen)
    Klicke in das obere (grün) Canvas, um Punkte hinzuzufügen.  
    Klicke in das untere (rot) Canvas, um Punkte zu markieren, die gelöscht werden sollen.  
    Drücke **Feedback speichern**, wenn du fertig bist.
    """)

    # Canvas: Punkte hinzufügen
    st.write("#### Punkte hinzufügen (Grün)")
    canvas_add = st_canvas(
        fill_color="rgba(0,255,0,0.6)",
        stroke_width=10,
        stroke_color="green",
        background_image=pil_img,
        update_streamlit=True,
        height=pil_img.height,
        width=pil_img.width,
        drawing_mode="point",
        key="canvas_add"
    )

    # Canvas: Punkte löschen
    st.write("#### Punkte löschen (Rot)")
    canvas_del = st_canvas(
        fill_color="rgba(255,0,0,0.6)",
        stroke_width=10,
        stroke_color="red",
        background_image=pil_img,
        update_streamlit=True,
        height=pil_img.height,
        width=pil_img.width,
        drawing_mode="point",
        key="canvas_del"
    )

    # Feedback speichern
    if st.button("💾 Feedback speichern"):
        added_points = canvas_add.json_data if canvas_add else None
        deleted_points = canvas_del.json_data if canvas_del else None
        st.success("✅ Feedback gespeichert!")
        st.write("Hinzugefügte Punkte:", added_points)
        st.write("Gelöschte Punkte:", deleted_points)
