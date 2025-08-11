import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import cv2

# Seiteneinstellungen
st.set_page_config(page_title="🧪 Interaktiver Lern-Zellkern-Zähler – Stufe 1", layout="wide")
st.title("🧪 Interaktiver Lern-Zellkern-Zähler – Stufe 1")

# Datei-Upload
uploaded_file = st.file_uploader(
    "📁 Bild hochladen",
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)

detected_points = []

if uploaded_file is not None:
    # Bild laden (PIL)
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Zellkern-Detektion (vereinfachtes Beispiel)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Konturen finden
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:  # Nur wenn Fläche > 0
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            detected_points.append((cx, cy))

    st.write(f"**Gefundene Zellkerne:** {len(detected_points)}")

    st.markdown("### ✏️ Interaktive Korrektur")
    st.markdown("**Oben Grün = Punkte hinzufügen, unten Rot = Punkte löschen. Danach auf Feedback speichern klicken.**")

    # Canvas für Punkte hinzufügen (Grün)
    st.subheader("Punkte hinzufügen (grün)")
    canvas_add = st_canvas(
        fill_color="rgba(0, 255, 0, 0.6)",
        stroke_width=3,
        background_image=image,
        update_streamlit=True,
        height=image.height,
        width=image.width,
        drawing_mode="point",
        point_display_radius=5,
        key="canvas_add"
    )

    # Canvas für Punkte löschen (Rot)
    st.subheader("Punkte löschen (rot)")
    canvas_del = st_canvas(
        fill_color="rgba(255, 0, 0, 0.6)",
        stroke_width=3,
        background_image=image,
        update_streamlit=True,
        height=image.height,
        width=image.width,
        drawing_mode="point",
        point_display_radius=5,
        key="canvas_del"
    )

    if st.button("💾 Feedback speichern"):
        add_points = []
        del_points = []

        if canvas_add.json_data is not None:
            for obj in canvas_add.json_data["objects"]:
                add_points.append((int(obj["left"]), int(obj["top"])))

        if canvas_del.json_data is not None:
            for obj in canvas_del.json_data["objects"]:
                del_points.append((int(obj["left"]), int(obj["top"])))

        # Punkte anpassen
        detected_points.extend(add_points)
        detected_points = [p for p in detected_points if p not in del_points]

        st.success(f"Neue Gesamtzahl der Zellkerne: {len(detected_points)}")

else:
    st.info("⬆️ Bitte ein Bild hochladen (PNG, JPG, JPEG, TIF, TIFF).")
