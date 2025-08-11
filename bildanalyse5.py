import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Seiteneinstellungen
st.set_page_config(page_title="🧪 Interaktiver Lern-Zellkern-Zähler – Stufe 1", layout="wide")
st.title("🧪 Interaktiver Lern-Zellkern-Zähler – Stufe 1")

# 1️⃣ Bild hochladen
uploaded_file = st.file_uploader("📁 Bild hochladen", type=["png", "jpg", "jpeg"])
if uploaded_file:
    # Bild laden
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    # 2️⃣ Zellkerne automatisch erkennen (Beispiel mit einfacher Schwelle)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sichere Mittelpunktberechnung
    detected_points = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            detected_points.append((cx, cy))

    # Ergebnisbild zeichnen
    result_img = img_np.copy()
    for (x, y) in detected_points:
        cv2.circle(result_img, (x, y), 5, (255, 0, 0), -1)

    st.subheader("Automatisch erkannte Zellkerne")
    st.image(result_img, caption=f"Gefundene Zellkerne: {len(detected_points)}", use_column_width=True)

    # 3️⃣ Interaktive Korrektur – Grün = Hinzufügen, Rot = Löschen
    st.markdown("### ✏️ Interaktive Korrektur")
    st.write("Oben **Grün** = Punkte hinzufügen, unten **Rot** = Punkte löschen. Danach auf **Feedback speichern** klicken.")

    # Canvas: Punkte hinzufügen
    canvas_add = st_canvas(
        fill_color="rgba(0, 255, 0, 0.6)",
        stroke_width=5,
        stroke_color="green",
        background_image=image,
        update_streamlit=True,
        height=image.height,
        width=image.width,
        drawing_mode="point",
        key="canvas_add"
    )

    # Canvas: Punkte löschen
    canvas_del = st_canvas(
        fill_color="rgba(255, 0, 0, 0.6)",
        stroke_width=5,
        stroke_color="red",
        background_image=image,
        update_streamlit=True,
        height=image.height,
        width=image.width,
        drawing_mode="point",
        key="canvas_del"
    )

    # 4️⃣ Feedback speichern
    if st.button("💾 Feedback speichern"):
        add_points = []
        del_points = []

        if canvas_add.json_data is not None:
            for obj in canvas_add.json_data["objects"]:
                x, y = obj["left"], obj["top"]
                add_points.append((int(x), int(y)))

        if canvas_del.json_data is not None:
            for obj in canvas_del.json_data["objects"]:
                x, y = obj["left"], obj["top"]
                del_points.append((int(x), int(y)))

        # Punkte aktualisieren
        updated_points = detected_points.copy()
        for p in add_points:
            updated_points.append(p)
        for p in del_points:
            updated_points = [pt for pt in updated_points if not (abs(pt[0]-p[0]) < 10 and abs(pt[1]-p[1]) < 10)]

        # Neues Bild mit finalen Punkten
        final_img = img_np.copy()
        for (x, y) in updated_points:
            cv2.circle(final_img, (x, y), 5, (0, 0, 255), -1)

        st.success(f"✅ Aktualisierte Zellkernanzahl: {len(updated_points)}")
        st.image(final_img, caption="Korrigierte Zellkerne", use_column_width=True)
else:
    st.info("Bitte zuerst ein Bild hochladen.")
