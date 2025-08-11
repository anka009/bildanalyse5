import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="Interaktiver Lern-Zellkern-Zähler – Stufe 1", layout="wide")
st.title("🧪 Interaktiver Lern-Zellkern-Zähler – Stufe 1")

# --------------------
# Bild-Upload
# --------------------
uploaded_file = st.file_uploader("📁 Bild hochladen", type=["png", "jpg", "jpeg"])
if uploaded_file:
    pil_image = Image.open(uploaded_file).convert("RGB")
    np_image = np.array(pil_image)

    # --------------------
    # Automatische Voranalyse (hier nur Dummy-Punkte)
    # --------------------
    # Beispiel: einfache Schwelle für Kern-Erkennung (sehr grob)
    gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_points = [tuple(map(int, cv2.moments(c)['m10']/cv2.moments(c)['m00'],
                                  cv2.moments(c)['m01']/cv2.moments(c)['m00']))
                       for c in contours if cv2.moments(c)['m00'] != 0]

    st.subheader("✏️ Interaktive Korrektur (Stufe 1)")
    st.markdown("**Oben grün** = Punkte **hinzufügen**, **unten rot** = Punkte **löschen**.\n"
                "Drücke **Feedback speichern**, wenn du fertig bist.")

    # --------------------
    # Canvas für Hinzufügen (Grün)
    # --------------------
    st.markdown("### Punkte hinzufügen (grün)")
    canvas_add = st_canvas(
        fill_color="",
        stroke_width=10,
        stroke_color="#00FF00",  # grün
        background_image=pil_image,  # PIL.Image statt NumPy
        update_streamlit=True,
        height=pil_image.height,
        width=pil_image.width,
        drawing_mode="point",
        key="canvas_add",
    )

    # --------------------
    # Canvas für Löschen (Rot)
    # --------------------
    st.markdown("### Punkte löschen (rot)")
    canvas_del = st_canvas(
        fill_color="",
        stroke_width=10,
        stroke_color="#FF0000",  # rot
        background_image=pil_image,  # PIL.Image statt NumPy
        update_streamlit=True,
        height=pil_image.height,
        width=pil_image.width,
        drawing_mode="point",
        key="canvas_del",
    )

    # --------------------
    # Button zum Speichern des Feedbacks
    # --------------------
    if st.button("💾 Feedback speichern"):
        added_points = []
        deleted_points = []

        # Punkte aus Canvas extrahieren (falls welche gesetzt wurden)
        if canvas_add.json_data and "objects" in canvas_add.json_data:
            for obj in canvas_add.json_data["objects"]:
                x = obj["left"] + obj["width"] / 2
                y = obj["top"] + obj["height"] / 2
                added_points.append((int(x), int(y)))

        if canvas_del.json_data and "objects" in canvas_del.json_data:
            for obj in canvas_del.json_data["objects"]:
                x = obj["left"] + obj["width"] / 2
                y = obj["top"] + obj["height"] / 2
                deleted_points.append((int(x), int(y)))

        st.success(f"✅ {len(added_points)} Punkte hinzugefügt, {len(deleted_points)} Punkte gelöscht.")

        # --------------------
        # "Lernen" = hier nur Aktualisierung der Liste
        # --------------------
        final_points = set(detected_points)
        final_points.update(added_points)
        for p in deleted_points:
            final_points.discard(p)

        st.write("📊 **Endgültige Punktzahl:**", len(final_points))
        st.image(pil_image, caption="Aktuelles Bild mit korrigierten Punkten")

else:
    st.info("⬆️ Bitte zuerst ein Bild hochladen.")
