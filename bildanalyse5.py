import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageDraw
import numpy as np

# Seiteneinstellungen
st.set_page_config(page_title="🧪 Interaktiver Lern-Zellkern-Zähler – Stufe 1", layout="wide")
st.title("🧪 Interaktiver Lern-Zellkern-Zähler – Stufe 1")

# Datei-Upload (nur PNG, JPG, JPEG)
uploaded_file = st.file_uploader(
    "📁 Bild hochladen", 
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    # Bild laden und als RGB konvertieren
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Beispiel: Dummy-Kernerkennung (hier nur zufällige Punkte)
    np.random.seed(42)
    points = [(np.random.randint(0, image.width), np.random.randint(0, image.height)) for _ in range(10)]

    # Original anzeigen mit Punkten
    preview_img = image.copy()
    draw = ImageDraw.Draw(preview_img)
    for p in points:
        r = 5
        draw.ellipse((p[0]-r, p[1]-r, p[0]+r, p[1]+r), outline="red", width=2)

    st.image(preview_img, caption=f"Gefundene Zellkerne: {len(points)}", use_container_width=True)

    st.markdown("""
    ✏️ **Interaktive Korrektur** (Grün = Hinzufügen, Rot = Löschen)  
    Klicke in das obere (grün) Canvas, um Punkte hinzuzufügen.  
    Klicke in das untere (rot) Canvas, um Punkte zu markieren, die gelöscht werden sollen.  
    Drücke **Feedback speichern**, wenn du fertig bist.
    """)

    # Canvas: Punkte hinzufügen (Grün)
    st.subheader("Punkte hinzufügen (Grün)")
    canvas_add = st_canvas(
        fill_color="rgba(0,255,0,0.6)",
        stroke_width=5,
        background_image=image,
        update_streamlit=True,
        height=image.height,
        width=image.width,
        drawing_mode="point",
        key="canvas_add"
    )

    # Canvas: Punkte löschen (Rot)
    st.subheader("Punkte löschen (Rot)")
    canvas_del = st_canvas(
        fill_color="rgba(255,0,0,0.6)",
        stroke_width=5,
        background_image=image,
        update_streamlit=True,
        height=image.height,
        width=image.width,
        drawing_mode="point",
        key="canvas_del"
    )

    # Feedback-Button
    if st.button("✅ Feedback speichern"):
        add_points = canvas_add.json_data["objects"] if canvas_add and canvas_add.json_data else []
        del_points = canvas_del.json_data["objects"] if canvas_del and canvas_del.json_data else []

        st.write(f"Hinzugefügte Punkte: {len(add_points)}")
        st.write(f"Gelöschte Punkte: {len(del_points)}")
