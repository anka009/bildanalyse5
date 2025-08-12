import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, UnidentifiedImageError
import numpy as np
import io

# --- Seiteneinstellungen ---
st.set_page_config(page_title="🧪 Interaktiver Lern-Zellkern-Zähler – Stufe 1", layout="wide")
st.title("🧪 Interaktiver Lern-Zellkern-Zähler – Stufe 1")

# --- Hilfsfunktion zum Laden & Konvertieren ---
def pil_open_rgb(uploaded_file):
    """Bild als PIL.Image öffnen und in RGB konvertieren."""
    try:
        img = Image.open(uploaded_file)
    except UnidentifiedImageError as e:
        st.error(f"❌ Bildformat nicht erkannt: {e}")
        st.stop()

    try:
        img.seek(0)  # Falls Multi-Frame TIFF, erstes Bild wählen
    except Exception:
        pass

    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

# --- Datei-Upload ---
uploaded_file = st.file_uploader(
    "📁 Bild hochladen",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
    help="Unterstützt PNG, JPG, JPEG, TIF, TIFF"
)

if uploaded_file:
    # Bild öffnen
    pil_img = pil_open_rgb(uploaded_file)

    # Beispiel: Zählung simulieren (hier Dummy)
    gefundene_kerne = np.random.randint(1, 100)

    st.markdown(f"**Gefundene Zellkerne:** {gefundene_kerne}")

    st.markdown("""
    ### ✏️ Interaktive Korrektur (Grün = Hinzufügen, Rot = Löschen)
    Klicke in das obere (grün) Canvas, um Punkte hinzuzufügen.  
    Klicke in das untere (rot) Canvas, um Punkte zu markieren, die gelöscht werden sollen.  
    Drücke **Feedback speichern**, wenn du fertig bist.
    """)

    # Bild für Canvas vorbereiten (ggf. skalieren)
    MAX_DIM = 1200
    orig_w, orig_h = pil_img.size
    scale = min(1.0, MAX_DIM / max(orig_w, orig_h))
    display_w = int(orig_w * scale)
    display_h = int(orig_h * scale)

    if scale < 1.0:
        pil_for_canvas = pil_img.resize((display_w, display_h), Image.Resampling.LANCZOS)
    else:
        pil_for_canvas = pil_img

    # --- Canvas 1: Punkte hinzufügen (Grün) ---
    st.subheader("Punkte hinzufügen (Grün)")
    canvas_add = st_canvas(
        fill_color="rgba(0,255,0,0.6)",
        stroke_width=10,
        stroke_color="#00FF00",
        background_image=pil_for_canvas,
        update_streamlit=True,
        height=display_h,
        width=display_w,
        drawing_mode="point",
        point_display_radius=8,
        key="canvas_add"
    )

    # --- Canvas 2: Punkte löschen (Rot) ---
    st.subheader("Punkte löschen (Rot)")
    canvas_del = st_canvas(
        fill_color="rgba(255,0,0,0.6)",
        stroke_width=10,
        stroke_color="#FF0000",
        background_image=pil_for_canvas,
        update_streamlit=True,
        height=display_h,
        width=display_w,
        drawing_mode="point",
        point_display_radius=8,
        key="canvas_del"
    )

    # --- Speichern-Button ---
    if st.button("💾 Feedback speichern"):
        add_points = canvas_add.json_data if canvas_add else {}
        del_points = canvas_del.json_data if canvas_del else {}
        st.success(f"✅ {len(add_points.get('objects', []))} Punkte hinzugefügt, "
                   f"{len(del_points.get('objects', []))} Punkte gelöscht")
