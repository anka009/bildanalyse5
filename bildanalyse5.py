import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, UnidentifiedImageError
import numpy as np
import io

st.set_page_config(page_title="🧪 Interaktiver Lern-Zellkern-Zähler – TIFF-fix", layout="wide")
st.title("🧪 Interaktiver Lern-Zellkern-Zähler – TIFF-fix")

# --- Hilfsfunktion ---
def load_and_prepare_image(uploaded_file, max_dim=1200):
    try:
        img = Image.open(uploaded_file)
    except UnidentifiedImageError as e:
        st.error(f"❌ Bildformat nicht erkannt: {e}")
        st.stop()

    try:
        img.seek(0)  # Erstes Frame bei Mehrseiten-TIFF
    except Exception:
        pass

    if img.mode != "RGB":
        img = img.convert("RGB")

    # Skalieren, wenn zu groß
    orig_w, orig_h = img.size
    scale = min(1.0, max_dim / max(orig_w, orig_h))
    if scale < 1.0:
        img = img.resize((int(orig_w * scale), int(orig_h * scale)), Image.Resampling.LANCZOS)

    # PIL.Image → Bytes (PNG) → Zurück zu PIL.Image (kompakt & kompatibel)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    img = Image.open(buf)
    return img

# --- Upload ---
uploaded_file = st.file_uploader(
    "📁 Bild hochladen",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
    help="Unterstützt PNG, JPG, JPEG, TIF, TIFF"
)

if uploaded_file:
    pil_img = load_and_prepare_image(uploaded_file)

    gefundene_kerne = np.random.randint(1, 100)
    st.markdown(f"**Gefundene Zellkerne:** {gefundene_kerne}")

    st.markdown("""
    ### ✏️ Interaktive Korrektur (Grün = Hinzufügen, Rot = Löschen)
    Klicke in das obere (grün) Canvas, um Punkte hinzuzufügen.  
    Klicke in das untere (rot) Canvas, um Punkte zu markieren, die gelöscht werden sollen.  
    Drücke **Feedback speichern**, wenn du fertig bist.
    """)

    w, h = pil_img.size

    # --- Canvas 1: Punkte hinzufügen ---
    st.subheader("Punkte hinzufügen (Grün)")
    canvas_add = st_canvas(
        fill_color="rgba(0,255,0,0.6)",
        stroke_width=10,
        stroke_color="#00FF00",
        background_image=pil_img,
        update_streamlit=True,
        height=h,
        width=w,
        drawing_mode="point",
        point_display_radius=8,
        key="canvas_add"
    )

    # --- Canvas 2: Punkte löschen ---
    st.subheader("Punkte löschen (Rot)")
    canvas_del = st_canvas(
        fill_color="rgba(255,0,0,0.6)",
        stroke_width=10,
        stroke_color="#FF0000",
        background_image=pil_img,
        update_streamlit=True,
        height=h,
        width=w,
        drawing_mode="point",
        point_display_radius=8,
        key="canvas_del"
    )

    # --- Speichern ---
    if st.button("💾 Feedback speichern"):
        add_points = canvas_add.json_data if canvas_add else {}
        del_points = canvas_del.json_data if canvas_del else {}
        st.success(f"✅ {len(add_points.get('objects', []))} Punkte hinzugefügt, "
                   f"{len(del_points.get('objects', []))} Punkte gelöscht")
