# app.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, UnidentifiedImageError
import numpy as np
import cv2
import json
import os
import time

# -------------------- Settings --------------------
st.set_page_config(page_title="🧬 Lernfähiger Zellkern-Zähler — Stufe 1", layout="wide")
st.title("🧬 Lernfähiger Zellkern-Zähler — Stufe 1 (Grün = add, Rot = remove)")

FEEDBACK_FILE = "feedback.json"
os.makedirs("training_data", exist_ok=True)

# -------------------- Helpers --------------------
def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            return json.load(f)
    return []

def save_feedback(entry):
    db = load_feedback()
    db.append(entry)
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(db, f, indent=2)

def pil_open_rgb(uploaded_file):
    """Öffnet viele Formate sicher und gibt PIL.Image in RGB zurück."""
    try:
        img = Image.open(uploaded_file)
    except UnidentifiedImageError:
        raise
    # tif/ tiff können mehrere Seiten haben — wir nehmen Frame 0
    try:
        img.seek(0)
    except Exception:
        pass
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def detect_nuclei_simple(np_rgb, min_area=10):
    """Einfache Erkennung als Fallback (Threshold + Konturen)."""
    gray = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # invert if nuclei appear darker/whiter heuristic (keine perfekte Lösung)
    # morpho clean
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        M = cv2.moments(c)
        if M.get("m00", 0) and cv2.contourArea(c) >= min_area:
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            centers.append((cx, cy))
    return centers

def canvas_objects_to_points(json_data, scale_x, scale_y):
    """Konvertiert canvas objects -> Originalbild-Koordinaten (mittig)."""
    pts = []
    if not json_data or "objects" not in json_data:
        return pts
    for obj in json_data["objects"]:
        # Some versions have 'left'/'top' and 'width'/'height', others just left/top.
        left = obj.get("left", None)
        top = obj.get("top", None)
        w = obj.get("width", 0)
        h = obj.get("height", 0)
        if left is None or top is None:
            continue
        # center in canvas coords
        cx = left + w/2
        cy = top + h/2
        # map back to original coords
        orig_x = int(round(cx * scale_x))
        orig_y = int(round(cy * scale_y))
        pts.append((orig_x, orig_y))
    return pts

def close_to_any(pt, pts_list, thr=8):
    return any(np.hypot(pt[0]-p[0], pt[1]-p[1]) < thr for p in pts_list)

# -------------------- UI: Upload --------------------
uploaded = st.file_uploader("📁 Bild hochladen (PNG, JPG, JPEG, TIF, TIFF)", type=["png","jpg","jpeg","tif","tiff"])
if not uploaded:
    st.info("Bitte lade ein Bild hoch (Unterstützt: png, jpg, jpeg, tif, tiff).")
    st.stop()

# Load PIL image in RGB (important for st_canvas)
try:
    pil_img = pil_open_rgb(uploaded)
except Exception as e:
    st.error(f"Fehler beim Öffnen des Bildes: {e}")
    st.stop()

# Convert to numpy for OpenCV processing (RGB order)
np_img = np.array(pil_img)

# -------------------- Detect (automatic) --------------------
st.sidebar.header("🔧 Erkennungs-Parameter")
min_area = st.sidebar.slider("Minimale Konturfläche (px)", 5, 5000, 30, 1)
detection_radius_visual = st.sidebar.slider("Visual Radius (px)", 2, 30, 6, 1)

with st.spinner("Automatische Vorverarbeitung & Erkennung..."):
    auto_centers = detect_nuclei_simple(np_img, min_area=min_area)

# Show summary
st.markdown(f"**Automatisch gefundene Punkte:** {len(auto_centers)}")

# -------------------- Scaling for Canvas (avoid huge canvases) --------------------
# We display a scaled down canvas if image > max_dim to keep UI responsive.
MAX_DIM = 1200  # max width/height for canvas display
orig_w, orig_h = pil_img.width, pil_img.height
scale = 1.0
if max(orig_w, orig_h) > MAX_DIM:
    scale = MAX_DIM / max(orig_w, orig_h)
display_w = int(round(orig_w * scale))
display_h = int(round(orig_h * scale))

# create a resized PIL for canvas background
if scale < 1.0:
    pil_for_canvas = pil_img.resize((display_w, display_h), Image.Resampling.LANCZOS)
else:
    pil_for_canvas = pil_img

# Compute mapping from canvas -> original: scale_x = orig_w / canvas_w
scale_x = orig_w / display_w
scale_y = orig_h / display_h

# Draw automatic centers on an image for preview
preview_rgb = np.array(pil_for_canvas).copy()
for (x, y) in auto_centers:
    # map to preview coords
    px = int(round(x / scale_x))
    py = int(round(y / scale_y))
    cv2.circle(preview_rgb, (px, py), detection_radius_visual, (255,0,0), -1)  # blue/red in RGB displayed by st.image

st.subheader("Original (links) — Automatisch erkannte Punkte (rechts)")
col1, col2 = st.columns(2)
col1.image(pil_img, caption="Original (PIL)", use_column_width=True)
col2.image(preview_rgb, caption=f"Automatisch: {len(auto_centers)} Punkte (rot)", use_column_width=True)

st.markdown("---")
st.subheader("✏️ Interaktive Korrektur (Grün = Hinzufügen, Rot = Löschen)")
st.markdown("Klicke in das obere (grün) Canvas, um Punkte hinzuzufügen. Klicke in das untere (rot) Canvas, um Punkte zu markieren, die gelöscht werden sollen. Drücke **Feedback speichern** wenn du fertig bist.")

# -------------------- Canvas Add (green) --------------------
canvas_add = st_canvas(
    fill_color="",  # no fill
    stroke_width=12,
    stroke_color="#00FF00",
    background_image=pil_for_canvas,  # MUST be PIL.Image
    update_streamlit=True,
    height=display_h,
    width=display_w,
    drawing_mode="point",
    point_display_radius=8,
    key="canvas_add",
)

# -------------------- Canvas Remove (red) --------------------
canvas_remove = st_canvas(
    fill_color="",
    stroke_width=12,
    stroke_color="#FF0000",
    background_image=pil_for_canvas,  # MUST be PIL.Image
    update_streamlit=True,
    height=display_h,
    width=display_w,
    drawing_mode="point",
    point_display_radius=8,
    key="canvas_remove",
)

# -------------------- Extract points and map back to original coords --------------------
added_canvas_pts = canvas_objects_to_points(canvas_add.json_data if canvas_add else None, scale_x, scale_y)
removed_canvas_pts = canvas_objects_to_points(canvas_remove.json_data if canvas_remove else None, scale_x, scale_y)

st.write(f"Manuell hinzufügte Punkte: {len(added_canvas_pts)} — Markierte Löschpunkte: {len(removed_canvas_pts)}")

# -------------------- Combine auto + manual corrections --------------------
# start with auto_centers
final_pts = auto_centers.copy()

# remove auto points close to any removed point
for rem in removed_canvas_pts:
    final_pts = [p for p in final_pts if not close_to_any(p, [rem], thr=12)]

# add manual adds if not duplicates
for add in added_canvas_pts:
    if not close_to_any(add, final_pts, thr=6):
        final_pts.append(add)

# draw final preview (rescale to preview)
final_preview = np.array(pil_for_canvas).copy()
for (x, y) in final_pts:
    px = int(round(x / scale_x))
    py = int(round(y / scale_y))
    cv2.circle(final_preview, (px, py), detection_radius_visual, (0,255,0), -1)  # green final

st.markdown("### Finale (nach Korrekturen)")
st.image(final_preview, caption=f"Finale Punkte: {len(final_pts)}", use_column_width=True)

# -------------------- Export / Save Feedback --------------------
st.markdown("---")
col_save1, col_save2 = st.columns([1,3])
with col_save1:
    save_label = st.text_input("Label für dieses Feedback (optional)", value="")
    if st.button("💾 Feedback speichern"):
        entry = {
            "timestamp": time.time(),
            "image_name": getattr(uploaded, "name", "uploaded_image"),
            "orig_shape": [orig_h, orig_w],
            "auto_count": len(auto_centers),
            "added_count": len(added_canvas_pts),
            "removed_count": len(removed_canvas_pts),
            "final_count": len(final_pts),
            "added_points": added_canvas_pts,
            "removed_points": removed_canvas_pts,
            "final_points": final_pts,
            "min_area": int(min_area),
            "label": save_label
        }
        save_feedback(entry)
        st.success(f"Feedback gespeichert in {FEEDBACK_FILE} (final points: {len(final_pts)})")

with col_save2:
    # allow CSV download of final points
    if len(final_pts) > 0:
        import io, csv
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["X","Y"])
        writer.writerows(final_pts)
        st.download_button("📥 Finale Punkte als CSV herunterladen", data=buf.getvalue().encode("utf-8"),
                            file_name="zellkerne_final.csv", mime="text/csv")

st.caption("Hinweis: Die Canvas-Hintergründe werden als PIL.Image übergeben (RGB). Große Bilder werden skaliert für die Anzeige; die Klickkoordinaten werden korrekt auf das Original zurückgerechnet.")
