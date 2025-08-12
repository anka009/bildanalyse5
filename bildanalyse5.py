# app.py — Robuste Stufe 1 (Canvas-Fallbacks)
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, UnidentifiedImageError
import numpy as np
import cv2
import io, base64, tempfile, traceback, os, time, json

st.set_page_config(page_title="Lern-Zellkern-Zähler (robust)", layout="wide")
st.title("🧬 Lern-Zellkern-Zähler — Stufe 1 (robust, Grün=Add, Rot=Remove)")

FEEDBACK_FILE = "feedback.json"
os.makedirs("training_data", exist_ok=True)

def save_feedback(entry):
    db = []
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r") as f:
                db = json.load(f)
        except Exception:
            db = []
    db.append(entry)
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(db, f, indent=2)

def pil_open_rgb(uploaded_file):
    try:
        img = Image.open(uploaded_file)
    except UnidentifiedImageError as e:
        st.error(f"Fehler beim Öffnen des Bildes: {e}")
        st.stop()
    # Bei Mehrseiten-TIFF: erstes Frame
    try:
        img.seek(0)
    except Exception:
        pass
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def detect_nuclei_simple(np_rgb, min_area=20):
    """Einfacher Threshold+Contours-Fallback zur Demonstration."""
    gray = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        area = cv2.contourArea(c)
        M = cv2.moments(c)
        if M.get("m00", 0) and area >= min_area:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))
    return centers

def pil_to_data_uri(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b}"

def pil_to_tempfile_path(pil_img):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    pil_img.save(tmp, format="PNG")
    tmp.close()
    return tmp.name

def try_create_canvas_variants(pil_for_canvas, canvas_kwargs, key_prefix):
    """
    Versuche st_canvas mit mehreren background_image-Varianten:
    0: PIL.Image
    1: data URI (base64 PNG)
    2: temporaere PNG-Datei (Pfad)
    Gibt das erste erfolgreiche Canvas-Objekt zurück oder None + Fehlerinfo.
    """
    attempts = []
    # variant 0: PIL.Image
    attempts.append(("pil", pil_for_canvas))
    # variant 1: data URI
    try:
        data_uri = pil_to_data_uri(pil_for_canvas)
        attempts.append(("data_uri", data_uri))
    except Exception as e:
        attempts.append(("data_uri_error", str(e)))
    # variant 2: temp file path
    try:
        tmp_path = pil_to_tempfile_path(pil_for_canvas)
        attempts.append(("tmpfile", tmp_path))
    except Exception as e:
        attempts.append(("tmpfile_error", str(e)))

    last_exc = None
    for idx, (tag, bg) in enumerate(attempts):
        if not isinstance(bg, (str, Image.Image)):
            continue
        key = f"{key_prefix}_bg_{tag}_{idx}"
        try:
            canvas = st_canvas(background_image=bg, key=key, **canvas_kwargs)
            return canvas, tag, None
        except Exception as e:
            last_exc = (tag, traceback.format_exc())
            # continue to next variant
    return None, None, last_exc

# ---------------- UI ----------------
uploaded = st.file_uploader("📁 Bild hochladen (png, jpg, jpeg, tif, tiff)", type=["png","jpg","jpeg","tif","tiff"])
if not uploaded:
    st.info("Bitte Bild hochladen.")
    st.stop()

# sichere PIL-Öffnung & RGB-Konvert
pil_img = pil_open_rgb(uploaded)
np_img = np.array(pil_img)  # RGB array

# detection
min_area = st.sidebar.slider("Minimale Konturfläche (px)", 5, 200, 20)
auto_centers = detect_nuclei_simple(np_img, min_area=min_area)

st.markdown(f"**Automatisch gefundene Kerne:** {len(auto_centers)}")

# scale for display to keep canvas responsive
MAX_DIM = 1200
orig_w, orig_h = pil_img.size
scale = 1.0
if max(orig_w, orig_h) > MAX_DIM:
    scale = MAX_DIM / max(orig_w, orig_h)
display_w = int(round(orig_w * scale))
display_h = int(round(orig_h * scale))
if scale < 1.0:
    pil_for_canvas = pil_img.resize((display_w, display_h), Image.Resampling.LANCZOS)
else:
    pil_for_canvas = pil_img

# draw preview with auto points
preview = np.array(pil_for_canvas).copy()
vis_rad = st.sidebar.slider("Visual Radius (px)", 3, 25, 6)
for (x,y) in auto_centers:
    px = int(round(x * (display_w / orig_w)))
    py = int(round(y * (display_h / orig_h)))
    cv2.circle(preview, (px, py), vis_rad, (255,0,0), -1)  # red in RGB

col1, col2 = st.columns(2)
col1.image(pil_img, caption="Original", use_container_width=True)
col2.image(preview, caption="Automatisch (rot)", use_container_width=True)

st.markdown("---")
st.subheader("✏️ Interaktive Korrektur — oben GRÜN = add, unten ROT = remove")
st.write("Klicke oben, um Punkte hinzuzufügen; unten, um Punkte zum Löschen zu markieren. Drücke 'Feedback speichern' wenn fertig.")

canvas_kwargs = dict(
    fill_color="",
    stroke_width=12,
    update_streamlit=True,
    height=display_h,
    width=display_w,
    drawing_mode="point",
    point_display_radius=8
)

# Try create green canvas
canvas_add, tag_add, err_add = try_create_canvas_variants(pil_for_canvas, {**canvas_kwargs, "stroke_color":"#00FF00"}, "canvas_add")
if canvas_add is None:
    st.error("Konnte das grüne Canvas nicht initialisieren. Letzter Versuch: " + (err_add[0] if err_add else "unknown"))
    st.stop()

# Try create red canvas
canvas_remove, tag_rem, err_rem = try_create_canvas_variants(pil_for_canvas, {**canvas_kwargs, "stroke_color":"#FF0000"}, "canvas_remove")
if canvas_remove is None:
    st.error("Konnte das rote Canvas nicht initialisieren. Letzter Versuch: " + (err_rem[0] if err_rem else "unknown"))
    st.stop()

# map canvas objects to original coords
def canvas_objs_to_orig(obj_json):
    pts = []
    if not obj_json or "objects" not in obj_json:
        return pts
    for obj in obj_json["objects"]:
        left = obj.get("left", 0)
        top = obj.get("top", 0)
        w = obj.get("width", 0)
        h = obj.get("height", 0)
        cx = left + w/2
        cy = top + h/2
        orig_x = int(round(cx / scale))
        orig_y = int(round(cy / scale))
        pts.append((orig_x, orig_y))
    return pts

added_pts = canvas_objs_to_orig(canvas_add.json_data)
removed_pts = canvas_objs_to_orig(canvas_remove.json_data)

st.write(f"Manuell hinzugefügt: {len(added_pts)} — Markiert zum Entfernen: {len(removed_pts)}")

# combine auto + manual
def close_to_any(pt, pts, thr=8):
    return any(np.hypot(pt[0]-p[0], pt[1]-p[1]) < thr for p in pts)

final_pts = [p for p in auto_centers if not close_to_any(p, removed_pts, thr=12)]
for a in added_pts:
    if not close_to_any(a, final_pts, thr=6):
        final_pts.append(a)

# preview final
final_preview = np.array(pil_for_canvas).copy()
for (x,y) in final_pts:
    px = int(round(x * (display_w / orig_w)))
    py = int(round(y * (display_h / orig_h)))
    cv2.circle(final_preview, (px,py), vis_rad, (0,255,0), -1)  # green

st.markdown("### Finale Punkte (nach Korrektur)")
st.image(final_preview, caption=f"Finale Punkte: {len(final_pts)}", use_container_width=True)

# save / export
if st.button("💾 Feedback speichern"):
    entry = {
        "timestamp": time.time(),
        "image_name": getattr(uploaded, "name", "uploaded_image"),
        "orig_shape": [orig_h, orig_w],
        "auto_count": len(auto_centers),
        "added_count": len(added_pts),
        "removed_count": len(removed_pts),
        "final_count": len(final_pts),
        "added_points": added_pts,
        "removed_points": removed_pts,
        "final_points": final_pts,
        "min_area": int(min_area)
    }
    save_feedback(entry)
    st.success(f"Feedback gespeichert ({len(final_pts)} Punkte). Datei: {FEEDBACK_FILE}")

# provide CSV download
if len(final_pts) > 0:
    import csv, io
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["X","Y"])
    writer.writerows(final_pts)
    st.download_button("📥 Finale Punkte als CSV herunterladen", data=buf.getvalue().encode("utf-8"),
                       file_name="zellkerne_final.csv", mime="text/csv")

st.caption("Wenn weiterhin Fehler auftreten: poste bitte die kurze Fehlermeldung aus der App-Log (traceback), damit ich den genauen Failing-Step sehen kann.")
