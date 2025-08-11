import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import json
import os
import time
from streamlit_drawable_canvas import st_canvas

# -------------------- Dateien --------------------
PARAM_DB = "zellkern_params.json"      # gespeicherte (feature -> params) Einträge
FEEDBACK_DB = "zellkern_feedback.json" # gespeicherte Feedback-Einträge (mit Korrekturen)

# -------------------- Hilfsfunktionen --------------------
def load_param_db():
    if os.path.exists(PARAM_DB):
        with open(PARAM_DB, "r") as f:
            return json.load(f)
    return []

def save_param_db(db):
    with open(PARAM_DB, "w") as f:
        json.dump(db, f, indent=2)

def load_feedback_db():
    if os.path.exists(FEEDBACK_DB):
        with open(FEEDBACK_DB, "r") as f:
            return json.load(f)
    return []

def save_feedback_db(db):
    with open(FEEDBACK_DB, "w") as f:
        json.dump(db, f, indent=2)

def get_image_features(img_gray):
    """Einfache Bildmerkmale für Vergleich"""
    return {
        "contrast": float(img_gray.std()),
        "mean_intensity": float(img_gray.mean()),
        "shape": [int(img_gray.shape[0]), int(img_gray.shape[1])]
    }

def similarity_score(features_a, features_b):
    """Kleinerer Score = ähnlicher"""
    c1 = features_a["contrast"]
    c2 = features_b["contrast"]
    m1 = features_a["mean_intensity"]
    m2 = features_b["mean_intensity"]
    s1 = features_a["shape"]
    s2 = features_b["shape"]
    score = abs(c1 - c2) + abs(m1 - m2) + (abs(s1[0] - s2[0]) + abs(s1[1] - s2[1])) / 1000.0
    return score

def find_best_params_kNN(features, db, k=3):
    """Finde k ähnlichste Einträge und mittlere Parameter zurückgeben"""
    if not db:
        return None
    scored = []
    for entry in db:
        score = similarity_score(features, entry["features"])
        scored.append((score, entry["params"]))
    scored.sort(key=lambda x: x[0])
    chosen = [p for _, p in scored[:k]]
    # Mittelwerte für numerische Parameter, take last for color if exists
    if not chosen:
        return None
    avg = {}
    # assume keys: min_size, radius, line_thickness, color
    nums = ["min_size", "radius", "line_thickness"]
    for n in nums:
        vals = [c[n] for c in chosen if n in c]
        if vals:
            avg[n] = int(round(sum(vals) / len(vals)))
    # color: take most common or last
    colors = [c.get("color", "#ff0000") for c in chosen]
    avg["color"] = max(set(colors), key=colors.count)
    return avg

def close_to_any(pt, points, threshold=10):
    return any(np.hypot(pt[0]-p[0], pt[1]-p[1]) < threshold for p in points)

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="🧬 Interaktiver Zellkern-Zähler (lernend)", layout="wide")
st.title("🧬 Interaktiver Zellkern-Zähler – lernfähig (Stufe 1)")

# make storage dirs if needed
os.makedirs("training_data", exist_ok=True)

# -------------------- Upload --------------------
uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg", "png", "tif", "tiff"])
if not uploaded_file:
    st.info("Bitte zuerst ein Bild hochladen.")
    st.stop()

# Load image (PIL for canvas + numpy for OpenCV)
image_pil = Image.open(uploaded_file).convert("RGB")
image = np.array(image_pil)  # RGB
image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
gray_orig = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
features = get_image_features(gray_orig)

# -------------------- Parameter DB & Auto-Params --------------------
param_db = load_param_db()
auto_params = find_best_params_kNN(features, param_db, k=3)

# Sidebar for parameter overrides
st.sidebar.header("⚙️ Parameter (kann überschrieben werden)")
min_size = st.sidebar.slider("Mindestfläche (Pixel)", 10, 20000,
                             auto_params.get("min_size", 1000) if auto_params else 1000, 10)
radius = st.sidebar.slider("Kreisradius Markierung", 2, 100,
                            auto_params.get("radius", 8) if auto_params else 8)
line_thickness = st.sidebar.slider("Liniendicke", 1, 30,
                                   auto_params.get("line_thickness", 2) if auto_params else 2)
color = st.sidebar.color_picker("Farbe der Markierung",
                                auto_params.get("color", "#ff0000") if auto_params else "#ff0000")
remove_threshold = st.sidebar.slider("Lösch-Radius (px)", 3, 50, 12)

# show what auto-params were suggested
if auto_params:
    st.sidebar.markdown("**Vorgeschlagene Parameter (aus ähnlichen Bildern)**")
    st.sidebar.write(auto_params)

# -------------------- Preprocessing & detection --------------------
# CLAHE heuristic
contrast_val = gray_orig.std()
if contrast_val < 40:
    clip_limit = 4.0
elif contrast_val < 80:
    clip_limit = 2.0
else:
    clip_limit = 1.5

clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
gray = clahe.apply(gray_orig)

# Otsu thresholding (invert logic kept)
otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, mask = cv2.threshold(gray, otsu_thresh, 255, cv2.THRESH_BINARY)
if np.mean(gray[mask == 255]) > np.mean(gray[mask == 0]):
    mask = cv2.bitwise_not(mask)

# Morphology
kernel_size = max(3, min(image.shape[0], image.shape[1]) // 300)
kernel = np.ones((kernel_size, kernel_size), np.uint8)
clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

# Watershed splitting (optional simple pipeline using distance transform)
dist_transform = cv2.distanceTransform(clean, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
sure_bg = cv2.dilate(clean, kernel, iterations=3)
unknown = cv2.subtract(sure_bg, sure_fg)
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0
markers = cv2.watershed(image_bgr.copy(), markers)
# remove watershed boundary lines
clean[markers == -1] = 0

# contours
contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = [c for c in contours if cv2.contourArea(c) >= min_size]
centers = []
for c in contours:
    M = cv2.moments(c)
    if M["m00"] != 0:
        centers.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

# -------------------- Show original and auto-detected --------------------
col1, col2 = st.columns(2)
col1.image(image_pil, caption="Original", use_container_width=True)

# marked auto
marked_auto = image.copy()
for (x, y) in centers:
    cv2.circle(marked_auto, (x, y), radius, (255, 0, 0), line_thickness)  # blue auto
col2.image(marked_auto, caption=f"Automatisch erkannte Kerne: {len(centers)}", use_container_width=True)

st.markdown("---")
st.subheader("✏️ Interaktive Korrektur (Stufe 1)")

st.write("Benutze die zwei Canvas-Felder: oben **grün** = Punkte **hinzufügen**, unten **rot** = Punkte **löschen**. Drücke **Feedback speichern**, wenn du fertig bist.")

# -------------------- Two canvases: add (green) and remove (red) --------------------
# Convert image to PIL (already have image_pil)
height, width = image_pil.height, image_pil.width

st.markdown("**Punkte hinzufügen (grün)**")
canvas_add = st_canvas(
    fill_color="",
    stroke_width=8,
    stroke_color="#00FF00",
    background_image=image_pil,
    update_streamlit=True,
    height=height,
    width=width,
    drawing_mode="point",
    key="canvas_add",
)

st.markdown("**Punkte löschen (rot)**")
canvas_remove = st_canvas(
    fill_color="",
    stroke_width=8,
    stroke_color="#FF0000",
    background_image=image_pil,
    update_streamlit=True,
    height=height,
    width=width,
    drawing_mode="point",
    key="canvas_remove",
)

# extract manual points from both canvases
manual_add = []
manual_remove = []
if canvas_add is not None and canvas_add.json_data and "objects" in canvas_add.json_data:
    for obj in canvas_add.json_data["objects"]:
        if "left" in obj and "top" in obj:
            manual_add.append((int(obj["left"]), int(obj["top"])))

if canvas_remove is not None and canvas_remove.json_data and "objects" in canvas_remove.json_data:
    for obj in canvas_remove.json_data["objects"]:
        if "left" in obj and "top" in obj:
            manual_remove.append((int(obj["left"]), int(obj["top"])))

st.write(f"Manuell hinzugefügte Punkte: {len(manual_add)} — Manuell markierte Löschpunkte: {len(manual_remove)}")

# -------------------- Combine automatic detection + manual corrections --------------------
# Start from automatic centers
final_points = centers.copy()

# Remove automatic points near any remove-point
final_points = [pt for pt in final_points if not close_to_any(pt, manual_remove, threshold=remove_threshold)]
# Add manual add points (avoid duplicates)
for p in manual_add:
    if not close_to_any(p, final_points, threshold=3):
        final_points.append(p)

# Show combined result
marked_final = image.copy()
for (x, y) in centers:
    cv2.circle(marked_final, (x, y), radius, (255, 0, 0), line_thickness)  # auto blue
for (x, y) in manual_add:
    cv2.circle(marked_final, (x, y), radius, (0, 255, 0), line_thickness)    # add green
for (x, y) in manual_remove:
    cv2.circle(marked_final, (x, y), radius, (0, 0, 255), max(1, line_thickness//2))  # remove red (marker)

st.image(marked_final, caption=f"Finale Punkte: {len(final_points)} (automatisch {len(centers)}, +{len(manual_add)}, -{len(manual_remove)})", use_container_width=True)

# -------------------- Export & Feedback saving --------------------
df_final = pd.DataFrame(final_points, columns=["X", "Y"])
csv = df_final.to_csv(index=False).encode("utf-8")
st.download_button("📥 Finale Zellkerne als CSV exportieren", data=csv, file_name="zellkerne_final.csv", mime="text/csv")

if st.button("💾 Feedback speichern (Parameter + Korrekturen)"):
    # Save a feedback entry describing what happened
    db = load_feedback_db()
    entry = {
        "timestamp": time.time(),
        "image_name": uploaded_file.name,
        "features": features,
        "params_used": {
            "min_size": min_size,
            "radius": radius,
            "line_thickness": line_thickness,
            "color": color
        },
        "auto_count": len(centers),
        "added_count": len(manual_add),
        "removed_count": len(manual_remove),
        "final_count": len(final_points),
        "added_points": manual_add,
        "removed_points": manual_remove
    }
    db.append(entry)
    save_feedback_db(db)
    # Also add to PARAM_DB as "example of good params" so find_best can pick it up later
    param_db = load_param_db()
    store_entry = {"features": features, "params": entry["params_used"], "meta": {"auto_count": entry["auto_count"], "final_count": entry["final_count"]}}
    param_db.append(store_entry)
    save_param_db(param_db)
    st.success("Feedback & Parameter gespeichert. Das Programm nutzt das künftig zur automatischen Parametrierung.")

st.markdown("---")
st.info("Hinweis: Gespeicherte Feedback-/Parameterdaten findest du in den Dateien 'zellkern_feedback.json' und 'zellkern_params.json' im Arbeitsverzeichnis. Diese Dateien kannst du später für ein echtes ML-Training (Stufe 2) verwenden.")
