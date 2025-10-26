import streamlit as st
from PIL import Image
import numpy as np
from pathlib import Path

st.set_page_config(page_title="AI2 Arasaka — Sugarcane Demo", layout="centered")
st.title("AI2 Arasaka — Web Inference (YOLO)")

# --- Find weights inside the repo (root + models/) ---
def discover_weight_paths():
    root = Path.cwd()
    candidates = []

    # common repo locations
    candidates += [root / "best.pt", root / "last.pt"]
    candidates += list((root / "models").glob("*.pt")) if (root / "models").exists() else []

    # light recursive search as fallback (one level deep to avoid slow builds)
    for p in root.glob("*/*.pt"):
        candidates.append(p)

    # de-dup + keep only existing files; prefer best.pt first
    uniq = {}
    for p in candidates:
        if p.exists() and p.suffix.lower() == ".pt":
            uniq[str(p.resolve())] = p
    found = list(uniq.values())
    found.sort(key=lambda p: (0 if p.name.lower() == "best.pt" else 1, len(str(p))))
    return found

available = discover_weight_paths()
if not available:
    st.error(
        "No .pt weights found in the repo. "
        "Add 'best.pt' to the repo root or place weights in a 'models/' folder."
    )
    st.stop()

weights = st.selectbox("Select weights", [str(p) for p in available], index=0)

# Optional manual override
manual = st.text_input("Or paste a full path/filename (optional):", "")
if manual.strip():
    p = Path(manual.strip().strip('"'))
    if p.exists() and p.suffix.lower() == ".pt":
        weights = str(p.resolve())
    else:
        st.warning("Path not found or not a .pt file; using the selected item above.")

# --- Load model (cached) ---
@st.cache_resource
def load_model(weights_path: str):
    from ultralytics import YOLO
    return YOLO(weights_path)

model = load_model(weights)

# --- Controls ---
conf  = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)
iou   = st.slider("IoU threshold", 0.1, 0.9, 0.50, 0.05)
imgsz = st.select_slider("Inference image size", options=[320, 384, 448, 512, 640], value=448)

with st.expander("Model info"):
    st.write("weights:", weights)
    try: st.write("task:", getattr(model, "task", "unknown"))
    except: pass
    try: st.write("classes:", model.names)
    except: pass

# --- UI ---
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input", use_container_width=True)

    arr = np.array(img)
    with st.spinner("Running inference..."):
        results = model.predict(source=arr, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
        r0 = results[0]

        # if nothing detected, say so; otherwise show overlay
        has_boxes = getattr(r0, "boxes", None) is not None and len(r0.boxes) > 0
        has_masks = getattr(r0, "masks", None) is not None and r0.masks is not None

        if not has_boxes and not has_masks:
            st.warning("No detections/masks on this image.")
        else:
            plotted_bgr = r0.plot()
            st.image(plotted_bgr[:, :, ::-1], caption="Detections / Segmentations", use_container_width=True)

        st.subheader("Raw results (JSON)")
        st.json(r0.tojson())
