import streamlit as st
from PIL import Image
import numpy as np
from pathlib import Path

st.set_page_config(page_title="AI2 Arasaka — Sugarcane Demo", layout="centered")
st.title("AI2 Arasaka — Web Inference (YOLO)")

# --- Find weights automatically ---
def discover_weight_paths():
    root = Path.cwd()
    candidates = []

    # common places
    candidates += [
        root / "Ultralytics Saves" / "segment" / "train" / "weights" / "best.pt",
        root / "Ultralytics Saves" / "segment" / "train" / "weights" / "last.pt",
    ]
    # any .pt in the project
    candidates += list(root.glob("**/*.pt"))

    # de-dup + keep only existing files
    seen = set()
    kept = []
    for p in candidates:
        if p.exists():
            s = str(p.resolve())
            if s not in seen:
                seen.add(s)
                kept.append(p)
    return kept

available = discover_weight_paths()
if not available:
    st.error("No .pt weights found. Put your trained best.pt/last.pt in the project or select a path below.")
    st.stop()

weights = st.selectbox("Select weights", [str(p) for p in available], index=0)

# optional: manual path override
manual = st.text_input("Or paste a full path to a .pt file (optional):", "")
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
conf = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)
imgsz = st.select_slider("Inference image size", options=[320, 384, 448, 512, 640], value=448)

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input", use_container_width=True)

    arr = np.array(img)
    with st.spinner("Running inference..."):
        results = model.predict(source=arr, imgsz=imgsz, conf=conf, verbose=False)
        r0 = results[0]
        plotted_bgr = r0.plot()
        st.image(plotted_bgr[:, :, ::-1], caption="Detections / Segmentations", use_container_width=True)

        st.subheader("Raw results (JSON)")
        st.json(r0.tojson())
