import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image
import json
import os

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title="PallorCheck", page_icon="🩸", layout="centered")
st.title("🩸 PallorCheck")
st.write("Experimental conjunctiva color analysis — NOT a medical device.")

# ---------------------------------------------------------------------------
# Load thresholds automatically from thresholds.json
# ---------------------------------------------------------------------------
@st.cache_data
def load_thresholds():
    if os.path.exists("thresholds.json"):
        with open("thresholds.json", "r") as f:
            data = json.load(f)
            return data.get("thr_high", 121.94), data.get("thr_low", 74.0)
    return 121.94, 74.0  # Fallback defaults

TH_HIGH, TH_LOW = load_thresholds()

# ---------------------------------------------------------------------------
# WHO Reference Standards Loader
# ---------------------------------------------------------------------------
def classify_who(hgb, gender):
    g = str(gender).strip().upper()
    if g == 'F':
        if hgb >= 12.0: return 'Normal'
        if hgb >= 11.0: return 'Mild'
        if hgb >= 8.0:  return 'Moderate'
        return 'Severe'
    else:
        if hgb >= 13.0: return 'Normal'
        if hgb >= 11.0: return 'Mild'
        if hgb >= 8.0:  return 'Moderate'
        return 'Severe'

@st.cache_data
def get_reference_standards():
    try:
        df = pd.read_excel('India.xlsx', sheet_name='Foglio1')
        df['WHO_Category'] = df.apply(lambda r: classify_who(r['Hgb'], r['Gender']), axis=1)
        counts = df['WHO_Category'].value_counts().to_dict()
        by_gender = df.groupby(['Gender', 'WHO_Category']).size().unstack(fill_value=0)
        return counts, by_gender
    except Exception:
        return None

ref_stats = get_reference_standards()

# ---------------------------------------------------------------------------
# Eye detection & Regions
# ---------------------------------------------------------------------------
@st.cache_resource
def load_eye_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def detect_eye_box(img_rgb, eye_cascade):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
    if len(eyes) == 0:
        return None
    return max(eyes, key=lambda b: b[2] * b[3])

def get_regions(img_rgb, eye_box):
    h, w, _ = img_rgb.shape
    # If the eye detector fails, assume the user's uploaded image 
    # is a close-up crop of the eye, and target the lower-inner lid directly.
    if eye_box is None:
        conj = img_rgb[int(h * 0.50):int(h * 0.90), int(w * 0.20):int(w * 0.80)]
        scl = img_rgb[int(h * 0.10):int(h * 0.40), int(w * 0.20):int(w * 0.80)]
        return conj, scl, False

    ex, ey, ew, eh = eye_box
    cy0, cy1 = ey + int(eh * 0.55), ey + int(eh * 0.95)
    cx0, cx1 = ex + int(ew * 0.15), ex + int(ew * 0.85)
    sy0, sy1 = ey + int(eh * 0.05), ey + int(eh * 0.30)
    sx0, sx1 = ex + int(ew * 0.15), ex + int(ew * 0.85)

    conj = img_rgb[max(cy0, 0):cy1, max(cx0, 0):cx1]
    scl = img_rgb[max(sy0, 0):sy1, max(sx0, 0):sx1]

    if conj.size == 0 or scl.size == 0:
        return get_regions(img_rgb, None)
    return conj, scl, True

# ---------------------------------------------------------------------------
# Feature Extraction & Scoring
# ---------------------------------------------------------------------------
def extract_features(patch_rgb):
    lab = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2HSV)
    return float(np.median(lab[:, :, 1])), float(np.median(lab[:, :, 0])), float(np.median(hsv[:, :, 1]))

def relative_pallor_score(conj_rgb, scl_rgb):
    a_conj, l_conj, sat_conj = extract_features(conj_rgb)
    a_scl, l_scl, sat_scl = extract_features(scl_rgb)

    a_diff = a_conj - a_scl
    sat_diff = sat_conj - sat_scl

    score = (a_diff * 1.25) + (sat_diff * 0.75)
    return score, {"a_diff": a_diff, "sat_diff": sat_diff}

# ---------------------------------------------------------------------------
# Main Interface Layout
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload Conjunctiva Photograph", type=["png", "jpg", "jpeg"], key="main")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)

    eye_cascade = load_eye_cascade()
    eye_box = detect_eye_box(img_rgb, eye_cascade)
    conj_roi, scl_roi, detected = get_regions(img_rgb, eye_box)

    # Clean multi-column preview layout
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image, caption="Original", use_column_width=True)
    with col2:
        st.image(conj_roi, caption="Conjunctiva", use_column_width=True)
    with col3:
        st.image(scl_roi, caption="Sclera Ref", use_column_width=True)

    if not detected:
        st.info("Eye not detected automatically — used fallback crops. For best results, use a close-up, well-lit photo.")

    score, details = relative_pallor_score(conj_roi, scl_roi)

    st.markdown("---")
    st.subheader("Analysis Result")
    st.metric(label="Calculated Pallor Score", value=f"{score:.2f}")

    # Risk evaluation using custom dataset thresholds
    if score > TH_HIGH:
        st.success("**Estimated Category: Lower Risk**\n\nConjunctiva color indicates adequate relative redness.")
    elif TH_LOW <= score <= TH_HIGH:
        st.warning("**Estimated Category: Mild Risk / Borderline**\n\nConsider consulting a healthcare provider if symptoms like fatigue persist.")
    else:
        st.error("**Estimated Category: Elevated Risk**\n\nConjunctiva color closely matches reference. A CBC blood test is recommended for clinical verification.")

    if ref_stats is not None:
        counts, by_gender = ref_stats
        with st.expander("View Reference Dataset Summary"):
            st.write({k: int(v) for k, v in counts.items()})
            st.dataframe(by_gender)
else:
    st.info("Please upload a photograph of the eyelid conjunctiva to begin analysis.")
