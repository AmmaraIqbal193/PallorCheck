import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title="PallorCheck", page_icon="🩸", layout="centered")
st.title("🩸 PallorCheck")
st.write("Experimental conjunctiva color analysis — NOT a medical device.")

st.warning(
    "⚠️ **Prototype, not a diagnostic tool.** The thresholds shown below are "
    "placeholders until calibrated on real Hgb-labeled photos (see the "
    "Calibration panel). Do not use this to make health decisions — a CBC "
    "blood test is the reliable way to check for anemia."
)

# ---------------------------------------------------------------------------
# Reference dataset — shown for context only (NOT used to calibrate the
# photo color formula below, since this file has no images — just Hgb,
# Gender, Age). What it CAN do properly is classify each row using real,
# gender-specific WHO cutoffs instead of one flat number for everyone.
#
# WHO Guideline on Haemoglobin Cutoffs to Define Anaemia (2024), adult
# non-pregnant thresholds:
#   Women (>=15y): normal >=12.0 | mild 11.0-11.9 | moderate 8.0-10.9 | severe <8.0
#   Men   (>=15y): normal >=13.0 | mild 11.0-12.9 | moderate 8.0-10.9 | severe <8.0
# Source: WHO 2024 guideline (iris.who.int/bitstream/handle/10665/376196/9789240088542-eng.pdf)
# ---------------------------------------------------------------------------
def classify_who(hgb, gender):
    g = str(gender).strip().upper()
    if g == 'F':
        if hgb >= 12.0: return 'Normal'
        if hgb >= 11.0: return 'Mild'
        if hgb >= 8.0:  return 'Moderate'
        return 'Severe'
    else:  # 'M' (or unspecified — use the higher male cutoff as default)
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
# Eye detection
# ---------------------------------------------------------------------------
@st.cache_resource
def load_eye_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def detect_eye_box(img_rgb, eye_cascade):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40))
    if len(eyes) == 0:
        return None
    return max(eyes, key=lambda b: b[2] * b[3])  # largest detected eye

def get_regions(img_rgb, eye_box):
    """
    Returns (conjunctiva_patch, sclera_patch, detected_bool).
    conjunctiva: lower-inner eyelid area (where pallor is assessed clinically)
    sclera: upper-white-of-eye area, used as an internal lighting reference
    Falls back to fixed proportional crops of the whole image if no eye found.
    """
    h, w, _ = img_rgb.shape
    if eye_box is None:
        cy0, cy1 = int(h * 0.40), int(h * 0.60)
        cx0, cx1 = int(w * 0.40), int(w * 0.60)
        sy0, sy1 = int(h * 0.25), int(h * 0.35)
        sx0, sx1 = int(w * 0.40), int(w * 0.60)
        return img_rgb[cy0:cy1, cx0:cx1], img_rgb[sy0:sy1, sx0:sx1], False

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
# Noise-robust feature extraction (median, not mean — resists eyelash/glare
# outlier pixels common in phone-camera close-ups)
# ---------------------------------------------------------------------------
def extract_features(patch_rgb):
    lab = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2HSV)
    a_med = float(np.median(lab[:, :, 1]))
    l_med = float(np.median(lab[:, :, 0]))
    sat_med = float(np.median(hsv[:, :, 1]))
    return a_med, l_med, sat_med

def relative_pallor_score(conj_rgb, scl_rgb):
    """
    Measures conjunctiva redness RELATIVE to the sclera in the same photo.
    Because both patches share the same lighting, this cancels most of the
    illumination/white-balance error that absolute color values suffer from.
    Lower score => conjunctiva closer to sclera color => paler => higher
    estimated risk. Higher score => conjunctiva noticeably redder => lower risk.
    """
    a_conj, l_conj, sat_conj = extract_features(conj_rgb)
    a_scl, l_scl, sat_scl = extract_features(scl_rgb)

    a_diff = a_conj - a_scl
    sat_diff = sat_conj - sat_scl

    score = (a_diff * 1.25) + (sat_diff * 0.75)
    details = {
        "a_conjunctiva": a_conj, "a_sclera": a_scl, "a_diff": a_diff,
        "sat_conjunctiva": sat_conj, "sat_sclera": sat_scl, "sat_diff": sat_diff,
    }
    return score, details

# ---------------------------------------------------------------------------
# Session state for user-adjustable thresholds (set via Calibration panel)
# ---------------------------------------------------------------------------
if "thr_high" not in st.session_state:
    st.session_state.thr_high = 20.0
if "thr_low" not in st.session_state:
    st.session_state.thr_low = 8.0

# ---------------------------------------------------------------------------
# Calibration panel
# ---------------------------------------------------------------------------
with st.expander("🔧 Calibration (recommended before trusting any result)"):
    st.write(
        "Upload a few photos with **known** Hgb values or severity labels to "
        "see where this method's scores actually fall for real cases, then "
        "set the thresholds to match — instead of relying on guessed numbers."
    )
    calib_files = st.file_uploader(
        "Upload calibration photos", type=["png", "jpg", "jpeg"],
        accept_multiple_files=True, key="calib"
    )
    if calib_files:
        eye_cascade = load_eye_cascade()
        rows = []
        for f in calib_files:
            img = np.array(Image.open(f).convert("RGB"))
            box = detect_eye_box(img, eye_cascade)
            conj, scl, detected = get_regions(img, box)
            score, _ = relative_pallor_score(conj, scl)
            rows.append({"file": f.name, "score": round(score, 2), "eye_detected": detected})
        calib_df = pd.DataFrame(rows)
        st.dataframe(calib_df, use_container_width=True)
        st.caption(
            "Manually note the Hgb value for each file elsewhere, then pick "
            "thresholds below that separate them the way you'd expect."
        )

    st.session_state.thr_high = st.slider(
        "Score above this = lower risk", -20.0, 60.0, st.session_state.thr_high
    )
    st.session_state.thr_low = st.slider(
        "Score below this = elevated risk", -20.0, 60.0, st.session_state.thr_low
    )
    if st.session_state.thr_low > st.session_state.thr_high:
        st.error("Lower-risk threshold must be higher than elevated-risk threshold.")

# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload Conjunctiva Photograph", type=["png", "jpg", "jpeg"], key="main")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)

    eye_cascade = load_eye_cascade()
    eye_box = detect_eye_box(img_rgb, eye_cascade)
    conj_roi, scl_roi, detected = get_regions(img_rgb, eye_box)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image, caption="Original", use_column_width=True)
    with col2:
        st.image(conj_roi, caption="Conjunctiva patch", use_column_width=True)
    with col3:
        st.image(scl_roi, caption="Sclera reference", use_column_width=True)

    if not detected:
        st.info(
            "Could not detect an eye automatically — used fallback crops. "
            "Results are much less reliable; try a closer, well-lit, "
            "in-focus photo of the eye."
        )

    score, details = relative_pallor_score(conj_roi, scl_roi)

    st.markdown("---")
    st.subheader("Estimated Result (not a diagnosis)")
    st.metric(label="Relative Pallor Score", value=f"{score:.2f}")

    with st.expander("See raw feature values"):
        st.json(details)

    thr_high = st.session_state.thr_high
    thr_low = st.session_state.thr_low

    if score > thr_high:
        st.success("**Estimated category: Lower risk**\n\nConjunctiva notably redder than the sclera reference.")
    elif thr_low <= score <= thr_high:
        st.warning("**Estimated category: Possible mild risk**\n\nConsider mentioning to a clinician if other symptoms are present (fatigue, breathlessness, pale skin).")
    else:
        st.error("**Estimated category: Possible elevated risk**\n\nConjunctiva color close to the sclera reference. A CBC blood test is the reliable way to confirm.")

    if ref_stats is not None:
        counts, by_gender = ref_stats
        st.markdown("---")
        st.caption(
            "Reference dataset (India.xlsx), classified using WHO 2024 "
            "gender-specific Hgb cutoffs — shown for population context only, "
            "not used to set the photo score thresholds above:"
        )
        st.write({k: int(v) for k, v in counts.items()})
        with st.expander("Breakdown by gender"):
            st.dataframe(by_gender)
else:
    st.info("Please upload a photograph of the eyelid conjunctiva to begin.")
