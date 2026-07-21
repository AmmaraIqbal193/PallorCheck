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
    "⚠️ **This is a prototype, not a diagnostic tool.** Color-based screening "
    "from uncalibrated photos is not clinically validated. Do not use this to "
    "make health decisions — see a clinician and get a blood test (CBC) for "
    "any real anemia concern."
)

# ---------------------------------------------------------------------------
# Reference dataset (used only to show context, NOT to set thresholds —
# doing that properly would require paired image+Hgb data to fit a model,
# which this script does not have)
# ---------------------------------------------------------------------------
@st.cache_data
def get_reference_standards():
    try:
        df = pd.read_excel('India.xlsx', sheet_name='Foglio1')
        normal_count = (df['Hgb'] >= 11.0).sum()
        mild_count = ((df['Hgb'] >= 8.0) & (df['Hgb'] < 11.0)).sum()
        severe_count = (df['Hgb'] < 8.0).sum()
        return normal_count, mild_count, severe_count
    except Exception:
        return None

ref_stats = get_reference_standards()

# ---------------------------------------------------------------------------
# Eye/conjunctiva localization
# Instead of blindly cropping the center of whatever image is uploaded, try
# to actually find an eye region with a Haar cascade. Falls back to a center
# crop only if no eye is detected, and tells the user which happened.
# ---------------------------------------------------------------------------
@st.cache_resource
def load_eye_cascade():
    cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
    return cv2.CascadeClassifier(cascade_path)

def locate_roi(img_rgb, eye_cascade):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40))

    h, w, _ = img_rgb.shape
    if len(eyes) == 0:
        # Fallback: center crop, but flagged so the user knows detection failed
        ymin, ymax = int(h * 0.35), int(h * 0.65)
        xmin, xmax = int(w * 0.35), int(w * 0.65)
        return img_rgb[ymin:ymax, xmin:xmax], False

    # Take the largest detected eye region
    ex, ey, ew, eh = max(eyes, key=lambda b: b[2] * b[3])
    # Focus on the lower-inner portion of the eye box, closer to where the
    # lower eyelid/conjunctiva tends to sit
    yin = ey + int(eh * 0.55)
    yax = ey + int(eh * 0.95)
    xin = ex + int(ew * 0.15)
    xax = ex + int(ew * 0.85)
    roi = img_rgb[max(yin, 0):yax, max(xin, 0):xax]

    if roi.size == 0:
        ymin, ymax = int(h * 0.35), int(h * 0.65)
        xmin, xmax = int(w * 0.35), int(w * 0.65)
        return img_rgb[ymin:ymax, xmin:xmax], False

    return roi, True

# ---------------------------------------------------------------------------
# Illumination correction
# Color channels shift a lot with lighting/white balance. We approximate a
# correction by using the brightest, low-saturation patch in the ROI as a
# rough "white reference" and rescaling channels against it. This is a crude
# stand-in for a real calibration card — not a substitute for one.
# ---------------------------------------------------------------------------
def rough_white_balance(roi_rgb):
    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)

    # candidate "white-ish" pixels: bright and not too saturated
    mask = (v > np.percentile(v, 85)) & (s < np.percentile(s, 40))
    if mask.sum() < 10:
        return roi_rgb  # not enough signal to correct, return as-is

    ref = roi_rgb[mask].reshape(-1, 3).mean(axis=0)
    ref = np.clip(ref, 1, 255)
    gain = 200.0 / ref  # target each channel toward a common gray level
    gain = np.clip(gain, 0.5, 2.0)  # avoid extreme corrections

    corrected = roi_rgb.astype(np.float32) * gain
    return np.clip(corrected, 0, 255).astype(np.uint8)

# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload Conjunctiva Photograph", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)

    eye_cascade = load_eye_cascade()
    conjunctiva_roi, detected = locate_roi(img_rgb, eye_cascade)
    corrected_roi = rough_white_balance(conjunctiva_roi)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Upload", use_column_width=True)
    with col2:
        st.image(corrected_roi, caption="ROI (white-balance corrected)", use_column_width=True)

    if not detected:
        st.info(
            "Could not automatically detect an eye in this photo — used a "
            "center crop instead. Results will be less reliable. Try a "
            "closer, well-lit photo of the eye."
        )

    # Feature extraction on the corrected ROI
    lab_roi = cv2.cvtColor(corrected_roi, cv2.COLOR_RGB2LAB)
    a_mean = float(np.mean(lab_roi[:, :, 1]))

    hsv_roi = cv2.cvtColor(corrected_roi, cv2.COLOR_RGB2HSV)
    sat_mean = float(np.mean(hsv_roi[:, :, 1]))

    pallor_score = (a_mean * 1.25) + (sat_mean * 0.75)

    st.markdown("---")
    st.subheader("Estimated Result (not a diagnosis)")
    st.metric(label="Pallor Index (uncalibrated)", value=f"{pallor_score:.2f}")

    st.caption(
        "These thresholds are placeholders, not fitted to validated paired "
        "image/Hgb data. Treat categories below as illustrative only."
    )

    if pallor_score > 142.0:
        st.success("**Estimated category: Lower risk**\n\nNo action implied by this tool either way — if you have symptoms, see a clinician.")
    elif 128.0 <= pallor_score <= 142.0:
        st.warning("**Estimated category: Possible mild risk**\n\nConsider mentioning this to a clinician if you have other anemia symptoms (fatigue, pallor, shortness of breath).")
    else:
        st.error("**Estimated category: Possible elevated risk**\n\nThis tool cannot confirm anemia. A CBC blood test is the reliable way to check.")

    if ref_stats is not None:
        normal_count, mild_count, severe_count = ref_stats
        st.markdown("---")
        st.caption(
            f"Reference dataset (India.xlsx) for context only — not used to "
            f"set the thresholds above: {normal_count} normal, {mild_count} "
            f"mild, {severe_count} severe by Hgb value."
        )
else:
    st.info("Please upload a photograph of the eyelid conjunctiva to begin.")
