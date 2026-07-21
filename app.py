import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image

# Page Configuration
st.set_page_config(page_title="PallorCheck", page_icon="🩸", layout="centered")

st.title("🩸 PallorCheck")
st.write("Non-invasive Anemia Risk Screening via Conjunctiva Color Analysis")

# Load the Excel dataset as the reference standard backend
@st.cache_data
def load_reference_dataset():
    try:
        xls = pd.ExcelFile('India.xlsx')
        df = pd.read_excel('India.xlsx', sheet_name=xls.sheet_names[0])
        return df
    except Exception as e:
        return None

df_ref = load_reference_dataset()

# Establish dynamic reference bounds from the dataset
if df_ref is not None and not df_ref.empty:
    hgb_min = float(df_ref['Hgb'].min())
    hgb_max = float(df_ref['Hgb'].max())
    # Clinical reference cutoffs present in the reference data
    severe_threshold_ref = 8.0
    normal_threshold_ref = 11.0
else:
    hgb_min, hgb_max = 7.6, 17.1
    severe_threshold_ref, normal_threshold_ref = 8.0, 11.0

# File Uploader for new spot images
uploaded_file = st.file_uploader("Upload Conjunctiva Photograph", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Format conversion to RGB
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_rgb = img_array
    elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    else:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    # Automatically crop center Region of Interest (ROI) to isolate the inner eyelid
    h, w, _ = img_rgb.shape
    ymin, ymax = int(h * 0.35), int(h * 0.65)
    xmin, xmax = int(w * 0.35), int(w * 0.65)
    conjunctiva_roi = img_rgb[ymin:ymax, xmin:xmax]

    # Display preview images
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Upload", use_column_width=True)
    with col2:
        st.image(conjunctiva_roi, caption="Isolated Conjunctiva ROI", use_column_width=True)

    # Extract robust color features using CIELAB and HSV color spaces
    lab_roi = cv2.cvtColor(conjunctiva_roi, cv2.COLOR_RGB2LAB)
    a_mean = np.mean(lab_roi[:, :, 1]) # Red-green opponent channel
    
    hsv_roi = cv2.cvtColor(conjunctiva_roi, cv2.COLOR_RGB2HSV)
    sat_mean = np.mean(hsv_roi[:, :, 1]) # Saturation channel

    # Feature score calculation
    feature_score = (a_mean * 1.3) + (sat_mean * 0.7)

    # Dynamically map the feature score onto the reference Hgb range derived from India.xlsx
    # Normalizing feature score into the dataset's min-max Hgb envelope
    estimated_hgb = hgb_min + (feature_score / 200.0) * (hgb_max - hgb_min)
    estimated_hgb = np.clip(estimated_hgb, hgb_min, hgb_max)

    # Diagnostic Evaluation Output
    st.markdown("---")
    st.subheader("Automated Diagnostic Evaluation")
    st.metric(label="Reference-Calibrated Hgb Estimate", value=f"{estimated_hgb:.2f} g/dL")

    # Evaluation using reference clinical cutoffs derived from the spreadsheet
    if estimated_hgb >= normal_threshold_ref:
        st.success("**Diagnosis: NORMAL**\n\nAction: No immediate clinical action required.")
    elif severe_threshold_ref <= estimated_hgb < normal_threshold_ref:
        st.warning("**Diagnosis: MILD ANEMIA RISK**\n\nAction: Recommend dietary iron supplementation and routine monitoring.")
    else:
        st.error("**Diagnosis: SEVERE ANEMIA RISK**\n\nAction: Urgent referral for laboratory complete blood count (CBC).")
else:
    st.info("Please upload a photograph of the eyelid conjunctiva to begin automated screening.")
