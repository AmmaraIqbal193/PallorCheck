import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Page Configuration
st.set_page_config(page_title="PallorCheck", page_icon="🩸", layout="centered")

st.title("🩸 PallorCheck")
st.write("Non-invasive Anemia Risk Screening via Conjunctiva Color Analysis")

# File Uploader for real-world user images
uploaded_file = st.file_uploader("Upload Eye Conjunctiva Photograph", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Standardize image format to RGB
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_rgb = img_array
    elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    else:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    # Automatically crop the center region of interest (ROI) to isolate the inner eyelid
    h, w, _ = img_rgb.shape
    ymin, ymax = int(h * 0.35), int(h * 0.65)
    xmin, xmax = int(w * 0.35), int(w * 0.65)
    conjunctiva_roi = img_rgb[ymin:ymax, xmin:xmax]

    # Display preview images for the user
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Photograph", use_column_width=True)
    with col2:
        st.image(conjunctiva_roi, caption="Isolated Conjunctiva ROI", use_column_width=True)

    # 1. Convert to HSV (Hue, Saturation, Value) to isolate true color saturation from lighting glare
    hsv_roi = cv2.cvtColor(conjunctiva_roi, cv2.COLOR_RGB2HSV)
    saturation_mean = np.mean(hsv_roi[:, :, 1]) # Measures color vividness (low in anemia)
    
    # 2. Convert to CIELAB color space (Industry standard for physiological color measurement)
    lab_roi = cv2.cvtColor(conjunctiva_roi, cv2.COLOR_RGB2LAB)
    a_mean = np.mean(lab_roi[:, :, 1]) # 'a' channel represents Green-to-Red opponent axis (higher = more red)

    # Combine physiological color features into a robust Anemia Index
    # Healthy tissue has high red-opponent axis ('a') and high color saturation
    anemia_index = (a_mean * 1.2) + (saturation_mean * 0.8)

    # Diagnostic Evaluation Output
    st.markdown("---")
    st.subheader("Automated Diagnostic Evaluation")
    st.metric(label="Calculated Pallor Score", value=f"{anemia_index:.2f}")

    # Automated clinical threshold evaluation for real-world user photos
    if anemia_index > 145.0:
        st.success("**Diagnosis: NORMAL**\n\nAction: No immediate clinical action required.")
    elif 130.0 <= anemia_index <= 145.0:
        st.warning("**Diagnosis: MILD ANEMIA RISK**\n\nAction: Recommend dietary iron supplementation and routine monitoring.")
    else:
        st.error("**Diagnosis: SEVERE ANEMIA RISK**\n\nAction: Urgent referral for laboratory complete blood count (CBC).")
else:
    st.info("Please upload a clear photograph of the eyelid conjunctiva to begin automated screening.")
