import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Page Configuration
st.set_page_config(page_title="PallorCheck", page_icon="🩸", layout="centered")

st.title("🩸 PallorCheck")
st.write("Non-invasive Anemia Risk Screening via Conjunctiva Color Analysis")

# File Uploader for automated screening
uploaded_file = st.file_uploader("Upload Conjunctiva Image for Automated Screening", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_rgb = img_array
    elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    else:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    # Automatically crop the center region of interest (ROI) to isolate the conjunctiva
    h, w, _ = img_rgb.shape
    ymin, ymax = int(h * 0.3), int(h * 0.7)
    xmin, xmax = int(w * 0.3), int(w * 0.7)
    conjunctiva_roi = img_rgb[ymin:ymax, xmin:xmax]

    # Display preview images
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
    with col2:
        st.image(conjunctiva_roi, caption="Isolated Conjunctiva ROI", use_column_width=True)

    # Calculate Normalized Red Chromaticity (Independent of lighting brightness/flash glare)
    r_mean = np.mean(conjunctiva_roi[:, :, 0]).astype(float)
    g_mean = np.mean(conjunctiva_roi[:, :, 1]).astype(float)
    b_mean = np.mean(conjunctiva_roi[:, :, 2]).astype(float)
    
    total = r_mean + g_mean + b_mean
    if total > 0:
        red_chromaticity = (r_mean / total) * 100
    else:
        red_chromaticity = 0.0

    # Diagnostic Evaluation Output
    st.markdown("---")
    st.subheader("Automated Diagnostic Evaluation")
    st.metric(label="Calculated Red Chromaticity Index", value=f"{red_chromaticity:.2f}%")

    # Automated clinical threshold evaluation based on chromaticity proportion
    if red_chromaticity > 38.5:
        st.success("**Diagnosis: NORMAL**\n\nAction: No immediate clinical action required.")
    elif 36.5 <= red_chromaticity <= 38.5:
        st.warning("**Diagnosis: MILD ANEMIA RISK**\n\nAction: Recommend dietary iron supplementation and routine monitoring.")
    else:
        st.error("**Diagnosis: SEVERE ANEMIA RISK**\n\nAction: Urgent referral for laboratory complete blood count (CBC).")
else:
    st.info("Please upload an image of the eyelid conjunctiva to run automated screening.")
