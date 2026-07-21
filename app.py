import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Page Configuration
st.set_page_config(page_title="PallorCheck", page_icon="🩸", layout="centered")

st.title("🩸 PallorCheck")
st.write("Non-invasive Anemia Risk Screening via Conjunctiva Color Analysis")

# File Uploader
uploaded_file = st.file_uploader("Upload Palpebral Conjunctiva Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and process the uploaded image directly (skip secondary cropping for pre-cropped images)
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_rgb = img_array
    elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
        # Handle PNG files with transparency (RGBA)
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    else:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    # Display preview image
    st.image(image, caption="Analyzed Image Preview", use_column_width=True)

    # Calculate dynamic normalized redness index from the full image
    r_mean = np.mean(img_rgb[:, :, 0])
    g_mean = np.mean(img_rgb[:, :, 1])
    b_mean = np.mean(img_rgb[:, :, 2])

    total_intensity = r_mean + g_mean + b_mean
    if total_intensity > 0:
        anemia_index = (r_mean / total_intensity) * 100
    else:
        anemia_index = 0.0

    # Diagnostic Evaluation Output
    st.markdown("---")
    st.subheader("Diagnostic Evaluation")
    st.metric(label="Calculated Redness Index", value=f"{anemia_index:.2f}%")

    # Threshold evaluation logic
    if anemia_index > 33.5:
        st.success("**Diagnosis: NORMAL**\n\nAction: No immediate clinical action required.")
    elif 32.0 <= anemia_index <= 33.5:
        st.warning("**Diagnosis: MILD ANEMIA RISK**\n\nAction: Recommend dietary iron supplementation and routine monitoring.")
    else:
        st.error("**Diagnosis: SEVERE ANEMIA RISK**\n\nAction: Urgent referral for laboratory complete blood count (CBC).")
else:
    st.info("Please upload an image of the palpebral conjunctiva to begin the screening analysis.")
