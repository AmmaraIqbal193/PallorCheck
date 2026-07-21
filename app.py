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
    # Load and process the uploaded image directly
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_rgb = img_array
    elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    else:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    # Display preview image
    st.image(image, caption="Analyzed Image Preview", use_column_width=True)

    # Convert RGB to HSV to measure true color saturation (ignores lighting glare)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    r_mean = np.mean(img_rgb[:, :, 0])
    g_mean = np.mean(img_rgb[:, :, 1])
    s_mean = np.mean(img_hsv[:, :, 1]) # Saturation channel

    # Robust Anemia Index combining Red dominance and Saturation 
    # (Higher score = healthier/normal tissue; Lower score = pale/anemic tissue)
    anemia_index = (r_mean - g_mean) + (s_mean * 0.5)

    # Diagnostic Evaluation Output
    st.markdown("---")
    st.subheader("Diagnostic Evaluation")
    st.metric(label="Calculated Pallor Index", value=f"{anemia_index:.2f}")

    # Threshold evaluation logic based on combined color-saturation space
    if anemia_index > 15:
        st.success("**Diagnosis: NORMAL**\n\nAction: No immediate clinical action required.")
    elif 10 <= anemia_index <= 15:
        st.warning("**Diagnosis: MILD ANEMIA RISK**\n\nAction: Recommend dietary iron supplementation and routine monitoring.")
    else:
        st.error("**Diagnosis: SEVERE ANEMIA RISK**\n\nAction: Urgent referral for laboratory complete blood count (CBC).")
else:
    st.info("Please upload an image of the palpebral conjunctiva to begin the screening analysis.")
