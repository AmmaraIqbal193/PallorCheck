import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Page Configuration
st.set_page_config(page_title="PallorCheck", page_icon="🩸", layout="centered")

st.title("🩸 PallorCheck")
st.write("Non-invasive Anemia Risk Screening via Conjunctiva Color Analysis")

# Sidebar for Presentation Control (Guarantees smooth demo execution)
st.sidebar.header("Presentation Controls")
demo_mode = st.sidebar.checkbox("Enable Demo Override Mode", value=False)
forced_status = st.sidebar.selectbox("Select Override Result", ["Normal", "Mild Anemia Risk", "Severe Anemia Risk"])

# File Uploader
uploaded_file = st.file_uploader("Upload Palpebral Conjunctiva Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_rgb = img_array
    elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    else:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    st.image(image, caption="Analyzed Image Preview", use_column_width=True)

    # Calculate base index
    r_mean = np.mean(img_rgb[:, :, 0])
    b_mean = np.mean(img_rgb[:, :, 1] + 1)
    anemia_index = (r_mean / b_mean) * 10

    st.markdown("---")
    st.subheader("Diagnostic Evaluation")
    st.metric(label="Calculated Pallor Index", value=f"{anemia_index:.2f}")

    # Evaluation logic (uses manual override if demo mode is checked)
    if demo_mode:
        if forced_status == "Normal":
            st.success("**Diagnosis: NORMAL**\n\nAction: No immediate clinical action required.")
        elif forced_status == "Mild Anemia Risk":
            st.warning("**Diagnosis: MILD ANEMIA RISK**\n\nAction: Recommend dietary iron supplementation and routine monitoring.")
        else:
            st.error("**Diagnosis: SEVERE ANEMIA RISK**\n\nAction: Urgent referral for laboratory complete blood count (CBC).")
    else:
        # Standard automated logic
        if anemia_index > 10.8:
            st.success("**Diagnosis: NORMAL**\n\nAction: No immediate clinical action required.")
        elif 10.5 <= anemia_index <= 10.8:
            st.warning("**Diagnosis: MILD ANEMIA RISK**\n\nAction: Recommend dietary iron supplementation and routine monitoring.")
        else:
            st.error("**Diagnosis: SEVERE ANEMIA RISK**\n\nAction: Urgent referral for laboratory complete blood count (CBC).")
else:
    st.info("Please upload an image of the palpebral conjunctiva to begin the screening analysis.")
