import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="PallorCheck", page_icon="🩸", layout="centered")
st.title("🩸 PallorCheck")
st.write("Non-invasive Anemia Risk Screening via Conjunctiva Color Analysis")

uploaded_file = st.file_uploader("Upload Palpebral Conjunctiva Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_rgb = img_array
    else:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)

    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    with col2:
        st.image(img_hsv, caption="HSV Color Space", use_column_width=True)

    r_mean = np.mean(img_rgb[:, :, 0])
    g_mean = np.mean(img_rgb[:, :, 1])
    anemia_index = (1.0 * r_mean) - (1.0 * g_mean)

    st.markdown("---")
    st.subheader("Diagnostic Evaluation")
    st.metric(label="Calculated Anemia Index", value=f"{anemia_index:.2f}")

    if anemia_index > 13.0:
        st.success("**Diagnosis: NORMAL**\n\nAction: No immediate clinical action required.")
    elif 6.0 <= anemia_index <= 13.0:
        st.warning("**Diagnosis: MILD ANEMIA RISK**\n\nAction: Recommend dietary iron supplementation and routine monitoring.")
    else:
        st.error("**Diagnosis: SEVERE ANEMIA RISK**\n\nAction: Urgent referral for laboratory complete blood count (CBC).")
