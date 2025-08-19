import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

model = tf.keras.models.load_model("mask_model.h5")

st.title("😷 Face Mask Detection with CNN")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)


    # Preprocess
    img = image.resize((128,128))
    img = img.convert("RGB")
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        st.error("❌ No Mask Detected")
    else:
        st.success("✅ Mask Detected")
