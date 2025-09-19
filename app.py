import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import time
from collections import Counter

# Load model
model = load_model("Models/best_model.keras")
classes = ["mask_weared_incorrect", "with_mask", "without_mask"]

st.title("😷 Mask Detection Scanner")
st.write("Click **Start Scan** and take multiple photos in 10 seconds.")

# Start Scan button
if st.button("Start Scan"):
    st.write("📸 Take photos repeatedly for 10 seconds.")
    predictions = []
    start_time = time.time()
    duration = 10  # seconds

    # Loop for 10 seconds
    while int(time.time() - start_time) < duration:
        img_file_buffer = st.camera_input("Take a photo")
        
        if img_file_buffer is not None:
            image = Image.open(img_file_buffer)
            st.image(image, caption="Captured Image", use_column_width=True)
            
            # Preprocess image
            face = image.resize((224, 224))
            face = np.array(face) / 255.0
            face = np.expand_dims(face, axis=0)
            
            preds = model.predict(face, verbose=0)[0]
            class_id = np.argmax(preds)
            predictions.append(class_id)
    
    # Final result
    if predictions:
        final_class_id = Counter(predictions).most_common(1)[0][0]
        st.success(f"✅ Final Result: **{classes[final_class_id]}**")
    else:
        st.warning("⚠️ No photos captured during scan.")
