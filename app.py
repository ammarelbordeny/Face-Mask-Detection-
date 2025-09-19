import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import time
from collections import Counter

# Load model
model = load_model("Models/best_model.keras")

# Load face detection model
prototxtPath = "deploy.prototxt"
weightsPath = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)

# Classes
classes = ["mask_weared_incorrect", "with_mask", "without_mask"]

st.title("😷 Mask Detection Scanner")
st.write("Click **Start Scan** and wait 10 seconds to get the final result.")

# Start button
if st.button("Start Scan"):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    predictions = []

    start_time = time.time()
    duration = 10  # seconds

    while int(time.time() - start_time) < duration:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Could not access webcam")
            break

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                    (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Clamp coords
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                face = frame[startY:endY, startX:endX]
                if face.size == 0:
                    continue

                # Preprocess face
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = face / 255.0
                face = np.expand_dims(face, axis=0)

                preds = model.predict(face, verbose=0)[0]
                class_id = np.argmax(preds)
                conf = preds[class_id]
                label = f"{classes[class_id]} ({conf:.2f})"

                predictions.append(class_id)

                # Colors
                if class_id == 1:
                    color = (0, 255, 0)
                elif class_id == 2:
                    color = (0, 0, 255)
                else:
                    color = (0, 165, 255)

                # Draw on frame
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # Show video during scan
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

    cap.release()

    # Final result
    if predictions:
        final_class_id = Counter(predictions).most_common(1)[0][0]
        st.success(f"✅ Final Result: **{classes[final_class_id]}**")
    else:
        st.warning("⚠️ No face detected during scan.")
