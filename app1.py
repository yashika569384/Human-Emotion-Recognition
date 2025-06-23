import streamlit as st
import cv2
from keras.models import load_model
import numpy as np
from PIL import Image
import gdown
import os
from tensorflow import keras

# Model file path
model_path = "facialemotionmodel1.keras"

# Download model from Google Drive if not already present
if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=1NJCsCzDMtKKJqcnzKf9Kvbiyf0vyeVXo"
    gdown.download(url, model_path, quiet=False)

# Load model and face detector
model = load_model("facialemotionmodel1.keras")
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Streamlit UI
st.title("Real-Time Facial Emotion Recognition")
st.markdown("This app detects facial emotions using your webcam in real-time.")

# Button to start camera
run = st.checkbox('Start Camera')

# Camera logic
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    success, frame = camera.read()
    if not success:
        st.error("Failed to access camera.")
        break
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            face = cv2.resize(roi_gray, (48, 48))
            face = face / 255.0
            face = np.expand_dims(face, axis=(0, -1))

            prediction = model.predict(face)
            emotion = emotion_labels[np.argmax(prediction)]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

camera.release()

