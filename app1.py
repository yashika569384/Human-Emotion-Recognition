import streamlit as st
import av
import cv2
import numpy as np
import os
import gdown
from keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase


model_path = "facialemotionmodel1.keras"

if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=1NJCsCzDMtKKJqcnzKf9Kvbiyf0vyeVXo"
    gdown.download(url, model_path, quiet=False)


model = load_model(model_path)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load OpenCV's face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            face = cv2.resize(roi_gray, (48, 48))
            face = face / 255.0
            face = np.expand_dims(face, axis=(0, -1))  # (1, 48, 48, 1)

            prediction = model.predict(face, verbose=0)
            emotion = emotion_labels[np.argmax(prediction)]

            # Draw rectangle and label
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)

        return img


st.title("😊 Real-Time Facial Emotion Recognition")
st.markdown("This app uses your **browser webcam** to detect facial emotions using deep learning.")


webrtc_streamer(
    key="emotion-detection",
    video_processor_factory=EmotionDetector,
    media_stream_constraints={"video": True, "audio": False}
)
