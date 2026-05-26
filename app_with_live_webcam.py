import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="Anurag04/Fer_model",
    filename="fer_model.h5"
)

model = load_model(model_path)

st.title("FACIAL EMOTION RECOGNITION")

class_names = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def predict_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        return frame, None, None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48))
    face = face / 255.0
    face = face.reshape(1, 48, 48, 1)

    prediction = model.predict(face, verbose=0)
    predicted_class = np.argmax(prediction)
    emotion = class_names[predicted_class]

    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(
        frame,
        emotion,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    df = pd.DataFrame(prediction, columns=class_names)

    return frame, emotion, df


option = st.radio(
    "Choose input mode",
    ["Upload Image", "Take Picture", "Live Webcam"]
)


if option == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        output_frame, emotion, df = predict_emotion(frame)

        st.image(
            cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB),
            caption="Detected Face"
        )

        if emotion is None:
            st.write("No face detected")
        else:
            st.subheader("Predicted Probabilities:")
            st.dataframe(df)
            st.subheader(f"Predicted Emotion: {emotion}")


elif option == "Take Picture":

    camera_image = st.camera_input("Take a picture")

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
        img_array = np.array(image)
        frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        output_frame, emotion, df = predict_emotion(frame)

        st.image(
            cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB),
            caption="Detected Face"
        )

        if emotion is None:
            st.write("No face detected")
        else:
            st.subheader("Predicted Probabilities:")
            st.dataframe(df)
            st.subheader(f"Predicted Emotion: {emotion}")


elif option == "Live Webcam":

    class EmotionProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")

            output_frame, emotion, df = predict_emotion(img)

            return av.VideoFrame.from_ndarray(output_frame, format="bgr24")

    webrtc_streamer(
        key="emotion-live",
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )