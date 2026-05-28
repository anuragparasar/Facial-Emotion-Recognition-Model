import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
import time
import base64

load_dotenv()

def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

img = get_base64("background.png")

st.markdown(
    f"""
    <style>

    /* Main app background */
    .stApp {{
        background-image:
            linear-gradient(rgba(0,0,0,0.78),
            rgba(0,0,0,0.78)),
            url("data:image/png;base64,{img}");

        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    /* Remove top black header */
    header[data-testid="stHeader"] {{
        background: transparent;
    }}

    /* Optional: hide deploy/menu */
    .stDeployButton {{
        display: none;
    }}

    #MainMenu {{
        visibility: hidden;
    }}

    /* Reduce top padding */
    .block-container {{
        padding-top: 2rem;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation",
    temperature=0.3
)
model1 = ChatHuggingFace(llm=llm)

model_path = hf_hub_download(
    repo_id="Anurag04/Fer_model",
    filename="fer_model.h5"
)

model = load_model(model_path)

template = """
The detected mood is {mood}.

Give exactly 3 helpful suggestions.
Keep each suggestion under 12 words.
Avoid medical claims.
Tone: supportive and practical.
"""
prompt_template = PromptTemplate(
    input_variables=["mood"],
    template=template
)


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
    ["Upload Image", "Take Picture"]
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
            final_prompt = prompt_template.format(mood=emotion)
            with st.spinner("Loading suggestions..."):
                result = model1.invoke(final_prompt)
                suggestion = result.content
            placeholder = st.empty()
            typed_text = ""
            for char in suggestion:
                typed_text += char
                placeholder.markdown(typed_text)
                time.sleep(0.01)


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
            final_prompt = prompt_template.format(mood=emotion)
            with st.spinner("Loading suggestions..."):
                result = model1.invoke(final_prompt)
                suggestion = result.content
            placeholder = st.empty()
            typed_text = ""
            for char in suggestion:
                typed_text += char
                placeholder.markdown(typed_text)
                time.sleep(0.01)