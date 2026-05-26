# Facial-Emotion-Recognition-Model
## 🧠 Model

The trained `.keras` model is hosted on Hugging Face.

### 📥 Download Model

```txt
https://huggingface.co/Anurag04/Fer_model/blob/main/fer_model.h5
```
## 🎥 Live Webcam Support

A real-time live webcam version using `streamlit-webrtc` was also implemented locally.

Due to current compatibility issues between Streamlit Cloud, Python 3.14, and WebRTC dependencies, the live webcam feature is not enabled in the deployed cloud version.

However, the local version supports:
- Real-time webcam emotion detection
- Live frame processing
- Continuous emotion prediction

The deployed application currently supports:
- Image Upload
- Camera Capture

