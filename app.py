import streamlit as st
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="Distraction Sense", layout="centered")
st.title("📚 Distraction Sense – Focus Detection")

mp_face = mp.solutions.face_mesh

class FaceDetector(VideoTransformerBase):
    def __init__(self):
        self.face_mesh = mp_face.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.status = "Focused"

    def transform(self, frame):
        img = frame.to_ndarray(format="rgb24")
        results = self.face_mesh.process(img)

        if results.multi_face_landmarks:
            self.status = "Focused"
        else:
            self.status = "Distracted"

        return img

ctx = webrtc_streamer(
    key="distraction-sense",
    video_transformer_factory=FaceDetector,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if ctx.video_transformer:
    st.success(f"Current Status: {ctx.video_transformer.status}")
