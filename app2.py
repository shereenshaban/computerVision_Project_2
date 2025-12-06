import streamlit as st
import cv2
import numpy as np
from PIL import Image

def detect_faces(img):

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    for x,y,w,h in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 10)

    return img

st.title("Face Detction")

upload = st.file_uploader("Choose an Image", type=['jpg', "jpeg"])
if upload is not None:
    image = Image.open(upload)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    result = detect_faces(image)

    st.image(result, channels="BGR")