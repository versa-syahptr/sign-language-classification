import streamlit as st
import numpy as np
from PIL import Image
from streamlit_cropper import st_cropper

import cv2
import numpy as np
from pred import model_prediction
import joblib
import string

classes = list(string.ascii_uppercase)

@st.cache_resource
def load_models():
    lgbm_model = joblib.load('tubes-pcd-LightGBM.pkl')
    svm_model = joblib.load('tubes-pcd-SVM.pkl')
    return lgbm_model, svm_model

with st.spinner("Loading models..."):
    lgbm_model, svm_model = load_models()

st.title("Hand Gesture Recognition")
source = st.radio("Image Source", ["Upload Image", "Camera"])

if source == "Upload Image":
    image = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
else:
    image = st.camera_input("Camera", disabled=source != "Camera")

if image:
    img = Image.open(image)
    st.write("**Please crop the image to focus on the hand**")
    cropped_img = st_cropper(img, aspect_ratio=(1,1))
    gray_cropped_img = cropped_img.convert('L')
    small_cropped_img = np.array(gray_cropped_img.resize((28, 28)))
    # clahe_image = apply_clahe(small_cropped_img)

    left, center, right = st.columns(3, vertical_alignment='center')
    left.image(cropped_img, caption="Cropped Image")
    center.image(gray_cropped_img, caption="Gray Image")
    right.image(small_cropped_img, caption="Small Image")

with st.sidebar:
    use_clahe = st.checkbox("Use CLAHE?")
    classifier = st.radio("**Classifier**", ["LightGBM", "SVM"])
    st.write(f"**Selected Model:** \n\n`{classifier}`")
    predict = st.button("Predict", disabled=not image, icon='🔍')
    if predict:
        with st.spinner("Predicting..."):
            if classifier == "LightGBM":
                prediction = model_prediction(small_cropped_img, lgbm_model, use_clahe)
            else:
                prediction = model_prediction(small_cropped_img, svm_model, use_clahe)
            st.write(f"**Predicted Class Index:** `{prediction[0]}`")
            st.write(f"**Predicted Character:** {classes[prediction[0]]}")

