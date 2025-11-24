import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

MODEL_PATH = "/home/gunin/Desktop/Project/AI-gen-detection/runs/models/20251122-153548/final_model.h5"
IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = ["Deepfake", "Real"]

st.set_page_config(
    page_title="AI vs Real Classifier",
    page_icon="🕵️",
    layout="centered"
)

st.title("🕵️ AI vs Real Classifier")
st.markdown("Upload a face image to detect AI-generated artifacts.")

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"❌ Model file not found at: {path}")
        st.info("Please update 'MODEL_PATH' in app.py to your actual .h5 file location.")
        return None
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

with st.spinner("Loading Model..."):
    model = load_model(MODEL_PATH)

if model:
    st.success("✅ Model loaded successfully")

def preprocess_image(image):
    image = ImageOps.fit(image, (IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = img_array / 255.0
    img_tensor = np.expand_dims(img_array, axis=0)
    return img_tensor, image

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None and model is not None:
    image_raw = Image.open(uploaded_file).convert("RGB")
    processed_tensor, display_image = preprocess_image(image_raw)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(display_image, caption="Input Image", use_container_width=True)

    with col2:
        st.subheader("Analysis")
        
        prediction = model.predict(processed_tensor)
        score = prediction[0][0]
        
        if score < 0.5:
            label = CLASS_NAMES[0] 
            confidence = 1 - score
            color_cls = "red"
            emoji = "🤖"
        else:
            label = CLASS_NAMES[1]
            confidence = score
            color_cls = "green"
            emoji = "👤"
            
        st.markdown(f"### Result: :{color_cls}[{label} {emoji}]")
        st.write(f"Confidence: **{confidence:.2%}**")
        st.progress(float(confidence))
        
        with st.expander("View Raw Output"):
            st.write(f"Raw Sigmoid Probability: {score:.6f}")
            st.write(f"0.0 = Certain Deepfake | 1.0 = Certain Real")