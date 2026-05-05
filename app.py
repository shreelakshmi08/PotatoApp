import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from PIL import Image

# =========================
# MODEL (MUST MATCH TRAINING)
# =========================
model = Sequential([
    Input(shape=(128,128,3)),

    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')
])

# Load weights
model.load_weights("potato.weights.h5")

# Class labels
class_names = ['Early_Blight', 'Healthy', 'Late_Blight']   

# =========================
# STREAMLIT UI
# =========================
st.title("🥔 Potato Disease Detection")

file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

if file is not None:
    img = Image.open(file).convert("RGB")
    img = img.resize((128,128))

    st.image(img, caption="Uploaded Image", width=150)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")