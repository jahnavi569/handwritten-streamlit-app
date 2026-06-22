import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os

st.title(" Handwritten Alphabet Recognition (A–Z)")

# Show debug (important for beginners)
st.write("Current folder:", os.getcwd())

# Load model
model = load_model("notebook/handwritten_alphabet_cnn.keras", compile=False)

# Label mapping
word_dict = {i: chr(65+i) for i in range(26)}

# Preprocess image
def preprocess(img):
    img = img.convert('L')
    img = img.resize((28, 28))
    img = np.array(img)
    img = 255 - img
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

# Upload image
uploaded_file = st.file_uploader("Upload a handwritten letter", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=200)

    processed = preprocess(image)

    prediction = model.predict(processed, verbose=0)

    pred_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"Predicted Letter: {word_dict[pred_class]}")
    st.write(f"Confidence: {confidence*100:.2f}%")