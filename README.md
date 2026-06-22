## Handwritten Alphabet Recognition (A–Z) using CNN + Streamlit
# Overview
This project is a deep learning-based web application that recognizes handwritten English alphabets (A–Z). It uses a Convolutional Neural Network (CNN) trained on a large A–Z handwritten dataset and is deployed using Streamlit for real-time predictions.

# Problem Statement
Manual recognition of handwritten characters is slow and error-prone. This project automates the process using deep learning for accurate classification.

# Dataset
- A–Z Handwritten Alphabets Dataset
- 372,000+ grayscale images
- Image size: 28×28 pixels
- 26 classes (A to Z)

# Tech Stack
- Python
- TensorFlow / Keras
- NumPy
- Pillow (PIL)
- Streamlit
- Matplotlib
# Features
- Upload handwritten image
- Predict alphabet (A–Z) in real time
- Displays confidence score
- Simple and interactive web interface
# Performance
- Training Accuracy: ~99%
- Validation Accuracy: ~99%
- Works well on clean handwritten inputs
# Project Structure
Handwritten/
│
├── app.py
├── notebook/
│     └── handwritten_alphabet_cnn.keras
|      └── handwritten.ipynb
├── data/
├── test images/
├── requirements.txt
└── README.md
# How to Run
1. Install dependencies
pip install -r requirements.txt
2. Run Streamlit app
streamlit run app.py

