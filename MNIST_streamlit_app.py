import streamlit as st
import requests
from PIL import Image

# FastAPI endpoint
API_URL = "http://127.0.0.1:8000/predict/"

st.title("MNIST Digit Classifier 🎯")
st.write("Upload a digit image (28x28 grayscale or will be resized).")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=False)

    # Button to trigger prediction
    if st.button("Predict"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Digit: {result['predicted_class']}")
            st.write("Probabilities:")
            st.json(result["probabilities"])
        else:
            st.error(f"Error {response.status_code}: {response.text}")
