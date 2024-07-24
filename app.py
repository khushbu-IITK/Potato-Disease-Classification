import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

# Define your model path
model_path = 'Potato_classification_model.h5'  # Update this to your model's path

# Function to get script directory (useful for relative paths)
def get_script_directory():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()

# Full path to the model
model_full_path = os.path.join(get_script_directory(), model_path)

# Initialize model variable
model = None

# Check if the model path exists
if not os.path.exists(model_full_path):
    st.error(f"Model path does not exist: {model_full_path}")
else:
    st.write(f"Loading model from: {model_full_path}")

    try:
        # Custom object dictionary
        custom_objects = {
            'SparseCategoricalCrossentropy': SparseCategoricalCrossentropy
        }

        # Load the model with custom objects
        model = tf.keras.models.load_model(model_full_path, custom_objects=custom_objects)

        st.write("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((224, 224))  # Adjust to your model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Streamlit app interface
st.title("Potato Disease Classification")
st.write("Upload an image of a potato leaf to classify the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_array = preprocess_image(img)

    # Ensure model is loaded before making predictions
