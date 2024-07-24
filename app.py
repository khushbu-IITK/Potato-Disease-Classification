import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os

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

# Check if the model path exists
if not os.path.exists(model_full_path):
    st.error(f"Model path does not exist: {model_full_path}")
else:
    st.write(f"Loading model from: {model_full_path}")

    try:
        # Import your custom objects if any (if not, leave it empty)
        # from my_custom_objects import CustomLayer, custom_loss_function

        # Load the model with custom objects if necessary
        model = tf.keras.models.load_model(model_full_path)  # , custom_objects={
        #    'CustomLayer': CustomLayer,
        #    'custom_loss_function': custom_loss_function
        # })

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

    # Predict the class and confidence
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100

    # Display the result
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
