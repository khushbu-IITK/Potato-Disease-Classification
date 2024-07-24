import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# Load the model
model_path = 'potatoes.h5'  # Update this to your model's path
if not os.path.exists(model_path):
    st.error(f"Model path does not exist: {model_path}")
else:
    st.write(f"Loading model from: {model_path}")

model = tf.keras.models.load_model(model_path)


# Saving the model
model.save('new_model_path.h5')

# Loading the model
model = tf.keras.models.load_model('new_model_path.h5')


# Define image preprocessing function
def preprocess_image(image):
    # Convert the image to RGB just in case it's a different mode
    image = image.convert("RGB")
    
    # Resize the image to the input size expected by your model
    image = image.resize((224, 224))  # Change (224, 224) to your model's input size
    
    # Convert the image to a numpy array and normalize it
    image_array = np.array(image) / 255.0
    
    # Expand dimensions to match the model's input shape
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

# Streamlit app
st.title("Disease Classification App")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("Classifying...")
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Make prediction
    predictions = model.predict(preprocessed_image)
    
    # Assuming your model's output is a single prediction with a confidence score
    predicted_class = np.argmax(predictions, axis=1)
    confidence = np.max(predictions) * 100
    
    st.write(f"Predicted class: {predicted_class[0]}")
    st.write(f"Confidence: {confidence:.2f}%")
