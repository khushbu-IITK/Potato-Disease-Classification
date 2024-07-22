# -*- coding: utf-8 -*-
"""app.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1csBhJii5ZwPcg8676GLrEjDbkU90mVWt
"""
# pip install --upgrade pip
# pip install streamlit

import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load the model
model = tf.keras.models.load_model('Potato_classification_model.keras')

st.title("Potato Leaf Disease Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

class_name = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Defining predict function
def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_name[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")
    # Preprocess the image and make predictions
    #prediction = model.predict(preprocess_image(image))  # Define preprocess_image function
    prediction, confidence = predict(model, uploaded_file)
    st.write(f"\nPredicted class: {prediction}.\n Confidence: {confidence: .2f}%")  # Adjust based on your output
