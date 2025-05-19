import base64
import streamlit as st
from PIL import ImageOps, Image
import numpy as np

def set_background(image_file):
    """
    Sets the background of a Streamlit app using the specified image file.

    Parameters:
        image_file (str): The path to the image file.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

def classify(image, model, class_names):
    """
    Classifies an image using a trained model and returns probability scores for all classes.

    Parameters:
        image (PIL.Image.Image): The input image.
        model (tensorflow.keras.Model): The trained classification model.
        class_names (list): A list of class names.

    Returns:
        np.ndarray: Probability scores for each class.
    """
    # Convert image to grayscale if needed
    if model.input_shape[-1] == 1:
        image = image.convert('L')  # Convert to grayscale

    # Resize image to model’s expected input size
    target_size = (model.input_shape[1], model.input_shape[2])  
    image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)

    # Convert image to numpy array
    image_array = np.asarray(image).astype(np.float32)

    # Normalize image
    image_array = (image_array / 127.5) - 1

    # Reshape for grayscale models
    if model.input_shape[-1] == 1:
        image_array = np.expand_dims(image_array, axis=-1)

    # Add batch dimension
    data = np.expand_dims(image_array, axis=0)

    # Make prediction
    prediction = model.predict(data)[0]  # Extract the first row of predictions

    # Ensure output length matches class_names length
    if len(prediction) != len(class_names):
        raise ValueError(f"Model output length {len(prediction)} does not match class_names length {len(class_names)}")

    return prediction  # Return probability array
