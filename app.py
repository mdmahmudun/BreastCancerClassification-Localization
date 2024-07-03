import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model

# Function to load and preprocess a single image
# Function to load and preprocess a single image
def load_and_preprocess_image(img, target_size=(512, 512)):
    img = img.convert('L')  # Convert to grayscale
    img = img.resize(target_size)  # Resize to target size
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array


# Function to make predictions on multiple images
def predict_multiple_images(model, images):
    predictions = []
    for img in images:
        img_array = load_and_preprocess_image(img)
        prediction = model.predict(img_array)
        class_id = prediction[0][0, 0]
        if class_id == 1:
            predictions.append('Cancer')
        else:
            predictions.append('Normal')
    return predictions

# Load the model at the beginning
model_path = 'model.h5'  # Update with your model path
from keras.src.legacy.saving import legacy_h5_format

model = legacy_h5_format.load_model_from_hdf5(model_path, custom_objects={'mse': 'mse'})

# Main function to run the Streamlit app
def main():
    st.title('Breast Cancer Detection App')
    st.sidebar.title('Upload Image(s)')

    # File upload functionality in sidebar
    uploaded_files = st.sidebar.file_uploader("Choose mammogram image(s)...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files is not None:
        # Display the uploaded image(s) in the main content area
        images = [Image.open(file) for file in uploaded_files]

        if len(images) == 1:
            st.image(images[0], caption='Uploaded Image', width=300)
        elif len(images) == 2:
            col1, col2 = st.columns(2)
            with col1:
                st.image(images[0], caption='Uploaded Image 1', width=300)
            with col2:
                st.image(images[1], caption='Uploaded Image 2', width=300)

        # Submit button to perform predictions
        if st.sidebar.button('Submit'):
            # Perform predictions
            predictions = predict_multiple_images(model, images)

            # Display prediction results
            for i, prediction in enumerate(predictions):
                st.write(f'Prediction for Image {i+1}:')
                st.write(f'Class: {prediction}')

if __name__ == '__main__':
    main()
