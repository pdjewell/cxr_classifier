import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import PIL
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers



def process_image(image):
    size=(256,256)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    try:
        image_array = image_array.reshape(256,256,1)
    except:
        pass
    image_array = np.mean(image_array, axis=-1, keepdims=True)
    normalised_image_array = (image_array.astype(np.float32) / 255)
    data = np.ndarray(shape=(1, 256, 256, 1), dtype=np.float32)
    data[0] = normalised_image_array
    return data 


def make_prediction(input, model_weights):
    model = models.load_model(model_weights)
    pred = model.predict(input)
    final_pred = pred.round()
    return final_pred, pred



if __name__ == "__main__":

    st.title('Chest Radiograph Analyser')
    st.header("Please upload an image of a chest radiograph to be analysed to either:")
    st.header("NORMAL or PNEUMONIA")
    st.text("")
    st.text("Created by Paul Jewell")

    uploaded_file = st.file_uploader("Uploaded a chest radiograph image here: ", type=["jpg","jpeg","png"])

    if uploaded_file is not None: 
        image = Image.open(uploaded_file)

        st.write("")
        if st.button("**ANALYSE**"):
            st.write("Classifying...")

            processed_image = process_image(image)
            label, prob = make_prediction(processed_image, 'models/model_1.h5')

            if label == 1:
                st.header(f"The chest radiograph shows signs of PNEUMONIA")
                st.text(f"Confidence level: {prob[0][0]}")
            elif label == 0: 
                st.header(f"The chest radiograph is NORMAL")
                st.text(f"Confidence level: {1 - prob[0][0]}")
            else:
                st.header("There was an error somewhere")

        st.write("")
        st.image(image, caption='Uploaded image', use_column_width=True)


st.text("Disclaimer: Created as a learning exercise and not intended for medical use")
