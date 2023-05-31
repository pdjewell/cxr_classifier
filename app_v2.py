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

    if len(image_array.shape) == 2:
        image_array = image_array.reshape(256,256,1)
    elif image_array.shape[2] == 1: 
        image_array = np.repeat(image_array, 3, axis=-1)   

    # standard normalisation 
    mean = np.mean(image_array)
    std = np.std(image_array)
    normalised_image_array = (image_array - mean) / std
    #normalised_image_array = (image_array.astype(np.float32) / 255)

    data = np.ndarray(shape=(1, 256, 256, 3), dtype=np.float32)
    data[0] = normalised_image_array

    return data 


def make_prediction(input, model_weights):
    model = models.load_model(model_weights)
    pred = model.predict(input)
    final_pred = pred.round()
    return final_pred, pred



if __name__ == "__main__":

    st.title('🫁  Chest Radiograph Analyser')
    st.text("Created by Paul Jewell")
    st.text("")
    st.header("Upload a chest radiograph image to be analysed to either: NO PNEUMONIA or PNEUMONIA")
    st.text("")

    uploaded_file = st.file_uploader("Upload a chest radiograph image here 👇 ", type=["jpg","jpeg","png"])

    if uploaded_file is not None: 
        image = Image.open(uploaded_file)

        st.write("")
        if st.button("**ANALYSE**"):
            st.write("Classifying...")

            processed_image = process_image(image)
            label, prob = make_prediction(processed_image, 'models/model_2.h5')

            if label == 1:
                st.header(f"The chest radiograph shows signs of PNEUMONIA")
                st.text(f"Confidence level: {prob[0][0]}")
            elif label == 0: 
                st.header(f"The chest radiograph does NOT show signs of Pneumonia")
                st.text(f"Confidence level: {1 - prob[0][0]}")
            else:
                st.header("There was an error somewhere")

        st.write("")
        st.image(image, caption='Uploaded image', use_column_width=True)

st.text("")
st.write("This application uses a deep convolutional neural network (CNN) trained from scratch, using the data from this publicly available [dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?datasetId=17810)")
st.text("")
st.text("Disclaimer: Created as a learning exercise and not intended for medical use")
