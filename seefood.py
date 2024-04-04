import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.utils import img_to_array


food_list = ['apple_pie','pizza','omelete']
def predict_class(model,images):
    img = image.img_to_array(images)                    
    img = np.expand_dims(img, axis=0)         
    img /= 255.
    pred = model.predict(img)
    index = np.argmax(pred)
    food_list.sort()
    pred_value = food_list[index]
    return pred_value
model = load_model('model_trained_3class.hdf5',compile=False)

upload= st.file_uploader('Insert image for classification', type=['png','jpg'])
c1, c2= st.columns(2)
if upload is not None:
  img = image.load_img(upload, target_size=(299, 299))
  c1.header('Input Image')
  c1.image(img)
  # prediction on model
  pred = predict_class(model=model,images=img)
  c2.header('Output')
  c2.subheader('Predicted class :')
  c2.subheader(pred)

