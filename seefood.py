import streamlit as st
import numpy as np
from tensorflow.keras.preprocess import image
from tensorflowkers import models
from tensforflow.keras.models import load_model, Model


food_list = ['apple_pie','pizza','omelete']
def predict_class(model,image):
    img = image.load_img(img, target_size=(299, 299))
    img = image.img_to_array(img)                    
    img = np.expand_dims(img, axis=0)         
    img /= 255.
    pred = model.predict(img)
    index = np.argmax(pred)
    food_list.sort()
    pred_value = food_list[index]
    return pred_value
model = load_model('best_model_3class.hdf5',compile=False)
  # prediction on model
pred = predict_class(model=model,image=img)

upload= st.file_uploader('Insert image for classification', type=['png','jpg'])
c1, c2= st.columns(2)
if upload is not None:
  img = image.load_img(upload, target_size=(299, 299))
  c1.header('Input Image')
  c1.image(img)

c2.header('Output')
c2.subheader('Predicted class :')
c2.write(pred)

