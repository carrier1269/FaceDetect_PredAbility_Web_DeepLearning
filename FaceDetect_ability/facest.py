import cv2
import numpy as np
import av
import mediapipe as mp
# from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.imagenet_utils import decode_predictions
# from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from keras.models import load_model
from tensorflow.keras.utils import to_categorical
import pathlib

st.markdown(
    f"""
         <style>
         .stApp {{
             background-image: url("https://wallpaperaccess.com/full/2131.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
    unsafe_allow_html=True
)

# resnet50_pre = tf.keras.applications.resnet.ResNet50(
# weights='imagenet', input_shape=(224, 224, 3))

new_title1 = '<p style="font-family:monospace; color:Coral; font-weight: bold; font-size: 40px;">당신의 연애 능력을 평가해드립니다ㅋㅋ</p>'
st.markdown(new_title1, unsafe_allow_html=True)
new_title = '<p style="font-family:monospace; color:Red; font-weight: bold; font-size: 18px;">⚠ ⚠ ⚠ ⚠ ⚠  외모에 자신있다면 촬영하거나 업로드해주세요.  ⚠ ⚠ ⚠ ⚠ ⚠</p>'
st.markdown("![Foo](https://search.pstatic.net/common?type=f&size=174x226&quality=75&direct=true&src=https%3A%2F%2Fshared-comic.pstatic.net%2Fthumb%2Fwebtoon%2F641253%2Fthumbnail%2Fthumbnail_IMAG21_01672165-03c8-44b1-ba0e-ef82c9cfcd10.jpg)")

# run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

picture = st.camera_input("") or st.file_uploader('', type=['jpg', 'png', 'jpeg'])

# picture = st.file_uploader('이미지 첨부.', type=['jpg', 'png', 'jpeg'])

# image = open ('test.jpg','wb').write(picture.getbuffer())

if picture is None:
    st.markdown(new_title, unsafe_allow_html=True)

else:
    image = Image.open(picture)
    st.image(image, use_column_width=True)
    img_resized = ImageOps.fit(image, (224, 224), Image.ANTIALIAS)
    img_resized = img_resized.convert('RGB')
    img_resized = np.asarray(img_resized)

    # image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    # # Show the image in a window
    # cv2.imshow('Webcam Image', image)
    # # Make the image a numpy array and reshape it to the models input shape.
    # image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    # # Normalize the image array
    # image = (image / 127.5) - 1
    # # Have the model predict wh
    def model1():
        model = load_model('keras_model.h5') 

        model.compile(
            optimizer = 'sgd',
            loss = 'categorical_crossentropy', 
            metrics = ['accuracy']
        )
        return model
    model = model1()

    labels = open("labels.txt", 'r', encoding="UTF-8").readlines()

    # print(labels)

    pred = model.predict(img_resized.reshape([1, 224, 224, 3]))
    # print(pred)
    # st.success(labels[np.argmax(pred)])
    # st.success(model.evaluate(img_resized))
    final_image = open ('test.jpg','wb').write(picture.getbuffer())

    final_dir = pathlib.Path("C:/workspace/face/face")
    finals = list(final_dir.glob('*.png'))
    final_result = Image.open(finals[np.argmax(pred)])
    st.image(final_result) 


    # , caption=finals[np.argmax(pred)]
 
    # Image.open(finals[0])