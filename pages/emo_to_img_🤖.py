import tensorflow as tf
import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import pygame
import threading
import time
import warnings

warnings.filterwarnings("ignore")

model1 = tf.keras.models.load_model('Image_classify_v1.h5')
data_cat = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
prediction_class = ""

def play_music(prediction_class):
    # Define the list of emotions you want to check for
    valid_emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    # Check if the predicted class belongs to the list of valid emotions
    if prediction_class.lower() in valid_emotions:
        pygame.init()
        pygame.mixer.music.load(f"{prediction_class.lower()}_song.mp3")  # Adjust for individual emotion songs
        pygame.mixer.music.play()
    else:
        # Play a default song if the emotion is not in the valid list
        pygame.init()
        pygame.mixer.music.load("default_song.mp3")  # Provide path to a default song
        pygame.mixer.music.play()

# Start the music thread with the predicted class
def start_music_thread(prediction_class):
    music_thread = threading.Thread(target=play_music, args=(prediction_class,))
    music_thread.daemon = True
    music_thread.start()    
   
def main():
    st.title("Suggest Song from Emotion 😁")
    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image_path = f"uploads/{uploaded_file.name}"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Image uploaded successfully!")
        input_image_path = image_path
        img = image.load_img(input_image_path, target_size=(180, 180))
        img_array = image.img_to_array(img)
        img_array = tf.image.convert_image_dtype(img_array, tf.float32)
        img_array = np.expand_dims(img_array, axis=0)  

        predict = model1.predict(img_array)

        score = tf.nn.softmax(predict)
        prediction_class = data_cat[np.argmax(score)]
        st.image(input_image_path, width=400)
        st.write('Emotion in this image is ' + prediction_class)
        st.write('With accuracy of ' + str(np.max(score)*100))
       
        start_music_thread(prediction_class)
    
    # st.write(prediction_class) 



      
            
    

if __name__ == "__main__":
    main()
    
warnings.filterwarnings("default")    