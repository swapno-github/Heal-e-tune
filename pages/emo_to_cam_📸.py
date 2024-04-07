import tensorflow as tf
from tensorflow.keras.preprocessing import image
import streamlit as st
import cv2
import numpy as np
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
    st.title("Suggest Song from Camera 📸")

    # Create the "captured_images" directory if it doesn't exist
    if not os.path.exists("captured_images"):
        os.makedirs("captured_images")

    # Function to capture and save image from camera
    def capture_image():
        cap = cv2.VideoCapture(0)  # Access the camera
        ret, frame = cap.read()  # Capture the frame

        if ret:
            image_path = "captured_images/captured_image.jpg"  # Set the path to save the image
            cv2.imwrite(image_path, frame)  # Save the captured image
            cap.release()  # Release the camera
            
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
            
    
        else:
            return None

    # Capture image and display in Streamlit
    if st.button("Capture Image"):
        image_path = capture_image()
        if image_path:
            st.image(image_path, caption='Captured Image', use_column_width=True)
            st.success(f"Image captured and saved at {image_path}")
        # else:
        #     st.error("Failed to capture image. Please try again.")

if __name__ == "__main__":
    main()

warnings.filterwarnings("default")    