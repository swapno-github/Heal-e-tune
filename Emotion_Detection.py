import streamlit as st 

st.set_page_config(
            page_title="Heal-a-tune",page_icon="🎶",
)

st.title("Heal-a-tune 🎶 ")
st.sidebar.success("Select the page above")
background_image = "1643831658110.jpeg"
st.image(background_image)
        
st.write("Heal-a-tune 🎶 ")
st.markdown(
    """
The face Heal-a-tune is an advanced software application that utilizes TensorFlow and deep learning from scratch to analyze human facial expressions and accurately identify the emotions depicted in the faces of individuals. This cutting-edge system can detect emotions such as happiness, sadness, anger, surprise, fear, and more from human pictures. Users have the option to upload images or capture them directly using their device's camera, making the process convenient and accessible.

### Key Features
Emotion Recognition: The system employs TensorFlow and deep learning from scratch to analyze facial features and recognize emotions depicted in the images with a high degree of accuracy.

Image Upload Option: Users can easily upload human pictures from their devices, enabling seamless input of images for emotion detection.

Real-time Camera Capture: The system offers the convenience of capturing live human facial expressions through the device's camera for instant emotion recognition.

User-Friendly Interface: The interface is designed to be intuitive and easy to navigate, ensuring a smooth user experience for both image uploads and camera capture.

Multiple Emotion Detection: The system can detect and classify multiple emotions present in the same image, providing a comprehensive understanding of the emotional expressions displayed.

Accuracy and Reliability: The use of TensorFlow and deep learning from scratch ensures that the system delivers dependable and precise emotion detection results.

### How It Works
Image Analysis: When a human picture is uploaded or captured, the system processes the image to identify facial landmarks and expressions.

Emotion Classification: Using sophisticated facial recognition algorithms based on TensorFlow and deep learning from scratch, the system identifies the specific emotions portrayed in the facial features, categorizing them into distinct emotional states.

Result Display: The detected emotions are displayed to the user, along with the option to view detailed insights into the analysis for each individual within the image.

Practical Applications
Emotion Research: Researchers and psychologists can utilize the system to gather data on human emotional responses for various studies and analyses.

Interactive Interfaces: It can be integrated into interactive interfaces and virtual environments to mimic real-time emotional responses in human-computer interactions.

Market Research: Companies can employ this system to gauge consumer reactions to products and advertisements through the analysis of facial expressions in images.
    
 ### Want to learn more? 
     - Check out [GitHub](https://github.com/enderXM249)  
    
""")
