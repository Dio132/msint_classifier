import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# Load the trained model
model = tf.keras.models.load_model('mnist_model.h1')

# Function to preprocess the drawn image
from PIL import Image
import numpy as np

def preprocess_image(drawn_image):
    # Convert to grayscale
    img = drawn_image.convert('L')
    
    # Resize to 28x28
    img = img.resize((28, 28))
    
    # Invert colors (MNIST-style: white digit on black background)
    img = Image.eval(img, lambda x: 255 - x)
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Normalize (match your training preprocessing)
    # Example 1: Simple scaling
    img_array = img_array / 255.0
    
    # Example 2: Use same mean/std as training data
    # img_array = (img_array - mean) / std
    
    # Flatten if needed
    img_array = img_array.reshape(1, -1)  # (1, 784)
    
    return img_array

# Streamlit drawing app
st.title("Digit Recognition")
st.write("Draw a digit and the model will predict it!")

# Canvas for drawing
from streamlit_drawable_canvas import st_canvas
canvas = st_canvas(
    fill_color="white", 
    stroke_width=30, 
    stroke_color="black", 
    background_color="white",
    height=200, 
    width=200, 
    drawing_mode="freedraw", 
    key="canvas"
)

# When the user draws something, we will predict the digit
if canvas.image_data is not None:
    # Preprocess the image
    image = Image.fromarray(canvas.image_data.astype('uint8'))
    image_array = preprocess_image(image)
    
    # Predict the digit
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)  # Get the predicted class (digit)
    
    # Display the prediction
    st.write(f"Predicted Digit: {predicted_class}")
