# MNIST Digit Classifier 

This is a real-time digit classification web application where users can draw a number on screen, and a Neural Network classifies it instantly.

## Description

-  Draw a digit using the mouse or touchpad in the app's canvas.
-  A NN trained on the MNIST dataset predicts which digit was drawn.

## Features

- Live digit input via Streamlit's drawable canvas
- Preprocessing pipeline to convert hand-drawn images to MNIST format
- TensorFlow-based NN for digit classification
- Simple and intuitive web interface

## Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow (PIL)

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mnist-digit-classifier.git
   cd mnist-digit-classifier
    ```
2. Install dependencies:
    ``` bash
    pip install requirements.txt
    ```
3. Run the app:
    ```bash
    streamlit run app.py
        
