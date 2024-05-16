from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the model
model = load_model(r'C:\Users\CH\Desktop\practice codes\koach\asl_alphabet_9575.h5')

# Ensure the uploads directory exists
os.makedirs('uploads', exist_ok=True)

# Function to prepare the image for prediction
def prepare_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

# Route for the upload form
@app.route('/')
def index():
    return render_template('upload.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        # Prepare the image for prediction
        img_array = prepare_image(file_path, target_size=(200, 200))

        # Make the prediction
        predictions = model.predict(img_array)
        os.remove(file_path)  # Remove the uploaded file after prediction

        # Assuming you have 29 classes
        class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
        'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del',
        'nothing', 'space']
        predicted_class = class_labels[np.argmax(predictions)]

        return render_template('upload.html', prediction=predicted_class)

    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
