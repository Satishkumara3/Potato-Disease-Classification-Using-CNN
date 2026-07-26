from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

app = Flask(__name__)

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
IMAGE_SIZE = 255

# Function to preprocess and predict
def predict(filepath):
    # Load and preprocess image like tf.keras.preprocessing.image
    img = Image.open(filepath).convert('RGB')
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict using TFLite
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = class_names[int(np.argmax(predictions[0]))]
    confidence = float(round(100 * (np.max(predictions[0])), 2))
    return predicted_class, confidence

# Route to the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', message='No selected file')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            os.makedirs('static', exist_ok=True)
            filepath = os.path.join('static', filename)
            file.save(filepath)

            # Predict
            predicted_class, confidence = predict(filepath)

            return render_template('index.html', image_path=filepath, actual_label=predicted_class, predicted_label=predicted_class, confidence=confidence)

    return render_template('index.html', message='Upload an image')

# Function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

if __name__ == '__main__':
    app.run(debug=True)
