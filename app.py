from flask import Flask, render_template, request, jsonify
import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
import os
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Load the model
model = load_model("Image_classifier.keras")

# Define class names - replace with your actual class names
class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog','frog','horse', 'ship', 'truck']  # Update this with your actual classes

def process_image(image_path):
    """Process an image and return the prediction"""
    # Read image
    img = cv.imread(image_path)
    
    # Convert from BGR to RGB
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    # Resize if your model expects specific dimensions
    # For example, if your model was trained on 224x224 images:
    img = cv.resize(img, (32, 32))  # Adjust dimensions based on your model
    
    # Make prediction
    prediction = model.predict(np.array([img])/255)
    
    # Get the class with highest probability
    index = np.argmax(prediction)
    
    # Get the confidence level
    confidence = float(prediction[0][index] * 100)
    
    return {
        "class": class_names[index],
        "confidence": confidence
    }

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Save the file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Process the image
        try:
            result = process_image(filepath)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)})

# Create a templates folder for HTML files
if not os.path.exists('templates'):
    os.makedirs('templates')

# Write the HTML template file
with open('templates/index.html', 'w') as f:
    f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Image Classifier</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .container {
            background-color: #f9f9f9;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .upload-area {
            border: 2px dashed #ccc;
            border-radius: 5px;
            padding: 20px;
            text-align: center;
            margin: 20px 0;
            cursor: pointer;
        }
        .upload-area:hover {
            border-color: #999;
        }
        .preview-container {
            text-align: center;
            margin: 20px 0;
        }
        #preview-image {
            max-width: 100%;
            max-height: 300px;
            display: none;
        }
        .result-container {
            margin-top: 20px;
            padding: 15px;
            background-color: #e9f7ef;
            border-radius: 5px;
            display: none;
        }
        .loading {
            text-align: center;
            display: none;
        }
        .btn {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        .btn:hover {
            background-color: #45a049;
        }
        .error {
            color: red;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Image Classifier</h1>
        <p>Upload an image to classify it using our AI model.</p>
        
        <div class="upload-area" id="drop-area">
            <p>Drag & drop an image here or click to select a file</p>
            <input type="file" id="file-input" accept="image/*" style="display: none;">
            <button class="btn" id="select-button">Select Image</button>
        </div>
        
        <div class="preview-container">
            <img id="preview-image" alt="Preview">
        </div>
        
        <div class="loading" id="loading">
            <p>Processing image, please wait...</p>
        </div>
        
        <div id="error-message" class="error"></div>
        
        <div class="result-container" id="result-container">
            <h2>Prediction Result</h2>
            <p id="result-class"></p>
            <p id="result-confidence"></p>
        </div>
    </div>

    <script>
        // Elements
        const dropArea = document.getElementById('drop-area');
        const fileInput = document.getElementById('file-input');
        const selectButton = document.getElementById('select-button');
        const previewImage = document.getElementById('preview-image');
        const loading = document.getElementById('loading');
        const resultContainer = document.getElementById('result-container');
        const resultClass = document.getElementById('result-class');
        const resultConfidence = document.getElementById('result-confidence');
        const errorMessage = document.getElementById('error-message');
        
        // Event listeners
        dropArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropArea.style.borderColor = '#45a049';
        });
        
        dropArea.addEventListener('dragleave', () => {
            dropArea.style.borderColor = '#ccc';
        });
        
        dropArea.addEventListener('drop', (e) => {
            e.preventDefault();
            dropArea.style.borderColor = '#ccc';
            
            if (e.dataTransfer.files.length) {
                handleFile(e.dataTransfer.files[0]);
            }
        });
        
        selectButton.addEventListener('click', () => {
            fileInput.click();
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length) {
                handleFile(e.target.files[0]);
            }
        });
        
        // Handle the file
        function handleFile(file) {
            // Reset UI
            resultContainer.style.display = 'none';
            errorMessage.textContent = '';
            
            // Check if file is an image
            if (!file.type.match('image.*')) {
                errorMessage.textContent = 'Please upload an image file.';
                return;
            }
            
            // Show preview
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
                previewImage.style.display = 'block';
            };
            reader.readAsDataURL(file);
            
            // Upload and get prediction
            uploadImage(file);
        }
        
        // Upload image and get prediction
        function uploadImage(file) {
            const formData = new FormData();
            formData.append('file', file);
            
            // Show loading
            loading.style.display = 'block';
            
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                loading.style.display = 'none';
                
                if (data.error) {
                    errorMessage.textContent = data.error;
                    return;
                }
                
                // Show results
                resultClass.textContent = `Predicted class: ${data.class}`;
                resultConfidence.textContent = `Confidence: ${data.confidence.toFixed(2)}%`;
                resultContainer.style.display = 'block';
            })
            .catch(error => {
                loading.style.display = 'none';
                errorMessage.textContent = 'An error occurred during processing.';
                console.error('Error:', error);
            });
        }
    </script>
</body>
</html>
    """)

if __name__ == '__main__':
    app.run(debug=True)