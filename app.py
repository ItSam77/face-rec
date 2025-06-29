from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import base64
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Define class names based on your training data
class_names = ['Adams', 'Jeremy', 'Samuel']

def preprocess_image(image_data):
    """
    Preprocess the image to match the model's expected input format
    """
    # Decode base64 image
    image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64, prefix
    image_bytes = base64.b64decode(image_data)
    
    # Convert to PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Resize to 224x224 (model's expected input size)
    image_resized = cv2.resize(image_array, (224, 224))
    
    # Normalize pixel values to [0, 1]
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    return image_batch

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data from request
        data = request.get_json()
        image_data = data['image']
        
        # Preprocess the image
        processed_image = preprocess_image(image_data)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = class_names[predicted_class_idx]
        
        # Get all class probabilities
        all_predictions = {}
        for i, class_name in enumerate(class_names):
            all_predictions[class_name] = float(predictions[0][i])
        
        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_predictions': all_predictions
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 