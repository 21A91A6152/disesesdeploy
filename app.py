import os
from flask import Flask, jsonify, request
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
from flask_cors import CORS

# Load disease and supplement information
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Load the TensorFlow model
model = tf.keras.models.load_model('Plant Disease Detection.h5', custom_objects={'KerasLayer': hub.KerasLayer})

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = np.expand_dims(np.array(image) / 255.0, axis=0)  # Normalize the image
    output = model.predict(input_data)
    index = np.argmax(output)
    return index


app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def get_data():
    data = {
        "message": "API is Running"
    }
    return jsonify(data)

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        try:
            image = request.files['image']
            filename = image.filename
            file_path = os.path.join('static/uploads', filename)
            image.save(file_path)

            pred = prediction(file_path)
            print(pred)
            if pred >= len(disease_info) or pred >= len(supplement_info):
                raise ValueError("Invalid prediction value")

            title = disease_info['disease_name'][pred]
            description = disease_info['description'][pred]
            prevent = disease_info['Possible Steps'][pred]
            image_url = disease_info['image_url'][pred]
            supplement_name = supplement_info['supplement name'][pred]
            supplement_image_url = supplement_info['supplement image'][pred]
            supplement_buy_link = supplement_info['buy link'][pred]

            
            if isinstance(pred, np.int64):
                pred = int(pred)
                
            return jsonify({
                'title': title,
                'desc': description,
                'prevent': prevent,
                'image_url': image_url,
                'pred': pred,
                'sname': supplement_name,
                'simage': supplement_image_url,
                'buy_link': supplement_buy_link
            })
        except Exception as e:
            print("error")
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
