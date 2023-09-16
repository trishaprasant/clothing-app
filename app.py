#Import necessary libraries
from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import os
from PIL import Image
from tensorflow.keras.models import Sequential
import cv2
from tensorflow.keras.models import load_model
from model_arch import create_model



UPLOAD_FOLDER = './static/images'

app = Flask(__name__)

#Set Max size of file as 10MB.
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

#def init():
    #global graph
    #graph = tf.get_default_graph()
    
   

# Function to load and prepare the image in right shape
def read_image(filename):
    image = load_img(filename, grayscale=True, target_size=(28, 28))
    image_data = np.array(image, dtype="float32")
    image_data = cv2.absdiff(image_data, 255)
    return image_data

#new
@app.route("/", methods=["GET", "POST"])
def predict():
    print('request.method: ', request.method)
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = UPLOAD_FOLDER + '/' + image_file.filename
            image_file.save(image_location)
            image = read_image(image_location)
            
            image = (np.expand_dims(image,0))
            image_array = np.array(image, dtype="float32")
            image_array = image_array / 255.0

            model = create_model()
            model.build(input_shape=(1, 28, 28, 1)) 
            model.load_weights('/Users/trishaprasant/Documents/CSE416/clothing-app/clothing-app-model-weights.h5')
            
            prediction = model.predict(image_array)
            article_num = np.argmax(prediction)
                    
            class_name_map = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}
            
            pred = class_name_map[article_num]
            return render_template("index.html", prediction=pred, image_loc=image_file.filename)
        
    return render_template("index.html", prediction=0, image_loc=None)


if __name__ == "__main__":
    #init()
    app.run()