import pickle
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, flash, redirect, url_for
import pathlib
import tensorflow.keras.preprocessing.image
from tensorflow import keras
from modelFunctions import CNNModel, RNNModel
import dataFunctions
import os
from tensorflow.python.keras.backend import set_session
import numpy as np
from PIL import Image
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
app = Flask(__name__)

session = tf.Session()
set_session(session)
graph = tf.get_default_graph()

# VARIABLES
NAME_1 = "vgg16_gru_False_50_1024_8"
NAME_2 = "inception_gru_False_50_512_8"
NAME_3 = "vgg16_gru_False_50_256_16"

CNN_1 = NAME_1.split("_")[0]
CNN_2 = NAME_2.split("_")[0]
CNN_3 = NAME_3.split("_")[0]

LEVEL_1_NAME = "Accesorios_Genericos_bd"
LEVEL_2_NAME = "Donasol_bd"
LEVEL_3_NAME = "Baquerizo_Joyeros_bd"

LEVEL_1_PATH = os.path.join("input", LEVEL_1_NAME)
LEVEL_2_PATH = os.path.join("input", LEVEL_2_NAME)
LEVEL_3_PATH = os.path.join("input", LEVEL_3_NAME)

LEVEL_1_CAPTIONS = os.path.join(LEVEL_1_PATH, "captions.txt")
LEVEL_2_CAPTIONS = os.path.join(LEVEL_2_PATH, "captions.txt")
LEVEL_3_CAPTIONS = os.path.join(LEVEL_3_PATH, "captions.txt")

LEVEL_1_MODEL = os.path.join("models", LEVEL_1_NAME, f"model_{NAME_1}.hdf5")
LEVEL_2_MODEL = os.path.join("models", LEVEL_2_NAME, f"model_{NAME_2}.hdf5")
LEVEL_3_MODEL = os.path.join("models", LEVEL_3_NAME, f"model_{NAME_3}.hdf5")

# Obtain useful data from datasets
data1, max_length1 = dataFunctions.getData(LEVEL_1_CAPTIONS)
data2, max_length2 = dataFunctions.getData(LEVEL_2_CAPTIONS)
data3, max_length3 = dataFunctions.getData(LEVEL_3_CAPTIONS)

# Create cnn models
cnn_model1 = CNNModel(CNN_1)
cnn_model2 = CNNModel(CNN_2)
cnn_model3 = CNNModel(CNN_3)

# Load models
model1 = RNNModel(max_length=max_length1)
model1.set_model(keras.models.load_model(LEVEL_1_MODEL))

model2 = RNNModel(max_length=max_length2)
model2.set_model(keras.models.load_model(LEVEL_2_MODEL))

model3 = RNNModel(max_length=max_length3)
model3.set_model(keras.models.load_model(LEVEL_3_MODEL))

with open(os.path.join("models", LEVEL_1_NAME, f'idxtoword_{NAME_1}.pk1'), 'rb') as f:
    idxtoword1 = pickle.load(f)

with open(os.path.join("models", LEVEL_1_NAME, f'wordtoidx_{NAME_1}.pk1'), 'rb') as f:
    wordtoidx1 = pickle.load(f)

with open(os.path.join("models", LEVEL_2_NAME, f'idxtoword_{NAME_2}.pk1'), 'rb') as f:
    idxtoword2 = pickle.load(f)

with open(os.path.join("models", LEVEL_2_NAME, f'wordtoidx_{NAME_2}.pk1'), 'rb') as f:
    wordtoidx2 = pickle.load(f)

with open(os.path.join("models", LEVEL_3_NAME, f'idxtoword_{NAME_3}.pk1'), 'rb') as f:
    idxtoword3 = pickle.load(f)

with open(os.path.join("models", LEVEL_3_NAME, f'wordtoidx_{NAME_3}.pk1'), 'rb') as f:
    wordtoidx3 = pickle.load(f)


# default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')


# To use the predict button in our web-app
@app.route('/predict', methods=['POST'])
def predict():
    global session
    global graph

    level = request.form['level']
    image = request.files['file']

    image_name = secure_filename(image.filename)
    image_path = os.path.join(pathlib.Path(
        __file__).parent.resolve(), "static", image_name)
    image.save(image_path)

    if level == '1':
        image = tensorflow.keras.preprocessing.image.load_img(
            image_path, target_size=(cnn_model1.get_height(), cnn_model1.get_width()))

        with session.as_default():
            with graph.as_default():
                image_encoded = cnn_model1.encode_image(image)
                caption = model1.generate_caption(
                    image_encoded, wordtoidx1, idxtoword1)

    elif level == '2':
        image = tensorflow.keras.preprocessing.image.load_img(
            image_path, target_size=(cnn_model2.get_height(), cnn_model2.get_width()))

        with session.as_default():
            with graph.as_default():
                image_encoded = cnn_model2.encode_image(image)
                caption = model2.generate_caption(
                    image_encoded, wordtoidx2, idxtoword2)

    elif level == '3':
        image = tensorflow.keras.preprocessing.image.load_img(
            image_path, target_size=(cnn_model3.get_height(), cnn_model3.get_width()))

        with session.as_default():
            with graph.as_default():
                image_encoded = cnn_model3.encode_image(image)
                caption = model3.generate_caption(
                    image_encoded, wordtoidx3, idxtoword3)

    data = {
        'image_path': image_name,
        'caption': caption
    }

    return data


if __name__ == "__main__":
    app.run(debug=True)
