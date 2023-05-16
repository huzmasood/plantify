from __future__ import division, print_function
import os
from flask import Flask, render_template, request
import numpy as np
import pickle
from keras.models import load_model
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

app = Flask(__name__)
crop_prediction_model = pickle.load(open('./models/crop-prediction.pkl', 'rb'))
banana_model = load_model("./models/banana.h5")
tomato_model = load_model("./models/tomato.h5")


@app.route('/')
def index():
    return render_template('dashboard.html', active_page="dashboard")


@app.route('/crop-prediction')
def crop_prediction():
    return render_template('crop-prediction.html', active_page="crop-prediction")


@app.route('/crop_predict', methods=['POST'])
def crop_predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    output = crop_prediction_model.predict(final_features)
    return render_template('crop-prediction.html', active_page="crop-prediction", prediction_text='Suggested crop for given soil health condition is: "{}".'.format(output[0]))


@app.route('/banana-panel')
def banana():
    return render_template('banana-panel.html', active_page="banana-panel")


@app.route('/banana_predict', methods=['GET', 'POST'])
def banana_predict():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads/banana', secure_filename(f.filename))
        f.save(file_path)
        preds = banana_model_predict(file_path, banana_model)
        return preds
    return None


@app.route('/tomato-panel')
def tomato():
    return render_template('tomato-panel.html', active_page="tomato-panel")


@app.route('/tomato_predict', methods=['GET', 'POST'])
def tomato_predict():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads/tomato', secure_filename(f.filename))
        f.save(file_path)
        preds = tomato_model_predict(file_path, tomato_model)
        result = preds
        return result
    return None


def banana_model_predict(img_path, model):
    img = tf.keras.utils.load_img(img_path, target_size=(100, 100))
    x = tf.keras.utils.img_to_array(img)
    x = np.true_divide(x, 255)
    y_pred = model.predict(x.reshape(1, 100, 100, 3))
    preds = y_pred
    str1 = ''
    result = np.argmax(preds, axis=1)
    if result == 0:
        str1 = 'Disease: Black Bacterial Wilt, For treatment use Fertilizers with Calcium(Ca)'
    elif result == 1:
        str1 = 'Disease: Black Sigatoka Disease, For treatment use fungicides like copper oxychloride, mancozeb, chlorothalonil or carbendazim'
    elif result == 2:
        str1 = 'Healthy Leaf'
    else:
        str1 = "It's not a banana leaf, Please upload a picture of Banana Leaf"
    return str1


def tomato_model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x/255
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds == 0:
        preds = "Bacterial Spot"
    elif preds == 1:
        preds = "Early Blight"
    elif preds == 2:
        preds = "Late Blight"
    elif preds == 3:
        preds = "Leaf Mold"
    elif preds == 4:
        preds = "Septoria Leaf Spot"
    elif preds == 5:
        preds = "Spider Mites - Two-Spotted Spider Mite"
    elif preds == 6:
        preds = "Target Spot"
    elif preds == 7:
        preds = "Tomato Yellow Leaf Curl Virus"
    elif preds == 8:
        preds = "Tomato Mosaic Virus"
    else:
        preds = "Healthy"
    return preds


if __name__ == "__main__":
    app.run(debug=True)
