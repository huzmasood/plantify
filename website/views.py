from flask import Blueprint, render_template, request, flash, redirect
from flask_login import login_required, current_user
from sqlalchemy import func
import os
import numpy as np
import pickle
from werkzeug.utils import secure_filename
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from .models import User, Plant
from werkzeug.security import generate_password_hash
from . import db
import json

crop_prediction_model = pickle.load(open('./website/models/crop-prediction.pkl', 'rb'))
banana_model = load_model("./website/models/banana.h5")
tomato_model = load_model("./website/models/tomato.h5")

views = Blueprint('views', __name__)

def get_user_emails_with_plant_count():
    query = db.session.query(User.email, func.count(Plant.id).label('plant_count')).outerjoin(Plant).group_by(User.email)
    results = query.all()
    return results

# Routes

@views.route('/')
def index():
    if current_user.is_authenticated:
        return redirect('dashboard')
    return render_template('index.html')

# Admin Routes

@views.route('/admin-dashboard')
@login_required
def admin_dashboard():
    if current_user.email != 'admin@admin.com':
        return redirect('dashboard')
    num_users = User.query.count()
    user_plant_counts = get_user_emails_with_plant_count()
    return render_template('admin-dashboard.html', active_page="dashboard", num_users=num_users-1, user_plant_counts=user_plant_counts)

@views.route('/add-user', methods=['GET', 'POST'])
@login_required
def add_user():
    if current_user.email != 'admin@admin.com':
        return redirect('dashboard')
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email already exists.', category='error')
        elif len(email) < 4:
            flash('Email must be greater than 3 characters.', category='error')
        elif len(name) < 2:
            flash('Name must be greater than 1 character.', category='error')
        elif len(password) < 7:
            flash('Password must be at least 7 characters.', category='error')
        else:
            new_user = User(email=email, name=name, password=generate_password_hash(password, method='sha256'))
            db.session.add(new_user)
            db.session.commit()
            flash('Account created!', category='success')

    return render_template('add-user.html', active_page="add-user")

# User Routes

@views.route('/dashboard')
@login_required
def dashboard():
    if current_user.email == 'admin@admin.com':
        return redirect('admin-dashboard')
    return render_template('dashboard.html', active_page="dashboard", name=current_user.name)

@views.route('/add-plant-details', methods=['GET', 'POST'])
@login_required
def add_plant_details():
    if current_user.email == 'admin@admin.com':
        return redirect('admin-dashboard')
    if request.method == 'POST':
        name = request.form.get('name')
        quantity = request.form.get('quantity')

        if len(name) < 1:
            flash('Name is too short!', category='error')
        else:
            new_plant = Plant(name=name, quantity=quantity, user_id=current_user.id)
            db.session.add(new_plant)
            db.session.commit()
            flash('Plant added!', category='success')
    return render_template('add-plant-details.html', active_page="add-plant-details")

@views.route('/view-plant-details')
@login_required
def view_plant_details():
    if current_user.email == 'admin@admin.com':
        return redirect('admin-dashboard')
    return render_template('view-plant-details.html', active_page="view-plant-details", user=current_user)

@views.route('/delete-plant', methods=['POST'])
def delete_plant():
    plant = json.loads(request.data)
    plantId = plant['plantId']
    plant = Plant.query.get(plantId)
    if plant:
        if plant.user_id == current_user.id:
            db.session.delete(plant)
            db.session.commit()

@views.route('/crop-prediction', methods=['GET', 'POST'])
@login_required
def crop_prediction():
    if current_user.email == 'admin@admin.com':
        return redirect('admin-dashboard')
    if request.method == 'POST':
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        output = crop_prediction_model.predict(final_features)
        return render_template('crop-prediction.html', active_page="crop-prediction", prediction_text='Suggested crop for given soil health condition is: "{}".'.format(output[0]))
    return render_template('crop-prediction.html', active_page="crop-prediction")

@views.route('/banana-panel')
@login_required
def banana():
    if current_user.email == 'admin@admin.com':
        return redirect('admin-dashboard')
    return render_template('banana-panel.html', active_page="banana-panel")

@views.route('/banana_predict', methods=['GET', 'POST'])
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


@views.route('/tomato-panel')
@login_required
def tomato():
    if current_user.email == 'admin@admin.com':
        return redirect('admin-dashboard')
    return render_template('tomato-panel.html', active_page="tomato-panel")


@views.route('/tomato_predict', methods=['GET', 'POST'])
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
    img = load_img(img_path, target_size=(100, 100))
    x = img_to_array(img)
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