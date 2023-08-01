from flask import Blueprint, render_template, request, flash, redirect, url_for
from flask_login import login_user, login_required, logout_user, current_user
from werkzeug.security import check_password_hash
from .models import User

auth = Blueprint('auth', __name__)

@auth.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    if current_user.is_authenticated:
        return redirect('dashboard')
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user:
            if email == 'admin@admin.com' and check_password_hash(user.password, password):
                login_user(user, remember=True)
                return redirect(url_for('views.admin_dashboard'))
            else:
                flash('Invalid credentials!', category='error')
    return render_template("admin-login.html")

@auth.route('/user-login', methods=['GET', 'POST'])
def user_login():
    if current_user.is_authenticated:
        return redirect('dashboard')
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user:
            if email == 'admin@admin.com':
                flash('Login from admin panel!', category='error')
            else:
                if check_password_hash(user.password, password):
                    login_user(user, remember=True)
                    return redirect(url_for('views.dashboard'))
                else:
                    flash('Incorrect password, try again.', category='error')
        else:
            if email != '':
                flash('Email does not exist.', category='error')

    return render_template('user-login.html')

@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.user_login'))