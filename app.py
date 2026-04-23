"""
Flask Application - Heart Disease Detection & Patient Management System
Main application file with all routes for auth, patients, predictions, and dashboard.
"""

import os
import pickle
import numpy as np
import pandas as pd
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, Patient, Prediction


# ─── Pure NumPy Inference (replaces PyTorch) ───
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def numpy_predict(x, w1, b1, w2, b2):
    """Forward pass: Linear → ReLU → Linear → Sigmoid"""
    h = np.maximum(0, x @ w1.T + b1)      # Layer 1 + ReLU
    out = _sigmoid(h @ w2.T + b2)          # Layer 2 + Sigmoid
    return float(out.flatten()[0])


# ─── App Factory ───
app = Flask(__name__)
app.config['SECRET_KEY'] = 'heart-disease-detection-secret-key-2026'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///heartcare.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# ─── Load ML Artifacts ───
ml_weights = None   # dict with w1, b1, w2, b2 as numpy arrays
scaler = None
feature_columns = None


def load_ml_model():
    """Load the numpy weights, scaler, and feature columns."""
    global ml_weights, scaler, feature_columns

    artifacts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_artifacts')

    try:
        with open(os.path.join(artifacts_dir, 'feature_columns.pkl'), 'rb') as f:
            feature_columns = pickle.load(f)

        with open(os.path.join(artifacts_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)

        data = np.load(os.path.join(artifacts_dir, 'model_weights.npz'))
        ml_weights = {k: data[k] for k in data.files}
        print("[✓] ML model (numpy) loaded successfully")
    except FileNotFoundError:
        print("[!] ML artifacts not found. Run 'python train_model.py' first.")
    except Exception as e:
        print(f"[!] Error loading ML model: {e}")


# ─── Auth Decorators ───
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        # Verify user still exists in DB (handles DB reset)
        user = User.query.get(session['user_id'])
        if not user:
            session.clear()
            flash('Session expired. Please log in again.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def patient_login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'patient_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('patient_login'))
        patient = Patient.query.get(session['patient_id'])
        if not patient:
            session.clear()
            flash('Session expired. Please log in again.', 'warning')
            return redirect(url_for('patient_login'))
        return f(*args, **kwargs)
    return decorated_function


def get_current_user():
    """Get the currently logged-in user."""
    if 'user_id' in session:
        return User.query.get(session['user_id'])
    return None


def get_current_patient():
    """Get the currently logged-in patient."""
    if 'patient_id' in session:
        return Patient.query.get(session['patient_id'])
    return None


# ─── Auth Routes ───
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    if 'patient_id' in session:
        return redirect(url_for('patient_dashboard'))
    return render_template('home.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['full_name'] = user.full_name
            flash('Welcome back, Dr. ' + user.full_name + '!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'danger')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        full_name = request.form.get('full_name', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        # Validation
        errors = []
        if len(username) < 3:
            errors.append('Username must be at least 3 characters.')
        if '@' not in email:
            errors.append('Please enter a valid email.')
        if len(password) < 6:
            errors.append('Password must be at least 6 characters.')
        if password != confirm_password:
            errors.append('Passwords do not match.')
        if User.query.filter_by(username=username).first():
            errors.append('Username already exists.')
        if User.query.filter_by(email=email).first():
            errors.append('Email already registered.')

        if errors:
            for error in errors:
                flash(error, 'danger')
        else:
            user = User(
                username=username,
                email=email,
                full_name=full_name,
                password_hash=generate_password_hash(password)
            )
            db.session.add(user)
            db.session.commit()
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


# ─── Dashboard ───
@app.route('/dashboard')
@login_required
def dashboard():
    user = get_current_user()
    patients = Patient.query.filter_by(doctor_id=user.id).order_by(Patient.created_at.desc()).all()

    # Stats
    total_patients = len(patients)
    total_predictions = Prediction.query.join(Patient).filter(Patient.doctor_id == user.id).count()
    disease_count = Prediction.query.join(Patient).filter(
        Patient.doctor_id == user.id, Prediction.result == 1
    ).count()
    healthy_count = total_predictions - disease_count

    return render_template('dashboard.html',
                           user=user,
                           patients=patients,
                           total_patients=total_patients,
                           total_predictions=total_predictions,
                           disease_count=disease_count,
                           healthy_count=healthy_count)


# ─── Patient Routes ───
@app.route('/patient/add', methods=['POST'])
@login_required
def add_patient():
    user = get_current_user()
    name = request.form.get('name', '').strip()
    age = request.form.get('age', type=int)
    sex = request.form.get('sex', '').strip()
    phone = request.form.get('phone', '').strip()
    patient_email = request.form.get('patient_email', '').strip()
    patient_password = request.form.get('patient_password', '').strip()

    if not name or not age or not sex:
        flash('Please fill in all required fields.', 'danger')
        return redirect(url_for('dashboard'))

    # Check if patient email already exists
    if patient_email and Patient.query.filter_by(email=patient_email).first():
        flash('A patient with this email already exists.', 'danger')
        return redirect(url_for('dashboard'))

    patient = Patient(name=name, age=age, sex=sex, phone=phone, doctor_id=user.id)
    if patient_email and patient_password:
        patient.email = patient_email
        patient.password_hash = generate_password_hash(patient_password)

    db.session.add(patient)
    db.session.commit()
    flash(f'Patient "{name}" added successfully!', 'success')
    return redirect(url_for('dashboard'))


@app.route('/patient/<int:patient_id>/delete', methods=['POST'])
@login_required
def delete_patient(patient_id):
    user = get_current_user()
    patient = Patient.query.filter_by(id=patient_id, doctor_id=user.id).first_or_404()

    # Delete associated predictions first
    Prediction.query.filter_by(patient_id=patient.id).delete()
    db.session.delete(patient)
    db.session.commit()
    flash(f'Patient "{patient.name}" deleted.', 'info')
    return redirect(url_for('dashboard'))


# ─── Prediction Routes ───
@app.route('/predict', methods=['GET'])
@login_required
def predict_page():
    user = get_current_user()
    patients = Patient.query.filter_by(doctor_id=user.id).order_by(Patient.name).all()
    return render_template('predict.html', user=user, patients=patients)


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    user = get_current_user()

    if ml_weights is None:
        flash('ML model not loaded. Please contact administrator.', 'danger')
        return redirect(url_for('predict_page'))

    try:
        # Collect form data
        patient_id = request.form.get('patient_id', type=int)
        age = request.form.get('age', type=int)
        sex = request.form.get('sex')
        chest_pain_type = request.form.get('chest_pain_type')
        resting_bp = request.form.get('resting_bp', type=int)
        cholesterol = request.form.get('cholesterol', type=int)
        fasting_bs = request.form.get('fasting_bs', type=int)
        resting_ecg = request.form.get('resting_ecg')
        max_hr = request.form.get('max_hr', type=int)
        exercise_angina = request.form.get('exercise_angina')
        oldpeak = request.form.get('oldpeak', type=float)
        st_slope = request.form.get('st_slope')

        # Build a DataFrame matching the training format
        input_data = pd.DataFrame([{
            'Age': age,
            'Sex': sex,
            'ChestPainType': chest_pain_type,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'RestingECG': resting_ecg,
            'MaxHR': max_hr,
            'ExerciseAngina': exercise_angina,
            'Oldpeak': oldpeak,
            'ST_Slope': st_slope
        }])

        # One-hot encode (same as training)
        input_encoded = pd.get_dummies(input_data, drop_first=True, dtype=int)

        # Ensure all feature columns exist (add missing ones as 0)
        for col in feature_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        # Reorder to match training
        input_encoded = input_encoded[feature_columns]

        # Scale numeric features
        cols_to_scale = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
        input_encoded[cols_to_scale] = scaler.transform(input_encoded[cols_to_scale])

        # NumPy inference (no PyTorch needed)
        input_array = input_encoded.to_numpy().astype(np.float32)
        probability = numpy_predict(
            input_array,
            ml_weights['w1'], ml_weights['b1'],
            ml_weights['w2'], ml_weights['b2']
        )
        result = 1 if probability >= 0.5 else 0

        # Save prediction to database
        prediction = Prediction(
            patient_id=patient_id,
            age=age,
            sex=sex,
            chest_pain_type=chest_pain_type,
            resting_bp=resting_bp,
            cholesterol=cholesterol,
            fasting_bs=fasting_bs,
            resting_ecg=resting_ecg,
            max_hr=max_hr,
            exercise_angina=exercise_angina,
            oldpeak=oldpeak,
            st_slope=st_slope,
            result=result,
            probability=round(probability, 4)
        )
        db.session.add(prediction)
        db.session.commit()

        # Return JSON for AJAX requests
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'success': True,
                'result': result,
                'probability': round(probability * 100, 2),
                'label': 'Heart Disease Detected' if result == 1 else 'Healthy',
                'prediction_id': prediction.id
            })

        flash_msg = '⚠️ Heart Disease Detected' if result == 1 else '✅ Patient appears Healthy'
        flash(f'{flash_msg} (Confidence: {probability * 100:.1f}%)', 'danger' if result == 1 else 'success')
        return redirect(url_for('history', patient_id=patient_id))

    except Exception as e:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': False, 'error': str(e)}), 400
        flash(f'Prediction error: {str(e)}', 'danger')
        return redirect(url_for('predict_page'))


# ─── API Predict (JSON) ───
@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    """JSON API endpoint for predictions."""
    request.headers.environ['HTTP_X_REQUESTED_WITH'] = 'XMLHttpRequest'
    return predict()


# ─── History Routes ───
@app.route('/history')
@login_required
def history():
    user = get_current_user()
    patient_id = request.args.get('patient_id', type=int)

    query = Prediction.query.join(Patient).filter(Patient.doctor_id == user.id)
    if patient_id:
        query = query.filter(Prediction.patient_id == patient_id)

    predictions = query.order_by(Prediction.created_at.desc()).all()
    patients = Patient.query.filter_by(doctor_id=user.id).order_by(Patient.name).all()
    selected_patient = Patient.query.get(patient_id) if patient_id else None

    return render_template('history.html',
                           user=user,
                           predictions=predictions,
                           patients=patients,
                           selected_patient=selected_patient)


# ─── Patient Portal Routes ───
@app.route('/patient/login', methods=['GET', 'POST'])
def patient_login():
    if 'patient_id' in session:
        return redirect(url_for('patient_dashboard'))
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        patient = Patient.query.filter_by(email=email).first()
        if patient and patient.password_hash and check_password_hash(patient.password_hash, password):
            session['patient_id'] = patient.id
            session['patient_name'] = patient.name
            flash(f'Welcome back, {patient.name}!', 'success')
            return redirect(url_for('patient_dashboard'))
        else:
            flash('Invalid email or password.', 'danger')

    return render_template('patient_login.html')


@app.route('/patient/register', methods=['GET', 'POST'])
def patient_register():
    if 'patient_id' in session:
        return redirect(url_for('patient_dashboard'))

    if request.method == 'POST':
        name = request.form.get('full_name', '').strip()
        age = request.form.get('age', type=int)
        sex = request.form.get('sex', '').strip()
        phone = request.form.get('phone', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        errors = []
        if not name or len(name) < 2:
            errors.append('Full name is required.')
        if not age or age < 1:
            errors.append('Please enter a valid age.')
        if not sex:
            errors.append('Please select your sex.')
        if '@' not in email:
            errors.append('Please enter a valid email.')
        if len(password) < 6:
            errors.append('Password must be at least 6 characters.')
        if password != confirm_password:
            errors.append('Passwords do not match.')
        if Patient.query.filter_by(email=email).first():
            errors.append('Email already registered.')

        if errors:
            for error in errors:
                flash(error, 'danger')
        else:
            patient = Patient(
                name=name,
                age=age,
                sex=sex,
                phone=phone,
                email=email,
                password_hash=generate_password_hash(password)
            )
            db.session.add(patient)
            db.session.commit()
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('patient_login'))

    return render_template('patient_register.html')


@app.route('/patient/logout')
def patient_logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('patient_login'))


@app.route('/patient/dashboard')
@patient_login_required
def patient_dashboard():
    patient = get_current_patient()
    predictions = Prediction.query.filter_by(patient_id=patient.id).order_by(
        Prediction.created_at.desc()
    ).all()

    total_predictions = len(predictions)
    disease_count = sum(1 for p in predictions if p.result == 1)
    healthy_count = total_predictions - disease_count

    return render_template('patient_dashboard.html',
                           patient=patient,
                           predictions=predictions,
                           total_predictions=total_predictions,
                           disease_count=disease_count,
                           healthy_count=healthy_count)


# ─── Initialize ───
with app.app_context():
    db.create_all()
    load_ml_model()


if __name__ == '__main__':
    app.run(debug=False, port=5000)
