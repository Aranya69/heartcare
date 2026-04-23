"""
Database models for Heart Disease Detection System.
Uses Flask-SQLAlchemy with SQLite.
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class User(db.Model):
    """Doctor/Staff authentication model."""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    full_name = db.Column(db.String(150), nullable=False)
    role = db.Column(db.String(50), default='doctor')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    patients = db.relationship('Patient', backref='doctor', lazy=True)

    def __repr__(self):
        return f'<User {self.username}>'


class Patient(db.Model):
    """Patient record model."""
    __tablename__ = 'patients'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    sex = db.Column(db.String(10), nullable=False)
    phone = db.Column(db.String(20), nullable=True)
    email = db.Column(db.String(120), unique=True, nullable=True)
    password_hash = db.Column(db.String(256), nullable=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    predictions = db.relationship('Prediction', backref='patient', lazy=True,
                                  order_by='Prediction.created_at.desc()')

    def __repr__(self):
        return f'<Patient {self.name}>'


class Prediction(db.Model):
    """Heart disease prediction record."""
    __tablename__ = 'predictions'

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)

    # 13 input features
    age = db.Column(db.Integer, nullable=False)
    sex = db.Column(db.String(10), nullable=False)
    chest_pain_type = db.Column(db.String(10), nullable=False)
    resting_bp = db.Column(db.Integer, nullable=False)
    cholesterol = db.Column(db.Integer, nullable=False)
    fasting_bs = db.Column(db.Integer, nullable=False)
    resting_ecg = db.Column(db.String(20), nullable=False)
    max_hr = db.Column(db.Integer, nullable=False)
    exercise_angina = db.Column(db.String(5), nullable=False)
    oldpeak = db.Column(db.Float, nullable=False)
    st_slope = db.Column(db.String(10), nullable=False)

    # Prediction results
    result = db.Column(db.Integer, nullable=False)  # 0 = Healthy, 1 = Heart Disease
    probability = db.Column(db.Float, nullable=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Prediction {self.id} - Patient {self.patient_id} - Result {self.result}>'
