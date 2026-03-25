import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
import os
import json
import datetime

# Global model cache for performance
_MODEL = None
_SCALER = None
_ENCODERS = None

def train_model():
    """Train the student performance prediction model"""
    df = pd.read_csv('Student_Performance.csv')
    
    # Support both upper and lowercase grades
    grade_map = {
        'a': 1, 'b': 1, 'c': 1, 'd': 0, 'e': 0, 'f': 0,
        'A': 1, 'B': 1, 'C': 1, 'D': 0, 'E': 0, 'F': 0
    }
    df['pass_fail'] = df['final_grade'].str.strip().map(grade_map)
    df = df.dropna(subset=['pass_fail'])
    
    categorical_cols = ['gender', 'school_type', 'parent_education', 
                        'internet_access', 'travel_time', 
                        'extra_activities', 'study_method']
    
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    features = ['study_hours', 'attendance_percentage',
                'math_score', 'science_score', 'english_score', 'overall_score',
                'gender_enc', 'school_type_enc', 'parent_education_enc',
                'internet_access_enc', 'travel_time_enc',
                'extra_activities_enc', 'study_method_enc']
    
    X = df[features]
    y = df['pass_fail']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    stats = {
        'accuracy': round(accuracy * 100, 2),
        'total_students': int(len(df)),
        'pass_rate': round(float(y.mean()) * 100, 1),
        'features': features
    }
    
    # Save model files
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    with open('stats.json', 'w') as f:
        json.dump(stats, f)
    
    # Clear cache to force reload
    global _MODEL, _SCALER, _ENCODERS
    _MODEL = None
    _SCALER = None
    _ENCODERS = None
    
    print(f"Model trained on {len(df)} students")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Pass rate: {y.mean()*100:.1f}%")
    
    return round(accuracy * 100, 2)

def get_model_stats():
    """Get model statistics"""
    if os.path.exists('stats.json'):
        with open('stats.json', 'r') as f:
            return json.load(f)
    return {'accuracy': 0, 'total_students': 0, 'pass_rate': 0}

def load_models():
    """Load models once and cache them for performance"""
    global _MODEL, _SCALER, _ENCODERS
    
    if _MODEL is None:
        try:
            with open('model.pkl', 'rb') as f:
                _MODEL = pickle.load(f)
            with open('scaler.pkl', 'rb') as f:
                _SCALER = pickle.load(f)
            with open('encoders.pkl', 'rb') as f:
                _ENCODERS = pickle.load(f)
        except FileNotFoundError:
            raise Exception("Model not trained! Run train_model() first.")
        except Exception as e:
            raise Exception(f"Error loading model files: {str(e)}")
    
    return _MODEL, _SCALER, _ENCODERS

def predict_student(data):
    """Predict student performance"""
    # Load models (uses cache if already loaded)
    model, scaler, encoders = load_models()
    
    # Validate inputs
    if not (0 <= data['study_hours'] <= 24):
        raise ValueError("Study hours must be between 0 and 24")
    if not (0 <= data['attendance_percentage'] <= 100):
        raise ValueError("Attendance must be between 0 and 100")
    for score_key in ['math_score', 'science_score', 'english_score']:
        if not (0 <= data[score_key] <= 100):
            raise ValueError(f"{score_key} must be between 0 and 100")

    def safe_encode(encoder, value):
        try:
            return encoder.transform([str(value)])[0]
        except ValueError as e:
            print(f"Warning: Could not encode '{value}'. Using default (0). Error: {e}")
            return 0

    overall_score = (data['math_score'] + data['science_score'] + data['english_score']) / 3

    input_data = np.array([[
        data['study_hours'],
        data['attendance_percentage'],
        data['math_score'],
        data['science_score'],
        data['english_score'],
        overall_score,
        safe_encode(encoders['gender'], data['gender']),
        safe_encode(encoders['school_type'], data['school_type']),
        safe_encode(encoders['parent_education'], data['parent_education']),
        safe_encode(encoders['internet_access'], data['internet_access']),
        safe_encode(encoders['travel_time'], data['travel_time']),
        safe_encode(encoders['extra_activities'], data['extra_activities']),
        safe_encode(encoders['study_method'], data['study_method']),
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    # Get class order from the model to ensure correct probability mapping
    classes = model.classes_
    pass_idx = np.where(classes == 1)[0][0]
    fail_idx = np.where(classes == 0)[0][0]

    return {
        'prediction': 'Pass' if prediction == 1 else 'Fail',
        'pass_probability': round(probability[pass_idx] * 100, 2),
        'fail_probability': round(probability[fail_idx] * 100, 2),
        'overall_score': round(overall_score, 2)
    }

if __name__ == '__main__':
    train_model()