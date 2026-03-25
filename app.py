from flask import Flask, render_template, request, redirect, url_for
from model import predict_student, train_model, get_model_stats
import os
import json

app = Flask(__name__)

# Train model on first run if it doesn't exist
if not os.path.exists('model.pkl'):
    print('Training model for first time...')
    train_model()

@app.route('/')
def home():
    """Home page with prediction form"""
    stats = get_model_stats()
    return render_template('index.html', stats=stats)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        data = {
            'study_hours': float(request.form['study_hours']),
            'attendance_percentage': float(request.form['attendance_percentage']),
            'math_score': float(request.form['math_score']),
            'science_score': float(request.form['science_score']),
            'english_score': float(request.form['english_score']),
            'extra_activities': request.form['extra_activities'],
            'internet_access': request.form['internet_access'],
            'travel_time': request.form['travel_time'],
            'parent_education': request.form['parent_education'],
            'study_method': request.form['study_method'],
            'gender': request.form['gender'],
            'school_type': request.form['school_type'],
        }

        result = predict_student(data)

        return render_template('result.html',
            result=result,
            data=data
        )
    except Exception as e:
        return render_template('index.html', error=str(e), stats=get_model_stats())

@app.route('/add-data', methods=['GET', 'POST'])
def add_data():
    """Add new student data to the dataset"""
    if request.method == 'POST':
        try:
            import pandas as pd
            new_record = {
                'student_id': 99999,
                'age': int(request.form.get('age', 18)),
                'gender': request.form['gender'],
                'school_type': request.form['school_type'],
                'parent_education': request.form['parent_education'],
                'study_hours': float(request.form['study_hours']),
                'attendance_percentage': float(request.form['attendance_percentage']),
                'internet_access': request.form['internet_access'],
                'travel_time': request.form['travel_time'],
                'extra_activities': request.form['extra_activities'],
                'study_method': request.form['study_method'],
                'math_score': float(request.form['math_score']),
                'science_score': float(request.form['science_score']),
                'english_score': float(request.form['english_score']),
                'overall_score': (float(request.form['math_score']) + float(request.form['science_score']) + float(request.form['english_score'])) / 3,
                'final_grade': request.form['final_grade']
            }

            df = pd.read_csv('Student_Performance.csv')
            new_df = pd.DataFrame([new_record])
            df = pd.concat([df, new_df], ignore_index=True)
            df.to_csv('Student_Performance.csv', index=False)

            return render_template('add_data.html', success=True, total=len(df))
        except Exception as e:
            return render_template('add_data.html', error=str(e))

    return render_template('add_data.html')

@app.route('/retrain', methods=['POST'])
def retrain():
    """Retrain the model with updated data"""
    try:
        accuracy = train_model()
        return render_template('retrain.html', success=True, accuracy=accuracy)
    except Exception as e:
        return render_template('retrain.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)