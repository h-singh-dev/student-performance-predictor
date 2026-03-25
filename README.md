\# Student Performance Predictor



A Machine Learning web application built with Flask that predicts whether a student will Pass or Fail based on their academic and personal details.



\## Features



\- Predict student performance (Pass/Fail) with probability scores

\- Add new student data to improve the model

\- Retrain the model with updated data

\- Clean and responsive UI



\## Tech Stack



\- Backend: Python, Flask

\- ML Model: Random Forest Classifier (scikit-learn)

\- Frontend: HTML, CSS

\- Data: CSV dataset with 15,001 students



\## Model Performance



\- Accuracy: 99.90%

\- Pass Rate: 40.4%

\- Training Data: 15,001 students



\## Installation



1\. Clone the repository



git clone https://github.com/h-singh-dev/student-performance-predictor.git

cd student-performance-predictor



2\. Create virtual environment



python -m venv venv

venv\\Scripts\\activate



3\. Install dependencies



pip install flask scikit-learn pandas numpy



4\. Run the app



python app.py



5\. Open your browser and go to



http://localhost:5000



\## Project Structure



student-performance-predictor/

├── app.py              - Flask application

├── model.py            - ML model training and prediction

├── analyze.py          - Dataset analysis tool

├── Student\_Performance.csv  - Dataset

├── static/

│   └── style.css       - Styling

└── templates/

&#x20;   ├── index.html      - Home page

&#x20;   ├── result.html     - Prediction result page

&#x20;   ├── add\_data.html   - Add student data page

&#x20;   └── retrain.html    - Model retrain page



\## Input Features



\- Study Hours, Attendance Percentage

\- Math, Science, English Scores

\- Gender, School Type, Parent Education

\- Internet Access, Travel Time

\- Extra Activities, Study Method



\## Contributing



Pull requests are welcome. For major changes, please open an issue first.



\## License



MIT

