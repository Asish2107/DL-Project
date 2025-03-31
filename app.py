from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import pandas as pd
from predict_pipeline import PredictPipeline
from src.components.health_report_rules import generate_health_summary

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

pipeline = PredictPipeline()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_health_data(form_data):
    health_data = {
        'BMI': float(form_data['BMI']),
        'Blood_Pressure_mmHg': form_data['Blood_Pressure'],
        'Heart_Rate_bpm': int(form_data['Heart_Rate']),
        'Blood_Sugar_mg_dL': int(form_data['Blood_Sugar']),
        'Cholesterol_HDL': int(form_data['Cholesterol_HDL']),
        'Cholesterol_LDL': int(form_data['Cholesterol_LDL']),
        'SpO2_Percentage': int(form_data['SpO2']),
        'Sleep_Hours': float(form_data['Sleep_Hours']),
        'Smoking_Status': form_data['Smoking_Status'],
        'Alcohol_Consumption': form_data['Alcohol_Consumption']
    }
    return pd.Series(health_data)

@app.route('/', methods=['GET', 'POST'])
def home():
    health_report = None
    skin_result = None
    confidence = None
    img_path = None

    if request.method == 'POST':
        # Process health parameters
        try:
            health_series = process_health_data(request.form)
            health_report = generate_health_summary(health_series)
        except Exception as e:
            return render_template('home.html', error=f"Error processing health data: {str(e)}")

        # Process skin image if uploaded
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '' and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(save_path)
                
                class_name, conf = pipeline.predict(save_path)
                skin_result = class_name
                confidence = round(conf * 100, 2)
                img_path = save_path

    return render_template('home.html',
                         health_report=health_report,
                         skin_result=skin_result,
                         confidence=confidence,
                         img_path=img_path)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)