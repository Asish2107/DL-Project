from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import pandas as pd
from predict_pipeline import PredictPipeline
from src.components.health_report_rules import generate_health_summary
from rag_system import initialize_rag_system

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Initialize all systems
pipeline = PredictPipeline()
qa_system = initialize_rag_system()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

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

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/analyze_health', methods=['POST'])
def analyze_health():
    try:
        health_series = process_health_data(request.form)
        health_report = generate_health_summary(health_series)
        return render_template('home.html', health_report=health_report)
    except Exception as e:
        return render_template('home.html', error=f"Health analysis error: {str(e)}")

@app.route('/analyze_skin', methods=['POST'])
def analyze_skin():
    if 'file' not in request.files:
        return render_template('home.html', error="No file uploaded")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('home.html', error="No file selected")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        try:
            class_name, conf = pipeline.predict(save_path)
            confidence = round(conf * 100, 2)
            return render_template('home.html', 
                                 skin_result=class_name, 
                                 confidence=confidence, 
                                 img_path=filename)
        except Exception as e:
            return render_template('home.html', error=f"Skin analysis error: {str(e)}")

    return render_template('home.html', error="Invalid file type")

@app.route('/ask_docs', methods=['POST'])
def ask_docs():
    try:
        question = request.form['question']
        result = qa_system({"query": question})
        return render_template('home.html',
                             rag_answer=result["result"],
                             rag_sources=list({doc.metadata["source"] 
                                           for doc in result["source_documents"]}))
    except Exception as e:
        return render_template('home.html', 
                             rag_error=f"Document Q&A error: {str(e)}")

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5001, debug=True)