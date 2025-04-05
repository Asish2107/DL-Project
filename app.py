# Import required libraries and modules
# Flask: Web framework for creating the application
# render_template: Renders HTML templates, request: Handles HTTP requests
# redirect, url_for: URL redirection and generation
# secure_filename: Sanitizes filenames for safe storage and avoids situations where a malicious user could upload a file named ../../../etc/passwd to overwrite system files
# os: Operating system interface for file/directory operations
# pandas: Data manipulation library (used for health data processing)
# Custom modules: Prediction pipeline, health report generator, RAG system initializer
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import pandas as pd
from predict_pipeline import PredictPipeline
from src.components.health_report_rules import generate_health_summary
from rag_system import initialize_rag_system

# Initialize Flask application instance
app = Flask(__name__)

# Configure application settings:
# UPLOAD_FOLDER: Directory to store uploaded skin analysis images
# ALLOWED_EXTENSIONS: Permitted image file extensions for uploads
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Initialize application components:
# 1. PredictPipeline: Custom class for skin condition prediction
# 2. QA System: RAG (Retrieval-Augmented Generation) system for document queries
pipeline = PredictPipeline()
qa_system = initialize_rag_system()

# File validation function - checks if uploaded file has allowed extension
def allowed_file(filename):
    # Split filename into name and extension, check against allowed types
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Health data processing function - converts form data to standardized format
def process_health_data(form_data):
    # Create dictionary with validated and typed health parameters:
    # Converts form strings to appropriate numeric types
    # Handles both continuous (BMI) and categorical (Smoking_Status) variables
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
    # Convert to pandas Series for compatibility with health report generator
    return pd.Series(health_data)

# Home route - serves main application interface
@app.route('/')
def home():
    # Renders base template that contains all application features
    return render_template('home.html')

# Health analysis endpoint - handles POST requests with health parameters
@app.route('/analyze_health', methods=['POST'])
def analyze_health():
    try:
        # Process form data into standardized health data format
        health_series = process_health_data(request.form)
        # Generate health report using custom rules engine
        health_report = generate_health_summary(health_series)
        # Return results to same template with health report section populated
        return render_template('home.html', health_report=health_report)
    except Exception as e:
        # Handle errors gracefully with user-friendly message
        return render_template('home.html', error=f"Health analysis error: {str(e)}")

# Skin analysis endpoint - handles image uploads for skin condition analysis
@app.route('/analyze_skin', methods=['POST'])
def analyze_skin():
    # Check if file part exists in request
    if 'file' not in request.files:
        return render_template('home.html', error="No file uploaded")
    
    # Get file object from request
    file = request.files['file']
    # Check for empty filename (no file selected)
    if file.filename == '':
        return render_template('home.html', error="No file selected")

    # Validate file type and process
    if file and allowed_file(file.filename):
        # Sanitize filename to prevent path traversal attacks
        filename = secure_filename(file.filename)
        # Create full save path within upload directory
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # Persist file to storage
        file.save(save_path)

        try:
            # Use prediction pipeline to analyze image
            class_name, conf = pipeline.predict(save_path)
            # Format confidence percentage for display
            confidence = round(conf * 100, 2)
            # Return results with image path for display
            return render_template('home.html', 
                                 skin_result=class_name, 
                                 confidence=confidence, 
                                 img_path=filename)
        except Exception as e:
            # Handle prediction errors
            return render_template('home.html', error=f"Skin analysis error: {str(e)}")

    # Fallback error for invalid file types
    return render_template('home.html', error="Invalid file type")

# Document Q&A endpoint - handles natural language queries against documents
@app.route('/ask_docs', methods=['POST'])
def ask_docs():
    try:
        # Extract question from form data
        question = request.form['question']
        # Query RAG system with user's question
        result = qa_system({"query": question})
        # Return answer with unique source documents
        return render_template('home.html',
                             rag_answer=result["result"],
                             rag_sources=list({doc.metadata["source"] 
                                           for doc in result["source_documents"]}))
    except Exception as e:
        # Handle RAG system errors
        return render_template('home.html', 
                             rag_error=f"Document Q&A error: {str(e)}")

# Application entry point - runs the Flask development server
if __name__ == '__main__':
    # Ensure upload directory exists before starting
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    # Start server with specific configuration:
    # - Accessible from any network interface (0.0.0.0)
    # - Port 5001 to avoid conflicts with other services
    # - Debug mode enabled for development (auto-reload on changes)
    app.run(host='0.0.0.0', port=5001, debug=True)