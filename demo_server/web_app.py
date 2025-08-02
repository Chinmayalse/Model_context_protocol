"""
MCP Web Interface - A Flask-based web application for the MCP (Model Context Protocol) system.
This application provides a user-friendly interface for:
1. Uploading and extracting text from medical reports
2. Classifying medical reports
3. Viewing and managing extraction/classification results
"""

import os
import json
import asyncio
import tempfile
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_from_directory, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from users import User
from werkzeug.utils import secure_filename

# Import MCP functionality
from mcp_chat import process_medical_report as mcp_process_report, summarize_medical_report as mcp_summarize_report, enhance_text_with_gemini
import google.generativeai as genai
import tempfile
import re

# Configure Google API
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyDDBBTclRJjECny3q01Y57TIG9C6ZfVuTY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini model for chat
CHAT_MODEL = genai.GenerativeModel('gemini-2.0-flash')

# Create a custom context class for the web app
class WebAppContext:
    def __init__(self):
        self.logs = []
        self.progress = 0
        self.total_steps = 0
    
    async def report_progress(self, progress, total):
        self.progress = progress
        self.total_steps = total
        return True
    
    def info(self, message):
        self.logs.append({"level": "info", "message": message})
        print(f"INFO: {message}")
    
    def warning(self, message):
        self.logs.append({"level": "warning", "message": message})
        print(f"WARNING: {message}")
    
    def error(self, message):
        self.logs.append({"level": "error", "message": message})
        print(f"ERROR: {message}")

# Create wrapper function for MCP tool

async def process_medical_report_wrapper(file_path, extraction_method="auto"):
    try:
        ctx = WebAppContext()
        result = await mcp_process_report(file_path, ctx, extraction_method)
        
        # Ensure verified_text is prioritized
        if 'extracted_text' in result and 'verified_text' not in result:
            result['verified_text'] = result['extracted_text']
            
        return result
    except Exception as e:
        ctx.error(f"Error in process_medical_report: {str(e)}")
        return {"error": str(e), "extracted_text": "", "report_type": "Unknown", "confidence": 0.0, "keywords_identified": [], "reason": "Error during processing"}

async def summarize_medical_report_wrapper(file_path):
    try:
        ctx = WebAppContext()
        result = await mcp_summarize_report(file_path, ctx)
        return result
    except Exception as e:
        ctx.error(f"Error in summarize_medical_report: {str(e)}")
        return {
            "error": str(e), 
            "extracted_text": "", 
            "Age": "UNKNOWN",
            "Date": "UNKNOWN",
            "Extract Text": "Error during processing",
            "Gender": "UNKNOWN",
            "Patient Name": "UNKNOWN",
            "Summary": f"Error during processing: {str(e)}"
        }

# Initialize Flask application
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For flash messages and session

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

# Create necessary directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def save_result_to_json(result, filename_base):
    """Save results to a JSON file with timestamp"""
    timestamp = get_timestamp()
    json_filename = f"{filename_base}_{timestamp}.json"
    
    # Add a source marker to help identify the type later
    if isinstance(result, dict):
        result['_source'] = filename_base
    json_path = os.path.join(RESULTS_FOLDER, json_filename)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
    return json_filename

# Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.get_by_email(email)
        if user and user.verify_password(password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        
        flash('Invalid email or password')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match')
            return render_template('signup.html')
        
        user, error = User.create(username, email, password)
        if error:
            flash(error)
            return render_template('signup.html')
        
        login_user(user)
        return redirect(url_for('index'))
    
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return render_template('chat.html', user=current_user)

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        # Secure the filename
        filename = secure_filename(file.filename)
        
        # Save the file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Get the action type (process or summarize)
        action = request.form.get('action', 'process')
        
        # Redirect to appropriate route based on action
        if action == 'summarize':
            return redirect(url_for('summarize_report', filename=filename))
        else:
            return redirect(url_for('process_report', filename=filename))
    else:
        flash('File type not allowed')
        return redirect(url_for('index'))

@app.route('/process/<filename>')
@login_required
def process_report(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(file_path):
        flash('File not found')
        return redirect(url_for('index'))
    
    # Run the combined processing
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(process_medical_report_wrapper(file_path))
    loop.close()
    
    # Save result to JSON
    json_filename = save_result_to_json(result, 'process')
    
    return render_template('result.html', 
                          result=result,
                          filename=filename,
                          json_filename=json_filename)

@app.route('/summarize/<filename>')
@login_required
def summarize_report(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(file_path):
        flash('File not found')
        return redirect(url_for('index'))
    
    # Run the summarization
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(summarize_medical_report_wrapper(file_path))
    loop.close()
    
    # Save result to JSON
    json_filename = save_result_to_json(result, 'summary')
    
    return render_template('summary.html', 
                          result=result,
                          filename=filename,
                          json_filename=json_filename)

@app.route('/results')
@login_required
def view_results():
    # List all JSON files in the results directory
    results = []
    for filename in os.listdir(RESULTS_FOLDER):
        if filename.endswith('.json'):
            file_path = os.path.join(RESULTS_FOLDER, filename)
            file_stats = os.stat(file_path)
            
            # Initialize result type
            result_type = 'process'
            
            # Check if filename contains 'summarize' or starts with 'summary_'
            if 'summarize' in filename.lower() or filename.startswith('summary_'):
                result_type = 'summary'
            
            # Try to read the file to get more information from content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Check for summary-specific fields
                    if any(key in data for key in ['Extract Text', 'Summary', 'Age', 'Gender', 'Patient Name']):
                        result_type = 'summary'
                    # Check if the file was saved from summarize_report function
                    if data.get('_source') == 'summarize':
                        result_type = 'summary'
            except Exception as e:
                print(f"Error reading file {filename}: {str(e)}")
            
            results.append({
                'filename': filename,
                'type': result_type,
                'size': file_stats.st_size,
                'created': datetime.fromtimestamp(file_stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
            })
    
    # Sort by creation time (newest first)
    results.sort(key=lambda x: x['created'], reverse=True)
    
    return render_template('results.html', results=results)

@app.route('/download/<filename>')
@login_required
def download_result(filename):
    return send_from_directory(RESULTS_FOLDER, filename, as_attachment=True)

@app.route('/view/<filename>')
@login_required
def view_result(filename):
    file_path = os.path.join(RESULTS_FOLDER, filename)
    
    if not os.path.exists(file_path):
        flash('File not found')
        return redirect(url_for('results'))
    
    with open(file_path, 'r', encoding='utf-8') as f:
        result = json.load(f)
    
    return render_template('view_result.html', 
                          result=result, 
                          filename=filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
