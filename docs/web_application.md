# Medical Report Assistant - Web Application Documentation

## Overview

The web application component of the Medical Report Assistant provides a user-friendly interface for uploading, processing, viewing, and managing medical reports. It is built using Flask, a lightweight Python web framework, and offers a responsive design for both desktop and mobile users.

## Architecture

The web application follows a standard Model-View-Controller (MVC) architecture:

- **Model**: Handles data processing and storage (MCP backend)
- **View**: HTML templates with Bootstrap for responsive design
- **Controller**: Flask routes that handle user requests

## Key Components

### 1. Flask Application (`web_app.py`)

The main Flask application file contains:
- Route definitions
- Request handling logic
- Integration with the MCP backend
- File upload and management
- Result generation and storage

### 2. Templates

The application uses Jinja2 templates for rendering HTML:
- `base.html`: Base template with common layout, navigation, and styles
- `index.html`: Home page with file upload form
- `result.html`: Displays processed report results
- `summary.html`: Displays summarized report results
- `results.html`: Lists all processed reports
- `view_result.html`: Detailed view of a processed report

### 3. Static Assets

- CSS stylesheets for custom styling
- JavaScript for interactive elements
- Images and icons

### 4. File Storage

- `uploads/`: Directory for storing uploaded medical reports
- `results/`: Directory for storing processed results as JSON files

## Detailed Features

### File Upload

- Supports multiple file formats (PDF, PNG, JPG, JPEG)
- File size validation (max 16MB)
- Secure filename handling
- Progress indication during upload

### Processing Options

- **Process**: Extract text and classify the medical report
- **Summarize**: Generate a concise summary of the medical report

### Results Display

- Structured display of processed information
- Patient details section
- Report classification section
- Extracted text section
- Verification status

### Results Management

- List view of all processed reports
- Filtering and sorting options
- Download results as JSON
- View detailed results in browser

## Routes and Functions

### Home Page (`/`)

```python
@app.route('/')
def index():
    return render_template('index.html')
```

Renders the home page with the file upload form.

### File Upload (`/upload`)

```python
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
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
```

Handles file uploads, validates the file, saves it to the uploads directory, and redirects to the appropriate processing route.

### Process Report (`/process/<filename>`)

```python
@app.route('/process/<filename>')
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
```

Processes a medical report using the MCP backend, saves the result to a JSON file, and renders the result template.

### Summarize Report (`/summarize/<filename>`)

```python
@app.route('/summarize/<filename>')
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
```

Summarizes a medical report using the MCP backend, saves the result to a JSON file, and renders the summary template.

### View Results (`/results`)

```python
@app.route('/results')
def view_results():
    # List all JSON files in the results directory
    results = []
    for filename in os.listdir(RESULTS_FOLDER):
        if filename.endswith('.json'):
            file_path = os.path.join(RESULTS_FOLDER, filename)
            file_stats = os.stat(file_path)
            
            # Determine result type based on filename or content
            if 'summary_' in filename:
                result_type = 'summary'
            else:
                result_type = 'process'
                
            # Try to read the file to get more information
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # If it has Extract Text field, it's a summary
                    if 'Extract Text' in data:
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
```

Lists all processed reports, determines their type, and renders the results template.

### Download Result (`/download/<filename>`)

```python
@app.route('/download/<filename>')
def download_result(filename):
    return send_from_directory(RESULTS_FOLDER, filename, as_attachment=True)
```

Allows downloading a processed report as a JSON file.

### View Result (`/view/<filename>`)

```python
@app.route('/view/<filename>')
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
```

Displays a detailed view of a processed report.

## Helper Functions

### `allowed_file(filename)`

```python
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
```

Checks if a file has an allowed extension.

### `get_timestamp()`

```python
def get_timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')
```

Generates a timestamp for file naming.

### `save_result_to_json(result, filename_base)`

```python
def save_result_to_json(result, filename_base):
    """Save results to a JSON file with timestamp"""
    timestamp = get_timestamp()
    json_filename = f"{filename_base}_{timestamp}.json"
    json_path = os.path.join(RESULTS_FOLDER, json_filename)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    return json_filename
```

Saves a processed result to a JSON file with a timestamp.

## Templates

### Base Template (`base.html`)

The base template provides the common structure for all pages, including:
- HTML doctype and head section
- Bootstrap CSS and JavaScript
- Navigation bar
- Flash messages
- Footer
- Content block for page-specific content

### Index Template (`index.html`)

The home page template includes:
- File upload form
- Action selection (process or summarize)
- Instructions for users
- Links to view previous results

### Result Template (`result.html`)

The result template displays processed report information:
- Patient information section
- Report classification section
- Extracted text section
- Verification status
- Download and back buttons

### Summary Template (`summary.html`)

The summary template displays summarized report information:
- Patient information section
- Summary section
- Extracted text section (collapsible)
- Download and back buttons

### Results Template (`results.html`)

The results template lists all processed reports:
- Table of reports with type, size, and creation date
- View and download buttons for each report
- Sorting and filtering options

### View Result Template (`view_result.html`)

The view result template provides a detailed view of a processed report:
- All information from the JSON file
- Formatted display with sections
- Back button

## Styling

The web application uses Bootstrap for responsive design, with custom CSS for specific components:
- Card layouts for information sections
- Color coding for different report types
- Icons for visual cues
- Responsive adjustments for mobile devices

## JavaScript Functionality

- Form validation for file uploads
- Toggle functionality for collapsible sections
- Markdown parsing for formatted text display
- AJAX requests for asynchronous operations

## Security Considerations

- Secure filename handling to prevent path traversal attacks
- File size limits to prevent denial of service
- Content type validation
- Error handling and logging

## Performance Optimizations

- Asynchronous processing for long-running tasks
- Efficient file storage and retrieval
- Pagination for large result sets
- Caching of static assets

## Future Enhancements

- User authentication and authorization
- Multiple file upload support
- Real-time processing status updates
- Advanced filtering and search for results
- Integration with electronic health record systems
