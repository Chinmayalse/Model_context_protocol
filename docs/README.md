# Medical Report Assistant Documentation

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Web Application](#web-application)
   - [Features](#web-app-features)
   - [Routes](#routes)
   - [Templates](#templates)
   - [File Structure](#web-app-file-structure)
4. [Chat Interface](#chat-interface)
   - [Features](#chat-features)
   - [Implementation](#chat-implementation)
   - [User Interface](#chat-ui)
5. [MCP Backend](#mcp-backend)
   - [Core Functions](#core-functions)
   - [Processing Pipeline](#processing-pipeline)
6. [Installation and Setup](#installation-and-setup)
7. [Usage Guide](#usage-guide)
8. [API Reference](#api-reference)
9. [Troubleshooting](#troubleshooting)

## Overview

The Medical Report Assistant is a comprehensive application designed to process, analyze, and summarize medical reports. It leverages the Model Context Protocol (MCP) framework and Google's Gemini AI model to extract meaningful information from medical documents, classify them, and present the results in a user-friendly format.

The system consists of three main components:
1. **Web Application**: A Flask-based web interface for uploading, processing, and viewing medical reports
2. **Chat Interface**: An interactive chat interface for communicating with the AI assistant
3. **MCP Backend**: Core processing logic that handles the extraction, classification, and summarization of medical reports

## System Architecture

The Medical Report Assistant follows a modular architecture:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Web Interface  │◄───►│  MCP Backend    │◄───►│  Gemini AI      │
│  (Flask)        │     │  (Python)       │     │  (Google API)   │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        ▲                       ▲
        │                       │
        ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │
│  Chat Interface │     │  File Storage   │
│  (HTML/JS)      │     │  (Local)        │
│                 │     │                 │
└─────────────────┘     └─────────────────┘
```

- **User Interaction**: Users can interact with the system through either the web interface or the chat interface
- **Processing**: The MCP backend handles all the processing of medical reports
- **AI Integration**: Google's Gemini AI model is used for text extraction, classification, and summarization
- **Storage**: Processed reports and results are stored locally for future reference

## Web Application

The web application provides a user-friendly interface for uploading, processing, and viewing medical reports.

### Web App Features

- **File Upload**: Support for uploading medical reports in PDF and image formats
- **Processing Options**: Choose between processing (extraction and classification) or summarization
- **Results Viewing**: View processed results in a structured format
- **Results Management**: Download, view, and manage processed reports
- **Responsive Design**: Mobile-friendly interface

### Routes

| Route | Function | Description |
|-------|----------|-------------|
| `/` | `index()` | Home page with file upload form |
| `/upload` | `upload_file()` | Handles file uploads and redirects to processing |
| `/process/<filename>` | `process_report()` | Processes a medical report for extraction and classification |
| `/summarize/<filename>` | `summarize_report()` | Summarizes a medical report |
| `/results` | `view_results()` | Lists all processed reports |
| `/download/<filename>` | `download_result()` | Downloads a processed report as JSON |
| `/view/<filename>` | `view_result()` | Views a processed report in the browser |

### Templates

| Template | Description |
|----------|-------------|
| `base.html` | Base template with common layout and styles |
| `index.html` | Home page with file upload form |
| `result.html` | Displays processed report results |
| `summary.html` | Displays summarized report results |
| `results.html` | Lists all processed reports |
| `view_result.html` | Detailed view of a processed report |
| `chat.html` | Chat interface for interacting with the AI assistant |
| `chat_new.html` | New chat session template |
| `chat_results.html` | Displays chat results |

### Web App File Structure

```
demo_server/
├── web_app.py             # Main Flask application
├── uploads/               # Directory for uploaded files
├── results/               # Directory for processed results
└── templates/             # HTML templates
    ├── base.html
    ├── index.html
    ├── result.html
    ├── summary.html
    ├── results.html
    └── view_result.html
```

## Chat Interface

The chat interface provides an interactive way to communicate with the AI assistant for processing medical reports.

### Chat Features

- **Interactive Chat**: Conversational interface for interacting with the AI assistant
- **File Upload**: Upload medical reports directly in the chat
- **Report Processing**: Process and summarize medical reports through chat commands
- **Structured Display**: Well-formatted display of medical report summaries
- **Markdown Support**: Support for markdown formatting in messages
- **Responsive Design**: Mobile-friendly interface

### Chat Implementation

The chat interface is implemented using HTML, CSS, and JavaScript. It communicates with the backend server through AJAX requests.

Key components:
- **Message Handling**: Functions for displaying user and assistant messages
- **File Upload**: Drag-and-drop file upload functionality
- **Report Display**: Structured display of medical report summaries
- **UI Components**: Custom UI components for a modern chat experience

### Chat UI

The chat UI consists of several components:

- **Sidebar**: Navigation and features list
- **Chat Header**: Current chat information and actions
- **Chat Messages**: Display area for the conversation
- **Input Area**: Text input and file upload
- **Report Cards**: Structured display of medical report summaries

## MCP Backend

The MCP (Model Context Protocol) backend handles the core processing logic for medical reports.

### Core Functions

| Function | Description |
|----------|-------------|
| `process_medical_report()` | Main function for processing medical reports |
| `summarize_medical_report()` | Summarizes medical reports |
| `extract_text_from_pdf()` | Extracts text from PDF files |
| `extract_text_from_image()` | Extracts text from image files |
| `enhance_text_with_gemini()` | Enhances extracted text using Gemini AI |
| `classify_report_type()` | Classifies the type of medical report |
| `verify_extracted_text()` | Verifies and corrects extracted text |

### Processing Pipeline

1. **Text Extraction**: Extract text from the uploaded file (PDF or image)
2. **Text Enhancement**: Clean and enhance the extracted text
3. **Classification**: Determine the type of medical report
4. **Verification**: Verify and correct the extracted information
5. **Summarization**: Generate a concise summary of the report (if requested)
6. **Result Generation**: Format and return the processed results

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- Flask
- Google Generative AI Python SDK
- PyPDF2 (for PDF processing)
- Pillow (for image processing)
- Markdown (for text formatting)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/medical-report-assistant.git
   cd medical-report-assistant
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up Google API key:
   ```bash
   # On Windows
   set GOOGLE_API_KEY=your_api_key_here
   # On macOS/Linux
   export GOOGLE_API_KEY=your_api_key_here
   ```

5. Run the application:
   ```bash
   python web_app.py
   ```

6. Access the application at `http://localhost:5000`

## Usage Guide

### Processing a Medical Report

1. Navigate to the home page
2. Click "Choose File" and select a medical report (PDF or image)
3. Select "Process" as the action
4. Click "Upload"
5. View the processed results

### Summarizing a Medical Report

1. Navigate to the home page
2. Click "Choose File" and select a medical report (PDF or image)
3. Select "Summarize" as the action
4. Click "Upload"
5. View the summarized results

### Using the Chat Interface

1. Navigate to the chat page
2. Type a message or upload a file
3. Use commands like "summarize this report" or "extract information from this file"
4. View the structured results in the chat

## API Reference

### Web App API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload` | POST | Upload a file for processing |
| `/process/<filename>` | GET | Process a medical report |
| `/summarize/<filename>` | GET | Summarize a medical report |
| `/results` | GET | Get a list of all processed reports |
| `/download/<filename>` | GET | Download a processed report as JSON |
| `/view/<filename>` | GET | View a processed report in the browser |

### MCP API

| Function | Parameters | Returns |
|----------|------------|---------|
| `process_medical_report(file_path, ctx, extraction_method)` | File path, context, extraction method | Processed report data |
| `summarize_medical_report(file_path, ctx)` | File path, context | Summarized report data |

## Troubleshooting

### Common Issues

1. **File Upload Errors**
   - Ensure the file is in a supported format (PDF, PNG, JPG, JPEG)
   - Check that the file size is under 16MB

2. **Processing Errors**
   - Verify that the Google API key is correctly set
   - Check the logs for specific error messages

3. **Display Issues**
   - Clear your browser cache
   - Try a different browser

### Support

For additional support, please contact the development team or open an issue on the GitHub repository.
