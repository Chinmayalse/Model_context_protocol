
"""
MCP Chat Interface using Groq with FastMCP

This module provides a chat interface powered by Groq LLM with MCP tools for:
1. Report classification
2. Text extraction from medical reports

The interface follows the FastMCP structure for tool implementation.
"""
# Import necessary libraries
import os
import sys
import json
import asyncio
import argparse
from typing import List, Dict, Any, Optional, Union
import time
from datetime import datetime
import re
import io
import base64

# PDF processing
import pypdfium2 as pdfium
from pdf2image import convert_from_path

# Image processing
import pytesseract
from PIL import Image, ImageFilter, ImageEnhance

# AI models
import google.generativeai as genai
from mcp.server.fastmcp import FastMCP, Context

# For backward compatibility
from test_identification import (
    process_pdf,
    classify_report,
    extract_text_using_gemini,
    extract_text_using_openai,
    convert_pdf_to_images,
    load_system_prompt as old_load_system_prompt
)

# Configuration
# Google API key for Gemini model
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyDDBBTclRJjECny3q01Y57TIG9C6ZfVuTY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini models
MODEL = genai.GenerativeModel('gemini-2.0-flash')

# Set up Tesseract path and remove image size limit
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
Image.MAX_IMAGE_PIXELS = 500000000

# Default model
DEFAULT_MODEL = "gemini-2.0-flash"

# Available models
AVAILABLE_MODELS = [
    "gemini-2.0-flash"
]

# Initialize FastMCP with proper configuration
mcp = FastMCP(
    name="Medical Report Processing",
    description="Tools for processing and summarizing medical reports",
    host="0.0.0.0",  # Listen on all interfaces
    port=8050      # Port to use for the server
)

# Add a startup message
print("\n[SERVER] MCP Medical Report Processing Server initialized")
print("[SERVER] Available tools: summarize_medical_report, process_medical_report")

def load_system_prompt(report_type):
    """
    Load a system prompt based on the report type.
    Args:
        report_type (str): The type of medical report.
    Returns:
        str: The loaded system prompt string.
    """
    prompt_dir = "System_prompts"
    report_type = report_type.lower()
    histopathology_surgical_types = [
        "histopathology_biopsy",
        "histopathology_lumpectomy",
        "histopathology_resection"
    ]
    histopathology_surgical_types_1 = [
        "histopathology_cytology",
        "histopathology_fnac"
    ]
    if any(report_type.startswith(s) for s in histopathology_surgical_types):
        prompt_file_path = os.path.join(prompt_dir, "histopathology_surgical_prompt.txt")
        print(f"[Prompt] Using shared surgical prompt for: {report_type}")
    elif any(report_type.startswith(s) for s in histopathology_surgical_types_1):
        prompt_file_path = os.path.join(prompt_dir, "histopathology_surgical_prompt_1.txt")
        print(f"[Prompt] Using alternate surgical prompt for: {report_type}")
    else:
        prompt_file_path = os.path.join(prompt_dir, f"{report_type}_prompt.txt")
        print(f"[Prompt] Using report-type-specific prompt: {prompt_file_path}")
    if os.path.exists(prompt_file_path):
        with open(prompt_file_path, "r") as file:
            return file.read()
    else:
        print(f"[Prompt] Prompt not found for: {report_type}, using default")
        return "Assistant is a large language model trained to extract structured medical data."


def convert_pdf_to_images_option1(pdf_path):
    """
    Convert a PDF file to images using the pdf2image library.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        list: List of image file paths.
    """
    try:
        images = convert_from_path(pdf_path)
        image_paths = []
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        for i, image in enumerate(images):
            image_path = f"{base_name}_page_{i+1}_converted.png"
            image.save(image_path, "PNG")
            image_paths.append(image_path)
        return image_paths
    except Exception as e:
        return None


def convert_pdf_to_images_option2(pdf_path, scale=300/72):
    """
    Convert a PDF file to images using the pypdfium2 library.

    Args:
        pdf_path (str): Path to the PDF file.
        scale (float): Scale factor for rendering.

    Returns:
        list: List of image file paths.
    """
    try:
        pdf = pdfium.PdfDocument(pdf_path)
        image_paths = []
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        for i in range(len(pdf)):
            page = pdf[i]
            pil_image = page.render(scale=scale).to_pil()
            image_path = f"{base_name}_page_{i+1}_pdfium.png"
            pil_image.save(image_path, "PNG")
            image_paths.append(image_path)
        return image_paths
    except Exception as e:
        return None


def convert_pdf_to_images(pdf_path):
    """
    Convert a PDF file to images using a fallback mechanism.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        list: List of image file paths.
    """
    try:
        return convert_pdf_to_images_option1(pdf_path)
    except Exception as e:
        print(f"[PDF2IMAGE] Error: {e}, falling back to pdfium2...")
        return convert_pdf_to_images_option2(pdf_path)


def preprocess_image(image_path):
    """
    Preprocess an image for better OCR results.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Path to the preprocessed image file.
    """
    image = Image.open(image_path)
    image = image.convert('L')
    image = image.filter(ImageFilter.MedianFilter())
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    processed_path = image_path.replace('.png', '_preprocessed.png')
    image.save(processed_path)
    return processed_path


def extract_text_from_image(image_path):
    """
    Extract text from an image using Tesseract OCR.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Extracted text.
    """
    return pytesseract.image_to_string(Image.open(image_path))


def extract_text_with_gemini(image_paths):
    """
    Extract text from images using the Gemini 1.5 Flash model.

    Args:
        image_paths (list): List of image file paths.

    Returns:
        str: Concatenated extracted text from all images.
    """
    texts = []
    
    # Fallback: OCR-based extraction
    print("Gemini extraction insufficient, falling back to OCR...")
    ocr_text = ""
    for image_path in image_paths:
        page_text = extract_text_from_image(image_path)
        ocr_text += page_text + "\n\n"
    
    print(f"OCR extraction complete: {len(ocr_text.strip().split())} words extracted")
    return ocr_text

def classify_report_with_gemini(corrected_text, system_prompt_content):
    """
    Classifies the report using the Gemini 1.5 Flash model.
    """
    if not system_prompt_content:
        print("Error: System prompt is empty")
        return {"report_type": "Unknown", "confidence_score": 0, "keywords_identified": []}

    try:
        # Print first 100 characters of system prompt for verification
        print(f"Using system prompt (first 100 chars): {system_prompt_content[:100]}...")
        
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        chat = model.start_chat(history=[])
        response = chat.send_message(
            f"{system_prompt_content}\n\n{corrected_text}"
        )

        classification_result = response.text
        print(f"Raw Classification Result: {classification_result}")

        # Clean up the response and ensure it's valid JSON
        cleaned_result = classification_result.strip()

        # Handle different JSON formatting cases
        if cleaned_result.startswith("```json"):
            cleaned_result = cleaned_result[7:]
            if cleaned_result.endswith("```"):
                cleaned_result = cleaned_result[:-3]
        elif cleaned_result.startswith("```"):
            cleaned_result = cleaned_result[3:]
            if cleaned_result.endswith("```"):
                cleaned_result = cleaned_result[:-3]

        # Remove any additional whitespace or newlines
        cleaned_result = cleaned_result.strip()

        # Add error handling for JSON parsing
        try:
            result_json = json.loads(cleaned_result)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Cleaned result: {cleaned_result}")
            # Try to extract JSON from the text if possible
            json_match = re.search(r'\{.*\}', cleaned_result, re.DOTALL)
            if json_match:
                try:
                    result_json = json.loads(json_match.group())
                except:
                    return {
                        "report_type": "Unknown",
                        "confidence_score": 0,
                        "keywords_identified": []
                    }
            else:
                return {
                    "report_type": "Unknown",
                    "confidence_score": 0,
                    "keywords_identified": []
                }

        # Map category or type to standardized report_type
        report_type = None

        # Check various possible keys
        if "category" in result_json:
            report_type = result_json["category"]
        elif "type" in result_json:
            report_type = result_json["type"]
        elif "report_type" in result_json:
            report_type = result_json["report_type"]

        # Standardize the report type
        if report_type:
            # Map common variations to standard names
            report_type_mapping = {
                "SERUM": "Serum_Analysis",
                "CBC": "CBC",
                "ENDOSCOPY": "Endoscopy",
                "CLINICAL BIOCHEMISTRY": "Clinical_Biochemistry",
                "BIOCHEMISTRY": "Clinical_Biochemistry",
                "IMMUNOHISTOCHEMISTRY": "IMMUNOHISTOCHEMISTRY",
                "IHC_HCG": "IHC_HCG"
            }

            report_type = report_type.upper()  # Convert to uppercase for consistent matching
            standardized_type = report_type_mapping.get(report_type, report_type)

            # Create standardized response
            standardized_response = {
                "report_type": standardized_type,
                "confidence_score": result_json.get("confidence", result_json.get("match_percentage", 0)) if isinstance(result_json.get("confidence"), (int, float)) else 0.5,
                "keywords_identified": result_json.get("matched_keywords", [])
            }
            if report_type.upper() == "HISTOPATHOLOGY":
                subcategory = result_json.get("subcategory", "Unknown")
                standardized_response["subcategory"] = subcategory
                # Modify report_type to include subcategory
                standardized_response["report_type"] = f"histopathology_{subcategory.lower()}"
            return standardized_response

        return {"report_type": "Unknown", "confidence_score": 0, "keywords_identified": []}

    except Exception as e:
        print(f"Error in classification: {str(e)}")
        print(f"Full error details: {str(e.__class__.__name__)}")
        return {"report_type": "Unknown", "confidence_score": 0, "keywords_identified": []}
    
def enhance_text_with_gemini(extracted_text, report_type, image_paths=None):
    """
    Enhance extracted text using Gemini 2.0 Flash based on report type
    """
    try:
        # Load type-specific prompt
        type_specific_prompt = load_system_prompt(report_type)
        if not type_specific_prompt:
            return None

        # Print the first 100 characters of the prompt for debugging
        print(f"Using prompt for {report_type} (first 100 chars): {type_specific_prompt[:100]}...")

        # Add explicit instruction to follow the structure exactly
        enhanced_prompt = f"{type_specific_prompt}\n\nIMPORTANT: Follow the EXACT JSON structure provided above. Do not add or remove fields. If a value is not present in the text, use 'NA' as the value. Do not include any explanatory text outside the JSON structure."

        # Create a Gemini model instance with appropriate configuration
        gemini_model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config={
                "temperature": 0.2,  # Lower temperature for more deterministic output
                "max_output_tokens": 2048,
                "top_p": 0.95
            }
        )
        
        # Create a chat with the system prompt properly set
        chat = gemini_model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": ["I need you to act as a medical report processor with these instructions:"]
                },
                {
                    "role": "model",
                    "parts": ["I'll act as a medical report processor following your instructions carefully."]
                },
                {
                    "role": "user",
                    "parts": [enhanced_prompt]
                },
                {
                    "role": "model",
                    "parts": ["I understand the instructions and will follow the exact JSON structure provided. I'll extract information from medical reports according to these guidelines."]
                }
            ]
        )
        
        # Send user message with the extracted text
        response = chat.send_message(f"Extract information from this medical report according to the specified structure:\n\n{extracted_text}")
        
        # Get the response text
        enhanced_text = response.text
        print(f"Raw enhanced text (first 200 chars): {enhanced_text[:200]}...")
        
        # Clean up the response to ensure it's valid JSON
        cleaned_text = enhanced_text.strip()
        
        # Remove markdown formatting if present
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
        
        cleaned_text = cleaned_text.strip()
        
        try:
            # Try to parse as JSON
            json_result = json.loads(cleaned_text)
            # Check if the result is a dictionary or a list
            if isinstance(json_result, dict):
                print(f"Successfully parsed enhanced text as JSON with keys: {list(json_result.keys())}")
            elif isinstance(json_result, list):
                print(f"Successfully parsed enhanced text as JSON array with {len(json_result)} items")
            return json_result
        except json.JSONDecodeError as json_error:
            print(f"JSON parsing error: {json_error}")
            # Try to extract JSON from the text if possible
            json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
            if json_match:
                try:
                    json_result = json.loads(json_match.group())
                    print(f"Extracted JSON from text with keys: {list(json_result.keys())}")
                    return json_result
                except json.JSONDecodeError:
                    print("Failed to parse extracted JSON pattern")
            
            # Return as is if JSON parsing fails
            print("Returning raw text as JSON parsing failed")
            return enhanced_text

    except Exception as e:
        print(f"Error in enhancing text: {str(e)}")
        return None

def generate_report_summary(extracted_text, image_paths=None):
    """
    Generate a structured summary of a medical report using Gemini 2.0 Flash
    
    Args:
        extracted_text: The extracted text from the medical report
        image_paths: Optional list of paths to the report images
        
    Returns:
        A dictionary containing the structured summary
    """
    try:
        # Create a prompt for summarization
        summary_prompt = """
        You are a medical report summarization expert. Extract key information from the provided medical report text and format it into a structured JSON format.
        
        Extract the following information:
        1. Patient Name (if available, otherwise use "UNKNOWN")
        2. Age (numeric value only, if available, otherwise use "UNKNOWN")
        3. Gender ("M", "F", or "UNKNOWN")
        4. Date (in DD-MM-YYYY format if available, otherwise use "UNKNOWN")
        5. Extract Text (include ALL text from the original report, formatted in Markdown for better readability - do not omit any information)
        6. Summary (provide a brief 1-2 sentence summary of the key findings)
        
        Return ONLY a valid JSON object with the following structure:
        {
            "Patient Name": "string",
            "Age": "string",
            "Gender": "string",
            "Date": "string",
            "Extract Text": "string with markdown formatting",
            "Summary": "string"
        }
        IMPORTANT MARKDOWN FORMATTING GUIDELINES FOR EXTRACT TEXT FIELD:
        - Use proper markdown syntax throughout the document
        - Use # for main title (hospital/lab name)
        - Use ## for major sections (Patient Information, Test Results, etc.)
        - Use ### for subsections (Diagnosis, Impression, etc.)
        - Use bullet points with - or * for lists of findings or test results
        - Use **bold** for important terms, values, or parameter names
        - Use tables with | for tabular data when appropriate
        - Maintain a clean, hierarchical structure with proper spacing
        - Format the text to be highly readable when rendered as markdown
        
        EXAMPLE MARKDOWN FORMAT:
        ```markdown
        # Hospital Name
        
        ## Patient Information
        - **Name**: John Doe
        - **Age**: 45
        - **Gender**: Male
        - **Date**: 01-01-2023
        
        ## Test Results
        
        ### Blood Work
        - **Hemoglobin**: 14.5 g/dL
        - **WBC**: 7,500/μL
        
        ### Impression
        Normal blood work results with no significant abnormalities.
        ```
        
        OTHER IMPORTANT NOTES:
        - Do NOT include any explanatory text outside the JSON structure
        - If information is not available, use "UNKNOWN" as the value
        - For Gender, use only "M", "F", or "UNKNOWN"
        - For Age, extract only the numeric value
        """
        
        # Create a Gemini model instance with appropriate configuration
        gemini_model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config={
                "temperature": 0.2,  # Lower temperature for more deterministic output
                "max_output_tokens": 2048,
                "top_p": 0.95
            }
        )
        
        # Start a chat session
        chat = gemini_model.start_chat(history=[])
        
        # Send system prompt first
        chat.send_message(summary_prompt)
        
        # Send user message with the extracted text
        response = chat.send_message(f"Generate a structured summary for this medical report:\n\n{extracted_text}")
        
        # Get the response text
        summary_text = response.text
        print(f"Raw summary text (first 200 chars): {summary_text[:200]}...")
        
        # Clean up the response to ensure it's valid JSON
        cleaned_text = summary_text.strip()
        
        # Remove markdown formatting if present
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
        elif cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
        
        cleaned_text = cleaned_text.strip()
        
        try:
            # Try to parse as JSON
            summary_result = json.loads(cleaned_text)
            # Check if the result is a dictionary or a list
            if isinstance(summary_result, dict):
                print(f"Successfully parsed summary as JSON with keys: {list(summary_result.keys())}")
            elif isinstance(summary_result, list):
                print(f"Successfully parsed summary as JSON array with {len(summary_result)} items")
            
            # Create a filtered result with only the specified fields
            filtered_result = {
                "Patient Name": summary_result.get("Patient Name", "UNKNOWN"),
                "Age": summary_result.get("Age", "UNKNOWN"),
                "Gender": summary_result.get("Gender", "UNKNOWN"),
                "Date": summary_result.get("Date", "UNKNOWN"),
                "Extract Text": summary_result.get("Extract Text", ""),
                "Summary": summary_result.get("Summary", "Medical report processed successfully."),
                "extracted_text": extracted_text
            }
            return filtered_result
            
        except json.JSONDecodeError as json_error:
            print(f"JSON parsing error: {json_error}")
            # Try to extract JSON from the text if possible
            json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
            if json_match:
                try:
                    json_result = json.loads(json_match.group())
                    print(f"Extracted JSON from text")
                    # Create a filtered result with only the specified fields
                    # Ensure Extract Text contains ALL the text from the original report
                    filtered_result = {
                        "Patient Name": json_result.get("Patient Name", "UNKNOWN"),
                        "Age": json_result.get("Age", "UNKNOWN"),
                        "Gender": json_result.get("Gender", "UNKNOWN"),
                        "Date": json_result.get("Date", "UNKNOWN"),
                        "Extract Text": json_result.get("Extract Text", extracted_text),  # Use original text as fallback
                        "Summary": json_result.get("Summary", "Medical report processed successfully."),
                        "extracted_text": extracted_text
                    }
                    return filtered_result
                except json.JSONDecodeError:
                    print("Failed to parse extracted JSON pattern")
            
            # Return a default structure if JSON parsing fails
            # Ensure Extract Text contains ALL the text from the original report
            print("Returning default structure as JSON parsing failed")
            return {
                "Patient Name": "UNKNOWN",
                "Age": "UNKNOWN",
                "Date": "UNKNOWN",
                "Extract Text": extracted_text,  # Include the complete original text
                "Gender": "UNKNOWN",
                "Summary": "Medical report processed successfully. Please refer to the extracted text for details.",
                "extracted_text": extracted_text
            }

    except Exception as e:
        print(f"Error in generating summary: {str(e)}")
        return None

def verify_results_with_gemini(enhanced_text, image_paths, report_type):
    """
    Verify the enhanced results using Gemini 1.5 Flash model.
    """
    try:
        # First, try to parse the enhanced_text as JSON if it isn't already
        if isinstance(enhanced_text, str):
            try:
                enhanced_json = json.loads(enhanced_text)
            except json.JSONDecodeError:
                enhanced_json = {"raw_text": enhanced_text}
        else:
            enhanced_json = enhanced_text

        system_prompt = load_system_prompt(report_type)
        print(f"Using system prompt for verification (first 100 chars): {system_prompt[:100]}...")

        verification_prompt = f"""
        {system_prompt}

        Verify and enhance the following medical report data.
        Return the response in the EXACT same JSON format:

        {json.dumps(enhanced_json, indent=2)}

        CRITICAL INSTRUCTIONS:
        1. Maintain the EXACT same JSON structure - do not add or remove any fields
        2. Update any incorrect values
        3. Add any missing values visible in the images
        4. Keep all field names identical
        5. Return ONLY the JSON data without any additional text
        6. Do NOT use markdown formatting
        7. If a value is not present in the text, use 'NA' as the value
        """

        # Prepare content parts for Gemini
        content_parts = [verification_prompt]

        # Add images to content parts
        if image_paths:
            for image_path in image_paths:
                try:
                    image = Image.open(image_path)
                    content_parts.append(image)
                    print(f"Added image {image_path} to verification")
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")
                    continue

        # Generate response using Gemini with proper system prompt handling
        print("Generating verification response with Gemini...")
        
        # Create a Gemini model instance with appropriate configuration
        verification_model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config={
                'temperature': 0.2,  # Lower temperature for more deterministic output
                'top_p': 0.95,
                'max_output_tokens': 2048,
            }
        )
        
        # Start a chat with proper system prompt
        chat = verification_model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": ["I need you to act as a medical report verification system with these instructions:"]
                },
                {
                    "role": "model",
                    "parts": ["I'll act as a medical report verification system following your instructions carefully."]
                },
                {
                    "role": "user",
                    "parts": [verification_prompt]
                },
                {
                    "role": "model",
                    "parts": ["I understand the verification instructions and will maintain the exact JSON structure while verifying and enhancing the medical report data."]
                }
            ]
        )
        
        # Create message content with images if available
        message_parts = ["Please verify this medical report data:"]
        if image_paths:
            for image_path in image_paths:
                try:
                    with open(image_path, "rb") as f:
                        image_data = f.read()
                    message_parts.append(
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": base64.b64encode(image_data).decode("utf-8")
                            }
                        }
                    )
                    print(f"Added image {image_path} to verification")
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")
        
        # Send the message with images
        response = chat.send_message(message_parts)

        if not response or not response.text:
            raise Exception("Empty response from Gemini")

        # Clean up the response text
        response_text = response.text.strip()
        print(f"Raw verification response (first 200 chars): {response_text[:200]}...")

        # Remove markdown formatting if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

        response_text = response_text.strip()

        # Try to parse the response as JSON
        try:
            verified_json = json.loads(response_text)
            # Check if the result is a dictionary or a list
            if isinstance(verified_json, dict):
                print(f"Successfully parsed verification response as JSON with keys: {list(verified_json.keys())}")
            elif isinstance(verified_json, list):
                print(f"Successfully parsed verification response as JSON array with {len(verified_json)} items")

            # Clean text values
            def clean_text_values(obj):
                if isinstance(obj, dict):
                    return {k: clean_text_values(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_text_values(item) for item in obj]
                elif isinstance(obj, str):
                    cleaned = obj.replace('\n', ' ').replace('/', ' ')
                    return ' '.join(cleaned.split())
                return obj

            # Clean and format the JSON
            cleaned_json = clean_text_values(verified_json)
            return json.dumps(cleaned_json, indent=2)

        except json.JSONDecodeError as json_error:
            print(f"Error parsing Gemini verification response as JSON: {json_error}")
            # Try to extract JSON from the text if possible
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    extracted_json = json.loads(json_match.group())
                    # Check if the result is a dictionary or a list
                    if isinstance(extracted_json, dict):
                        print(f"Extracted JSON from verification response with keys: {list(extracted_json.keys())}")
                    elif isinstance(extracted_json, list):
                        print(f"Extracted JSON from verification response as array with {len(extracted_json)} items")
                    return json.dumps(extracted_json, indent=2)
                except json.JSONDecodeError:
                    print("Failed to parse extracted JSON pattern from verification response")
            
            # If JSON parsing fails, try to maintain the original structure
            print("Falling back to original JSON structure")
            return json.dumps(enhanced_json, indent=2)

    except Exception as e:
        print(f"Error in Gemini verification: {e}")
        # Return the original enhanced text if verification fails
        if isinstance(enhanced_text, str):
            return enhanced_text
        return json.dumps(enhanced_text, indent=2)

def process_pdf_report(pdf_path):
    """
    Main function to process PDF reports with classification and parameter extraction
    """
    try:
        # Step 1: Convert PDF to images
        print("Converting PDF to images...")
        image_paths = convert_pdf_to_images(pdf_path)
        
        if not image_paths:
            raise Exception("Failed to convert PDF to images")

        # Step 2: Extract text using Gemini with OCR fallback
        print("Extracting text from images...")
        full_text = extract_text_from_pdf(image_paths)
        
        if not full_text or len(full_text.strip()) < 20:
            raise Exception("Failed to extract meaningful text from images")

        # Step 3: Classify the report
        print("Classifying report...")
        # Read classification system prompt
        try:
            with open(os.path.join("System_prompts", "classify_sys_prompt.txt"), "r") as f:
                classification_system_prompt = f.read()
        except FileNotFoundError:
            print("Classification system prompt file not found, using default prompt")
            classification_system_prompt = """
            You are a medical report classifier. Analyze the provided text and determine the type of medical report.
            Provide your classification as a JSON object with fields: 
            - report_type: The specific type of report
            - confidence: A score between 0 and 1 indicating your confidence
            - matched_keywords: List of keywords that helped identify the report type
            """
        
        classification_result = classify_report_with_gemini(full_text, classification_system_prompt)
        
        # Step 4: Extract parameters based on report type
        report_type = classification_result.get('report_type')
        if report_type != "Unknown":
            print(f"Extracting parameters for {report_type}...")
            # Load specific prompt for the report type
            type_specific_prompt = load_system_prompt(report_type)
            
            # Extract parameters using the specific prompt
            enhanced_text = enhance_text_with_groq(full_text, report_type, image_paths)
            
            print(enhanced_text)
            # Add enhanced text to the result
            classification_result['extracted_parameters'] = enhanced_text

            # Verify the results using Gemini
            verified_text = verify_results_with_gemini(enhanced_text, image_paths, report_type)
            classification_result['verified_parameters'] = verified_text

        # Return results
        return {
            "text": full_text,
            "classification": classification_result,
            "extraction_method": "gemini_with_ocr_fallback"
        }

    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return None
    finally:
        # Clean up temporary images
        if 'image_paths' in locals() and image_paths:
            for image_path in image_paths:
                if os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                    except Exception as e:
                        print(f"Error removing temporary image {image_path}: {e}")

def save_to_json(result, base_filename, report_type=None):
    """
    Save results to a JSON file with timestamp in a readable format.
    Each report is saved in its own file within a type-specific directory.
    """
    from datetime import datetime
    import json
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create base output directory if it doesn't exist
    base_output_dir = "results"
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    
    # Create report type specific directory
    if report_type:
        report_dir = os.path.join(base_output_dir, report_type.lower())
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
    else:
        report_dir = base_output_dir
    
    # Generate filename with timestamp
    json_file = f"{base_filename}_{timestamp}.json"
    
    # Clean and parse the JSON string if needed
    if isinstance(result, str):
        # Remove markdown and extra formatting
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0]
        try:
            result = json.loads(result.strip())
        except json.JSONDecodeError:
            print("Error parsing JSON string")
            result = {"error": "Invalid JSON format"}
    
    # Filter out internal fields (those starting with underscore) for summary results
    if report_type == "summary":
        # Keep only the specified fields for summary results
        filtered_result = {
            "Patient Name": result.get("Patient Name", "UNKNOWN"),
            "Age": result.get("Age", "UNKNOWN"),
            "Gender": result.get("Gender", "UNKNOWN"),
            "Date": result.get("Date", "UNKNOWN"),
            "Extract Text": result.get("Extract Text", ""),
            "Summary": result.get("Summary", "Medical report processed successfully."),
            "extracted_text": result.get("extracted_text", "")
        }
        result_to_save = filtered_result
    else:
        # For other result types, filter out fields starting with underscore
        result_to_save = {k: v for k, v in result.items() if not k.startswith('_')}
    
    # Save to JSON file with proper formatting
    json_path = os.path.join(report_dir, json_file)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result_to_save, f, indent=4, ensure_ascii=False, sort_keys=True)
    
    print(f"\nJSON results saved to {json_path}")
    return json_path

# MCP Tools Registration
# These tools will be available both in interactive mode and server mode


@mcp.tool()
async def summarize_medical_report(file_path: str, ctx: Context) -> Dict[str, Any]:
    """
    Summarize a medical report file - extract text and provide a structured summary.
    
    Args:
        file_path: Path to the PDF or image file
        ctx: MCP context for tracking progress and providing information
        
    Returns:
        Summary results including the extracted text and structured data
    """
    print(f"\n[SERVER] Tool called: summarize_medical_report with file: {file_path}")
    ctx.info(f"Summarizing medical report: {file_path}")
    await ctx.report_progress(0, 5)
    
    if not os.path.exists(file_path):
        ctx.error(f"File not found: {file_path}")
        return {"error": f"File not found: {file_path}"}
    
    start_time = time.time()
    result = {}
    
    try:
        # STEP 1: Convert PDF to images if needed
        if file_path.lower().endswith('.pdf'):
            ctx.info("Converting PDF to images...")
            await ctx.report_progress(1, 5)
            
            try:
                image_paths = convert_pdf_to_images(file_path)
                if image_paths and len(image_paths) > 0:
                    ctx.info(f"Converted PDF to {len(image_paths)} images using enhanced method")
                else:
                    # Fall back to original method
                    ctx.warning("Enhanced conversion failed, falling back to original method")
                    image_paths = convert_pdf_to_images(file_path)
            except Exception as conversion_error:
                ctx.warning(f"Enhanced conversion failed: {str(conversion_error)}, falling back to original method")
                image_paths = convert_pdf_to_images(file_path)
                
            if not image_paths:
                ctx.error("Failed to convert PDF to images")
                return {"error": "Failed to convert PDF to images"}
            
            ctx.info(f"Successfully converted PDF to {len(image_paths)} images")
        else:
            # Assume it's already an image
            image_paths = [file_path]
            await ctx.report_progress(1, 5)
        
        # STEP 2: Extract text from the images
        ctx.info("Extracting text using Gemini...")
        await ctx.report_progress(2, 5)
        
        # Use Gemini for extraction
        extracted_text = extract_text_from_pdf(image_paths)
        if not extracted_text or len(extracted_text.strip()) < 20:
            ctx.warning("Insufficient text extracted with Gemini, falling back to OCR")
            from test_identification import extract_text_from_pdf_with_ocr
            extracted_text = extract_text_from_pdf_with_ocr(image_paths)
        
        ctx.info("Text extraction complete")
        await ctx.report_progress(3, 5)
        
        # STEP 3: Generate summary with Gemini
        ctx.info("Generating summary with Gemini...")
        summary_result = generate_report_summary(extracted_text, image_paths)
        
        if not summary_result:
            ctx.warning("Failed to generate summary, returning extracted text only")
            summary_result = {
                "Age": "UNKNOWN",
                "Date": "UNKNOWN",
                "Extract Text": extracted_text,
                "Gender": "UNKNOWN",
                "Patient Name": "UNKNOWN",
                "Summary": "Failed to generate summary. Please refer to the extracted text for details."
            }
        
        # Create a filtered result with only the specified fields
        # Ensure Extract Text contains ALL the text from the original report
        filtered_result = {
            "Patient Name": summary_result.get("Patient Name", "UNKNOWN"),
            "Age": summary_result.get("Age", "UNKNOWN"),
            "Gender": summary_result.get("Gender", "UNKNOWN"),
            "Date": summary_result.get("Date", "UNKNOWN"),
            "Extract Text": summary_result.get("Extract Text", extracted_text),  # Use original text as fallback
            "Summary": summary_result.get("Summary", "Medical report processed successfully."),
            "extracted_text": extracted_text
        }
        
        # Update result with filtered fields only
        result.update(filtered_result)
        await ctx.report_progress(4, 5)
        
        # STEP 4: Finalize results
        processing_time = time.time() - start_time
        # These fields are used internally but won't be included in the final JSON
        result["_processing_time"] = f"{processing_time:.2f} seconds"
        result["_file_path"] = file_path
        result["_file_processed"] = os.path.basename(file_path)
        result["_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result["_success"] = True
        
        # Save results to JSON file
        try:
            base_filename = os.path.splitext(os.path.basename(file_path))[0]
            json_path = save_to_json(result, base_filename, "summary")
            result["json_saved_to"] = json_path
        except Exception as save_error:
            ctx.warning(f"Failed to save results to JSON: {str(save_error)}")
        
        await ctx.report_progress(5, 5)
        ctx.info(f"Medical report summarization complete in {processing_time:.2f} seconds")
        
        return result
        
    except Exception as e:
        ctx.error(f"Medical report summarization failed: {str(e)}")
        processing_time = time.time() - start_time
        return {
            "error": f"Summarization failed: {str(e)}",
            "processing_time": f"{processing_time:.2f} seconds",
            "file_path": file_path,
            "file_processed": os.path.basename(file_path),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "success": False
        }

@mcp.tool()
async def process_medical_report(file_path: str, ctx: Context, extraction_method: str = "auto") -> Dict[str, Any]:
    """
    Process a medical report file - extract text and classify it in a single operation.
    
    Args:
        file_path: Path to the PDF or image file
        ctx: MCP context for tracking progress and providing information
        extraction_method: Method to use for extraction ('auto', 'gemini', 'openai', 'ocr')
        
    Returns:
        Processing results including the extracted text, classification, and enhanced data
    """
    print(f"\n[SERVER] Tool called: process_medical_report with file: {file_path}, method: {extraction_method}")
    ctx.info(f"Processing medical report: {file_path}")
    await ctx.report_progress(0, 10)
    
    if not os.path.exists(file_path):
        ctx.error(f"File not found: {file_path}")
        return {"error": f"File not found: {file_path}"}
    
    start_time = time.time()
    result = {}
    
    try:
        # STEP 1: Convert PDF to images if needed
        if file_path.lower().endswith('.pdf'):
            ctx.info("Converting PDF to images...")
            await ctx.report_progress(1, 10)
            
            try:
                image_paths = convert_pdf_to_images(file_path)
                if image_paths and len(image_paths) > 0:
                    ctx.info(f"Converted PDF to {len(image_paths)} images using enhanced method")
                else:
                    # Fall back to original method
                    ctx.warning("Enhanced conversion failed, falling back to original method")
                    image_paths = convert_pdf_to_images(file_path)
            except Exception as conversion_error:
                ctx.warning(f"Enhanced conversion failed: {str(conversion_error)}, falling back to original method")
                image_paths = convert_pdf_to_images(file_path)
                
            if not image_paths:
                ctx.error("Failed to convert PDF to images")
                return {"error": "Failed to convert PDF to images"}
            
            ctx.info(f"Successfully converted PDF to {len(image_paths)} images")
        else:
            # Assume it's already an image
            image_paths = [file_path]
            await ctx.report_progress(1, 10)
        
        # STEP 2: Extract text from the images
        ctx.info(f"Extracting text using method: {extraction_method}")
        await ctx.report_progress(2, 10)
        
        # Try the new Gemini-based extraction method first if auto or gemini is specified
        if extraction_method.lower() in ["auto", "gemini"]:
            ctx.info("Attempting extraction with Gemini 2.0 Flash...")
            try:
                extracted_text = extract_text_from_pdf(image_paths)
                if extracted_text and len(extracted_text.strip()) > 20:
                    ctx.info("Extraction with Gemini 2.0 Flash successful")
                    success = True
                    method_used = "gemini_2.0_flash"
                else:
                    raise Exception("Insufficient text extracted")
            except Exception as gemini_error:
                ctx.warning(f"Gemini 2.0 Flash extraction failed: {str(gemini_error)}")
                
                # Fall back to original methods
                if extraction_method.lower() == "gemini" or extraction_method.lower() == "auto":
                    ctx.info("Falling back to original Gemini method...")
                    extracted_text, success = extract_text_using_gemini(image_paths)
                    method_used = "gemini"
                    
                    # Fall back to OpenAI if Gemini fails and method is 'auto'
                    if not success and extraction_method.lower() == "auto":
                        ctx.info("Original Gemini extraction failed, falling back to OpenAI...")
                        extracted_text, success = extract_text_using_openai(image_paths)
                        method_used = "openai" if success else "ocr"
                        
                        # Fall back to OCR if OpenAI fails
                        if not success:
                            ctx.info("OpenAI extraction failed, falling back to OCR...")
                            from test_identification import extract_text_from_pdf_with_ocr
                            extracted_text = extract_text_from_pdf_with_ocr(image_paths)
                            success = True
                            method_used = "ocr"
        
        elif extraction_method.lower() == "openai":
            ctx.info("Attempting extraction with OpenAI...")
            extracted_text, success = extract_text_using_openai(image_paths)
            method_used = "openai"
            
            # Fall back to OCR if OpenAI fails
            if not success:
                ctx.info("OpenAI extraction failed, falling back to OCR...")
                from test_identification import extract_text_from_pdf_with_ocr
                extracted_text = extract_text_from_pdf_with_ocr(image_paths)
                success = True
                method_used = "ocr"
        
        elif extraction_method.lower() == "ocr":
            ctx.info("Extracting text using OCR...")
            from test_identification import extract_text_from_pdf_with_ocr
            extracted_text = extract_text_from_pdf_with_ocr(image_paths)
            success = True
            method_used = "ocr"
        
        else:
            ctx.error(f"Unknown extraction method: {extraction_method}")
            return {"error": f"Unknown extraction method: {extraction_method}"}
        
        # Store extraction results
        result["extracted_text"] = extracted_text
        result["extraction_method"] = method_used
        await ctx.report_progress(4, 10)
        
        # STEP 3: Classify the report
        try:
            ctx.info("Reading classification system prompt...")
            await ctx.report_progress(5, 10)
            
            # Read classification system prompt
            try:
                with open(os.path.join("System_prompts", "classify_sys_prompt.txt"), "r") as f:
                    classification_system_prompt = f.read()
            except FileNotFoundError:
                ctx.warning("Classification system prompt file not found, using default prompt")
                classification_system_prompt = """
                You are a medical report classifier. Analyze the provided text and determine the type of medical report.
                Provide your classification as a JSON object with fields: 
                - report_type: The specific type of report
                - reason: Brief explanation for your classification
                - confidence: A score between 0 and 100 indicating your confidence
                - matched_keywords: List of keywords that helped identify the report type
                """
            
            ctx.info("Classifying the medical report...")
            await ctx.report_progress(6, 10)
            
            # Try the new Gemini-based classification method first
            try:
                classification_result = classify_report_with_gemini(extracted_text, classification_system_prompt)
                ctx.info("Classification with Gemini successful")
            except Exception as gemini_error:
                ctx.warning(f"Gemini classification failed: {str(gemini_error)}. Falling back to default method.")
                # Fall back to the original classification method
                classification_result = classify_report(extracted_text, classification_system_prompt)
            
            # Ensure result is a dictionary
            if not isinstance(classification_result, dict):
                if isinstance(classification_result, str):
                    try:
                        classification_result = json.loads(classification_result)
                    except json.JSONDecodeError:
                        ctx.warning("Failed to parse classification result as JSON")
                        classification_result = {
                            "report_type": "Unknown",
                            "reason": "Failed to parse classification result",
                            "confidence": 0.0,
                            "keywords_identified": []
                        }
                else:
                    ctx.warning("Classification returned non-dictionary result")
                    classification_result = {
                        "report_type": "Unknown",
                        "reason": "Classification failed",
                        "confidence": 0.0,
                        "keywords_identified": []
                    }
            
            # Standardize result keys
            standardized_result = {
                "report_type": classification_result.get("report_type", "Unknown"),
                "confidence": classification_result.get("confidence_score", classification_result.get("confidence", 0.0)),
                "keywords_identified": classification_result.get("keywords_identified", classification_result.get("matched_keywords", [])),
                "reason": classification_result.get("reason", "")
            }
            
            classification_result = standardized_result
            ctx.info(f"Classification complete: {classification_result.get('report_type', 'Unknown')}")
            
            # Store classification results
            result.update(classification_result)
            await ctx.report_progress(7, 10)
        except Exception as classification_error:
            ctx.warning(f"Classification failed: {str(classification_error)}")
            classification_result = {
                "report_type": "Unknown",
                "confidence": 0.0,
                "keywords_identified": [],
                "reason": f"Classification error: {str(classification_error)}"
            }
            result.update(classification_result)
        
        # STEP 4: Enhance the text if we have a valid report type
        enhanced_text = None
        verified_text = None
        
        report_type = classification_result.get('report_type')
        if report_type and report_type != "Unknown":
            ctx.info(f"Enhancing extracted text for report type: {report_type}")
            await ctx.report_progress(8, 10)
            
            enhanced_text = enhance_text_with_gemini(extracted_text, report_type, image_paths)
            if enhanced_text:
                ctx.info("Text enhancement successful")
                verified_text = verify_results_with_gemini(enhanced_text, image_paths, report_type)
                ctx.info("Verification with Gemini complete")
            else:
                ctx.warning("Text enhancement failed")
        else:
            ctx.info("Skipping enhancement due to unknown report type")
        
        # Store enhancement results
        result["enhanced_text"] = enhanced_text
        result["verified_text"] = verified_text
        await ctx.report_progress(9, 10)
        
        # STEP 5: Finalize results
        processing_time = time.time() - start_time
        result["processing_time"] = f"{processing_time:.2f} seconds"
        result["file_path"] = file_path
        result["file_processed"] = os.path.basename(file_path)
        result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result["success"] = True
        
        # Save results to JSON file
        try:
            base_filename = os.path.splitext(os.path.basename(file_path))[0]
            report_type = classification_result.get('report_type', 'unknown')
            json_path = save_to_json(result, base_filename, report_type)
            result["json_saved_to"] = json_path
        except Exception as save_error:
            ctx.warning(f"Failed to save results to JSON: {str(save_error)}")
        
        await ctx.report_progress(10, 10)
        ctx.info(f"Medical report processing complete in {processing_time:.2f} seconds")
        
        return result
        
    except Exception as e:
        ctx.error(f"Medical report processing failed: {str(e)}")
        processing_time = time.time() - start_time
        return {
            "error": f"Processing failed: {str(e)}",
            "processing_time": f"{processing_time:.2f} seconds",
            "file_path": file_path,
            "file_processed": os.path.basename(file_path),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "success": False
        }
    finally:
        # Clean up temporary images
        try:
            if 'image_paths' in locals() and image_paths:
                for image_path in image_paths:
                    if os.path.exists(image_path) and os.path.dirname(image_path) == "converted_images":
                        try:
                            os.remove(image_path)
                            ctx.info(f"Removed temporary image: {image_path}")
                        except Exception as cleanup_error:
                            ctx.warning(f"Failed to remove temporary image {image_path}: {str(cleanup_error)}")
        except Exception as cleanup_error:
            ctx.warning(f"Error during cleanup: {str(cleanup_error)}")


class GeminiChatInterface:
    """Chat interface using Gemini with MCP tools"""
    
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.messages = []
        
        # Initialize with system message
        self.system_prompt = """You are an AI assistant specialized in medical report processing using MCP (Model Context Protocol) tools.
        
You can help users with:
1. Processing medical reports (extraction and classification in one step)
2. Classifying medical reports
3. Extracting text from medical report files
4. Answering questions about medical reports and terminology

You have access to specialized MCP tools:
- process_medical_report: Processes a medical report file - extracts text and classifies it in a single operation
  Parameters: file_path (required) - Path to the PDF or image file
              extraction_method (optional) - Method to use for extraction ('auto', 'gemini', 'openai', 'ocr')

- classify_medical_report: Classifies the type of medical report based on its content
  Parameters: text (required) - The text content to classify

- extract_text_from_report: Extracts text from medical report PDFs or images
  Parameters: file_path (required) - Path to the PDF or image file
              extraction_method (optional) - Method to use for extraction ('auto', 'gemini', 'openai', 'ocr')

VERY IMPORTANT: When a user wants to use a tool, you MUST use the EXACT format below:
  "use <tool_name> with <parameter>=<value>"

For example:
  "use process_medical_report with file_path='C:\\path\\to\\report.pdf'"
  "use classify_medical_report with text='Patient shows signs of...'"
  "use extract_text_from_report with file_path='C:\\path\\to\\report.pdf'"

DO NOT use any other format like "process_medical_report with file_path=" or "process_medical_report(file_path=)". 
The ONLY correct format is "use <tool_name> with <parameter>=<value>"

Always explain your reasoning and provide helpful context about medical reports when possible.

IMPORTANT: When processing or classifying a report, your response should include:
- The report type (e.g., CBC, Endoscopy, Histopathology)
- Confidence score for the classification
- Key parameters or values extracted from the report
- Any identified keywords or medical terms
- Reasoning for the classification"""
        
        # Initialize the chat model
        self.chat_model = genai.GenerativeModel(
            model_name=self.model,
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 1024,
                "top_p": 0.95
            }
        )
        
        # Start a chat session
        self.chat = self.chat_model.start_chat(history=[])
        
        # Send the system prompt
        self.chat.send_message(self.system_prompt)
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.messages.append({"role": role, "content": content})
        
        # Also add to the Gemini chat session
        if role == "user":
            self.chat.send_message(content)
    
    async def get_completion(self) -> str:
        """Get completion from Gemini API with special handling for extraction requests"""
        try:
            # Get the last user message
            last_user_message = ""
            for msg in reversed(self.messages):
                if msg["role"] == "user":
                    last_user_message = msg["content"].lower()
                    break
            
            # Check if the user is asking to extract text from a specific type of report
            report_type_patterns = {
                "endoscopy": r"extract.*endoscopy|endoscopy.*extract",
                "colonoscopy": r"extract.*colonoscopy|colonoscopy.*extract",
                "histopathology": r"extract.*histopathology|histopathology.*extract",
                "cbc": r"extract.*cbc|cbc.*extract|extract.*complete blood count",
                "clinical_biochemistry": r"extract.*biochemistry|biochemistry.*extract",
                "serum_analysis": r"extract.*serum|serum.*extract"
            }
            
            # Check if this is an extraction request for a specific report type
            detected_report_type = None
            for report_type, pattern in report_type_patterns.items():
                if re.search(pattern, last_user_message, re.IGNORECASE):
                    detected_report_type = report_type
                    break
            
            # If this is a report extraction request, use the appropriate tool
            if detected_report_type and "extract" in last_user_message and "report" in last_user_message:
                # Check if we have a sample report for this type for demonstration
                sample_report_path = os.path.join("sample_reports", f"{detected_report_type}_sample.pdf")
                
                if not os.path.exists(sample_report_path):
                    # If no sample report, explain how to use the extraction tool
                    return f"I detected that you want to extract information from a {detected_report_type} report. " + \
                           f"To do this, please use the extraction tool with a specific file path: \n\n" + \
                           f"use extract_text_from_report with file_path='path/to/your/{detected_report_type}_report.pdf'\n\n" + \
                           f"This will extract the text according to the {detected_report_type} report structure."
                
                # Use the extraction tool with the sample report
                try:
                    print(f"Automatically using extraction tool for {detected_report_type} report...")
                    result = await self.run_mcp_tool("extract_text_from_report", file_path=sample_report_path)
                    return f"I've extracted the information from the {detected_report_type} report:\n\n{result}"
                except Exception as tool_error:
                    print(f"Error using extraction tool: {str(tool_error)}")
                    # Fall back to normal completion if tool fails
            
            # Get the last message from the chat history
            # Since we've already sent the user message in add_message,
            # the last response in the chat history will be the model's response
            last_response = self.chat.history[-1]
            if last_response.role == "model":
                return last_response.parts[0].text
            else:
                # If for some reason we don't have a model response, create a new one
                response = self.chat_model.generate_content(last_user_message)
                return response.text
        except Exception as e:
            return f"Error getting completion: {str(e)}"
    
    async def run_mcp_tool(self, tool_name: str, **kwargs) -> str:
        """Run an MCP tool with the given parameters"""
        ctx = Context()
        
        if tool_name == "process_medical_report":
            if "file_path" not in kwargs:
                return "Error: 'file_path' parameter is required for process_medical_report"
            
            extraction_method = kwargs.get("extraction_method", "auto")
            result = await process_medical_report(kwargs["file_path"], ctx, extraction_method)
            return json.dumps(result, indent=2)
            
        else:
            return f"Error: Unknown tool '{tool_name}'. Available tools: process_medical_report"
    
    async def run_interactive(self):
        """Run the chat interface interactively"""
        print(f"\n{'='*50}")
        print(f"MCP Chat Interface using Gemini ({self.model})")
        print(f"{'='*50}")
        print("Type 'exit' to quit, 'tools' to list available tools, or 'model <name>' to change the model.")
        print("\nTOOL USAGE INSTRUCTIONS:")
        print("To use a tool, you MUST type the command in this EXACT format:")
        print("  use <tool_name> with <parameter>=<value>")
        
        print("\nAvailable tools:")
        print("1. classify_medical_report - Classifies a medical report")
        print("   Example: use classify_medical_report with text='Patient shows signs of...'")
        
        print("\n2. extract_text_from_report - Extracts text from a PDF or image file")
        print("   Example: use extract_text_from_report with file_path='C:\\path\\to\\report.pdf'")
        
        print("\n3. summarize_medical_report - Summarizes a medical report file")
        print("   Example: use summarize_medical_report with file_path='C:\\path\\to\\report.pdf'")
        
        print(f"\n{'='*50}\n")
        
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            
            elif user_input.lower() == 'tools':
                print("\nAvailable MCP tools:")
                print("- process_medical_report: Processes medical report PDFs or images (extracts text and classifies in one step)")
                continue
            
            elif user_input.lower().startswith('model '):
                new_model = user_input[6:].strip()
                if new_model in AVAILABLE_MODELS:
                    self.model = new_model
                    print(f"Model changed to: {self.model}")
                else:
                    print(f"Invalid model. Available models: {', '.join(AVAILABLE_MODELS)}")
                continue
            
            elif user_input.lower().startswith('use '):
                # Parse tool request
                try:
                    # Extract tool name and parameters
                    parts = user_input[4:].split(' with ', 1)
                    if len(parts) != 2:
                        print("Invalid format. Use: use <tool_name> with <parameters>")
                        continue
                    
                    tool_name = parts[0].strip()
                    params_str = parts[1].strip()
                    
                    # Parse parameters
                    params = {}
                    if params_str:
                        # Simple parameter parsing (not a full parser)
                        param_pairs = params_str.split(',')
                        for pair in param_pairs:
                            if '=' in pair:
                                key, value = pair.split('=', 1)
                                key = key.strip()
                                value = value.strip()
                                
                                # Remove quotes if present
                                if value.startswith(("'", '"')) and value.endswith(("'", '"')):
                                    value = value[1:-1]
                                
                                params[key] = value
                    
                    # Execute the tool
                    print(f"\nRunning MCP tool: {tool_name}...")
                    result = await self.run_mcp_tool(tool_name, **params)
                    print(f"\nTool Result ({tool_name}):\n{result}")
                    
                    # Add to conversation
                    self.add_message("user", f"I want to use the {tool_name} tool with these parameters: {params_str}")
                    
                    # Add response to conversation
                    self.add_message("assistant", f"I've executed the {tool_name} tool for you. Here are the results:\n\n{result}")
                except Exception as e:
                    print(f"Error processing tool request: {str(e)}")
                continue
            
            # Regular chat message
            self.add_message("user", user_input)
            print("\nAI: ", end="", flush=True)
            
            response = await self.get_completion()
            print(response)
            
            self.add_message("assistant", response)

async def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description="MCP Medical Report Processing Server")
    parser.add_argument("--port", type=int, default=8050,
                        help="Port to run the MCP server on (default: 8050)")
    parser.add_argument("--transport", type=str, default="stdio",
                        choices=["stdio", "http", "websocket"],
                        help="Transport type for the MCP server (default: stdio)")
    
    args = parser.parse_args()
    
    # Update port if specified
    if args.port != 8050:
        mcp.port = args.port
    
    # Run as MCP server
    print(f"Starting MCP server on port {args.port} with {args.transport} transport...")
    # Tools are already registered with @mcp.tool() decorators
    await mcp.run(transport=args.transport)
import sys

if __name__ == "__main__":
    # This script can be run in two ways:
    # 1. Directly: python mcp_chat.py
    # 2. Through MCP CLI: mcp dev mcp_chat.py
    #
    # When run through MCP CLI, we don't need to call mcp.run() as the CLI handles that
    # When run directly, we need to handle the server startup ourselves
    
    # Check if we're being run directly (not through mcp dev)
    if not os.environ.get("MCP_MANAGED", False):
        # Check if we should use stdio transport directly
        if "--transport=stdio" in sys.argv or ("--transport" not in " ".join(sys.argv) and len(sys.argv) == 1):
            # Direct stdio transport doesn't need asyncio wrapping
            mcp.run(transport="stdio")
        else:
            # For HTTP/WebSocket transport, use asyncio
            import asyncio
            asyncio.run(main())