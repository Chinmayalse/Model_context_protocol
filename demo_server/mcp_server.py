"""
WebSocket-only MCP server for Medical Report Processing
"""
import os
import sys
import json
import asyncio
import uuid

import argparse
from typing import List, Dict, Any, Optional, Union
import time
from datetime import datetime
import re
import io
import base64

import websockets
import pypdfium2 as pdfium
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, ImageFilter, ImageEnhance
import google.generativeai as genai
from mcp.server.fastmcp import FastMCP, Context

from test_identification import (
    process_pdf,
    classify_report,
    extract_text_using_gemini,
    extract_text_using_openai,
    extract_text_from_pdf_with_ocr,
    load_system_prompt as old_load_system_prompt
)

# --- Configuration ---
GOOGLE_API_KEY = "AIzaSyDDBBTclRJjECny3q01Y57TIG9C6ZfVuTY"
genai.configure(api_key=GOOGLE_API_KEY)

MODEL = genai.GenerativeModel('gemini-2.0-flash')


# MCP Server
mcp = FastMCP(
    name="Medical Report Processing",
    description="Tools for processing and summarizing medical reports",
    host="0.0.0.0",
    port=8050
)

# --- Helper Functions ---
def convert_pdf_to_images_option1(pdf_path):
    """
    Convert PDF to images using pdf2image library
    """
    try:
        # Convert all pages
        images = convert_from_path(pdf_path)
        image_paths = []
        
        # Create a standardized name base for the images
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # Save each page as a separate image
        for i, image in enumerate(images):
            image_path = f"{base_name}_page_{i+1}_converted.png"
            image.save(image_path, "PNG")
            image_paths.append(image_path)
        
        return image_paths
    except Exception as e:
        return None

def convert_pdf_to_images_option2(pdf_path, scale=300/72):
    """
    Convert PDF to images using pypdfium2 library
    """
    try:
        pdf_file = pdfium.PdfDocument(pdf_path)
        # Get all pages
        page_indices = [i for i in range(len(pdf_file))]
        
        # Create a standardized name base for the images
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # Render pages
        renderer = pdf_file.render(
            pdfium.PdfBitmap.to_pil,
            page_indices=page_indices,
            scale=scale,
        )
        
        image_paths = []
        # Convert generator to list and save all images
        images = list(renderer)
        for i, image in enumerate(images):
            image_path = f"{base_name}_page_{i+1}_converted.png"
            image.save(image_path, format='PNG')
            image_paths.append(image_path)
            
        return image_paths
        
    except Exception as e:
        return None


def convert_pdf_to_images(pdf_path):
    """
    Convert PDF to images with fallback mechanism
    
    Args:
        pdf_path (str): Path to the PDF file to convert
        
    Returns:
        list: List of paths to the converted image files, or None if conversion failed
    """
    try:
        # Create output directory if it doesn't exist using absolute path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'converted_images')
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            # Ensure the directory is writable
            test_file = os.path.join(output_dir, '.permission_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except (OSError, IOError) as e:
            print(f"[ERROR] Cannot write to output directory {output_dir}: {e}")
            # Try using temp directory as fallback
            import tempfile
            output_dir = tempfile.mkdtemp(prefix='mcp_converted_')
            print(f"Using temporary directory: {output_dir}")
        
        print(f"Using output directory: {output_dir}")
        
        # Get base name for the images
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        final_image_paths = []
        
        # Attempt Option 1
        try:
            image_paths = convert_pdf_to_images_option1(pdf_path)
            if image_paths and len(image_paths) > 0:
                # Process each image
                insufficient_text = False
                temp_image_paths = []
                
                for i, img_path in enumerate(image_paths):
                    text = extract_text_from_image(img_path)
                    word_count = len(text.split())
                    print(f"Option 1 - Page {i+1} word count: {word_count}")
                    
                    if word_count < 14:  # Check if word count is less than 14
                        insufficient_text = True
                        print(f"Page {i+1} has insufficient text ({word_count} words)")
                        break
                    
                    final_path = os.path.join(output_dir, f"{base_name}_page_{i+1}_converted.png")
                    if os.path.exists(img_path):
                        os.replace(img_path, final_path)
                        temp_image_paths.append(final_path)
                
                # Clean up original images
                for img_path in image_paths:
                    if os.path.exists(img_path):
                        os.remove(img_path)
                
                if not insufficient_text:
                    return temp_image_paths
                else:
                    print("Switching to Option 2 due to insufficient text")
                    # Clean up Option 1 converted images
                    for path in temp_image_paths:
                        if os.path.exists(path):
                            os.remove(path)
        
        except Exception as e:
            print(f"Error in Option 1: {e}")
        
        # Use Option 2
        try:
            print("Attempting Option 2 conversion")
            image_paths = convert_pdf_to_images_option2(pdf_path)
            if image_paths and len(image_paths) > 0:
                # Move each image to the final location
                for i, img_path in enumerate(image_paths):
                    final_path = os.path.join(output_dir, f"{base_name}_page_{i+1}_converted.png")
                    if os.path.exists(img_path):
                        os.replace(img_path, final_path)
                        final_image_paths.append(final_path)
                return final_image_paths
        except Exception as e:
            print(f"Error in Option 2: {e}")
        
        return None
        
    except Exception as e:
        print(f"[ERROR] Unexpected error in convert_pdf_to_images: {e}")
        import traceback
        print(traceback.format_exc())
        return None


def preprocess_image(image_path):
    """
    Preprocess image for better OCR results
    """
    image = Image.open(image_path).convert("L")
    # Apply a median filter to reduce noise
    image = image.filter(ImageFilter.MedianFilter(size=3))
    # Sharpen the image
    image = image.filter(ImageFilter.SHARPEN)
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    # Convert to binary image
    image = image.point(lambda x: 0 if x < 128 else 255, '1')
    return image


def extract_text_from_image(image_path):
    """
    Extract text from an image using Tesseract OCR
    """
    preprocessed_image = preprocess_image(image_path)
    text = pytesseract.image_to_string(preprocessed_image, config='--psm 6 --oem 3')
    return text


def extract_text_with_gemini(image_paths):
    """
    Extract text from images using Gemini 1.5 Flash model
    """
    try:
        prompt = """
        Extract all text visible in these images. Return only the raw text, exactly as it appears, 
        without any additional comments or labels. Maintain the original structure and formatting as closely as possible.
        """
        
        content_parts = [prompt]
        
        # Add images to content parts
        for image_path in image_paths:
            try:
                image = Image.open(image_path)
                content_parts.append(image)
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue
        
        response = MODEL.generate_content(
            content_parts,
            generation_config={
                'temperature': 0.1,
                'top_p': 0.95,
                'max_output_tokens': 4096,
            }
        )
        
        if not response or not response.text:
            raise Exception("Empty response from Gemini")
        
        extracted_text = response.text.strip()
        return extracted_text
        
    except Exception as e:
        print(f"Error in Gemini text extraction: {e}")
        return None


def extract_text_from_pdf(image_paths):
    """
    Extract text from PDF using both Gemini and OCR, with Gemini as the primary method
    and OCR as fallback
    """
    # First try: Gemini-based extraction
    print("Attempting text extraction using Gemini 2.0 Flash...")
    gemini_text = extract_text_with_gemini(image_paths)
    
    # Check if Gemini extraction was successful
    if gemini_text and len(gemini_text.strip().split()) > 20:
        print(f"Gemini extraction successful: {len(gemini_text.strip().split())} words extracted")
        return gemini_text
    
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


def load_system_prompt(report_type):
    """
    Load system prompt based on report type.
    Expected format: {report_type}_prompt.txt
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


def save_to_json(result, base_filename, report_type=None):
    """
    Save results to a JSON file with timestamp in a readable format.
    Each report is saved in its own file within a type-specific directory.
    
    Args:
        result: The data to save (dict or JSON string)
        base_filename: Base name for the output file (without extension)
        report_type: Optional report type for organizing into subdirectories
        
    Returns:
        str: Path to the saved JSON file, or None if saving failed
    """
    from datetime import datetime
    import json
    import tempfile
    import shutil
    
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create base output directory if it doesn't exist
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_output_dir = os.path.join(script_dir, "results")
        
        try:
            os.makedirs(base_output_dir, exist_ok=True)
            # Test if directory is writable
            test_file = os.path.join(base_output_dir, '.permission_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except (OSError, IOError) as e:
            print(f"[ERROR] Cannot write to output directory {base_output_dir}: {e}")
            # Fallback to temp directory
            base_output_dir = tempfile.mkdtemp(prefix='mcp_results_')
            print(f"Using temporary directory: {base_output_dir}")
        
        # Create report type specific directory
        if report_type:
            report_dir = os.path.join(base_output_dir, report_type.lower())
            os.makedirs(report_dir, exist_ok=True)
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
        temp_path = f"{json_path}.tmp"
        
        try:
            # Write to a temporary file first
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(result_to_save, f, indent=4, ensure_ascii=False, sort_keys=True)
            
            # Rename the temporary file to the final name (atomic operation)
            if os.path.exists(json_path):
                os.remove(json_path)
            os.rename(temp_path, json_path)
            
            print(f"\nJSON results saved to {json_path}")
            return json_path
            
        except (IOError, OSError) as e:
            print(f"[ERROR] Failed to save JSON file: {e}")
            # Try saving to a temporary directory as last resort
            try:
                temp_dir = tempfile.mkdtemp(prefix='mcp_temp_')
                temp_json = os.path.join(temp_dir, os.path.basename(json_file))
                with open(temp_json, 'w', encoding='utf-8') as f:
                    json.dump(result_to_save, f, indent=4, ensure_ascii=False, sort_keys=True)
                print(f"[WARNING] Saved to temporary location: {temp_json}")
                return temp_json
            except Exception as inner_e:
                print(f"[CRITICAL] Failed to save JSON to temporary location: {inner_e}")
                return None
                
    except Exception as e:
        print(f"[ERROR] Unexpected error in save_to_json: {e}")
        import traceback
        print(traceback.format_exc())
        return None

# --- MCP Tools ---
@mcp.tool()
async def summarize_medical_report(file_path: str, ctx) -> Dict[str, Any]:
    """
    Summarize a medical report file - extract text and provide a structured summary.
    
    Args:
        file_path: Path to the PDF or image file
        ctx: MCP context for tracking progress and providing information
        
    Returns:
        Summary results including the extracted text and structured data
        
    Raises:
        FileNotFoundError: If the input file does not exist
        RuntimeError: If there's an error during processing
    """
    print(f"\n[SERVER] Tool called: summarize_medical_report with file: {file_path}")
    ctx.info(f"Summarizing medical report: {file_path}")
    await ctx.report_progress(0, 5)
    
    # Track temporary files for cleanup
    temp_files = []
    temp_dirs = []
    
    if not os.path.exists(file_path):
        error_msg = f"File not found: {file_path}"
        ctx.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    start_time = time.time()
    result = {}
    image_paths = []
    
    try:
        # STEP 1: Convert PDF to images if needed
        if file_path.lower().endswith('.pdf'):
            ctx.info("Converting PDF to images...")
            await ctx.report_progress(1, 5)
            
            try:
                # First try with enhanced method
                image_paths = convert_pdf_to_images(file_path)
                if image_paths and len(image_paths) > 0:
                    ctx.info(f"Converted PDF to {len(image_paths)} images using enhanced method")
                    temp_files.extend(image_paths)  # Track for cleanup
                else:
                    # Fall back to original method
                    ctx.warning("Enhanced conversion failed, falling back to original method")
                    image_paths = convert_pdf_to_images(file_path)
                    if image_paths:
                        temp_files.extend(image_paths)  # Track for cleanup
            except Exception as conversion_error:
                ctx.warning(f"Enhanced conversion failed: {str(conversion_error)}, falling back to original method")
                image_paths = convert_pdf_to_images(file_path)
                if image_paths:
                    temp_files.extend(image_paths)  # Track for cleanup
                
            if not image_paths:
                error_msg = "Failed to convert PDF to images after multiple attempts"
                ctx.error(error_msg)
                raise RuntimeError(error_msg)
            
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
            if json_path:
                result["_json_path"] = json_path
        except Exception as save_error:
            ctx.warning(f"Failed to save results to JSON: {str(save_error)}")
        
        await ctx.report_progress(5, 5)
        ctx.info(f"Summary generation complete in {processing_time:.2f} seconds")
        
        return result
        
    except Exception as e:
        ctx.error(f"Medical report summarization failed: {str(e)}")
        return {
            "error": f"Failed to process medical report: {str(e)}",
            "file_path": file_path,
            "success": False
        }
        
    finally:
        # Clean up temporary files and directories
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    ctx.debug(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                ctx.warning(f"Failed to clean up temporary file {temp_file}: {str(e)}")
        
        for temp_dir in temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    ctx.debug(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                ctx.warning(f"Failed to clean up temporary directory {temp_dir}: {str(e)}")

@mcp.tool()
async def process_medical_report(file_path: str, ctx, extraction_method: str = "auto") -> Dict[str, Any]:
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

# --- Custom Context for WebSocket ---
class WebSocketContext:
    """A simplified context for WebSocket handlers that mimics the FastMCP Context."""
    
    def __init__(self, websocket=None):
        self.websocket = websocket
        self.messages = []
    
    async def report_progress(self, current, total):
        """Report progress to the client."""
        if self.websocket:
            try:
                await self.websocket.send(json.dumps({
                    'event': 'progress',
                    'current': current,
                    'total': total
                }))
            except Exception as e:
                print(f"Error reporting progress: {str(e)}")
    
    def info(self, message):
        """Log an informational message."""
        print(f"[INFO] {message}")
        self.messages.append({"level": "info", "message": message})
    
    def warning(self, message):
        """Log a warning message."""
        print(f"[WARNING] {message}")
        self.messages.append({"level": "warning", "message": message})
    
    def error(self, message):
        """Log an error message."""
        print(f"[ERROR] {message}")
        self.messages.append({"level": "error", "message": message})

clients = set()
clients_by_chat_id = {}

async def websocket_handler(ws):
    """Handle WebSocket connections for MCP tools with robust error handling."""
    # Register client
    clients.add(ws)
    connection_id = str(uuid.uuid4())
    print(f"[SERVER] New connection established: {connection_id}")
    
    # Track active tasks for this connection
    active_tasks = set()
    
    # Track connection state
    connection_active = True
    
    # Set up ping/pong monitoring
    last_pong = time.time()
    
    # Define ping handler to track connection health
    async def ping_handler():
        nonlocal connection_active, last_pong
        try:
            while connection_active:
                await asyncio.sleep(15)  # Send ping every 15 seconds
                if not connection_active:
                    break
                try:
                    pong_waiter = await ws.ping()
                    await asyncio.wait_for(pong_waiter, timeout=10)
                    last_pong = time.time()
                except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                    print(f"[SERVER] Ping timeout for connection {connection_id}, marking as inactive")
                    connection_active = False
                    break
        except Exception as e:
            print(f"[SERVER] Error in ping handler for {connection_id}: {e}")
            connection_active = False
    
    # Start ping monitoring task
    ping_task = asyncio.create_task(ping_handler())
    
    # Define available tools
    tool_defs = [
        {
            "name": "summarize_medical_report",
            "description": "Summarize a medical report file - extract text and provide a structured summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the PDF or image file"
                    }
                },
                "required": ["file_path"]
            }
        },
        {
            "name": "process_medical_report",
            "description": "Process a medical report file - extract text and classify it in a single operation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the PDF or image file"
                    },
                    "extraction_method": {
                        "type": "string",
                        "description": "Method to use for text extraction (auto, gemini, ocr)",
                        "default": "auto"
                    }
                },
                "required": ["file_path"]
            }
        }
    ]
    
    async def send_message(message):
        """Safely send a message to the WebSocket client."""
        nonlocal connection_active
        
        if not connection_active:
            print(f"[SERVER] Not sending message to {connection_id} - connection already closed")
            return False
            
        try:
            if not isinstance(message, str):
                message = json.dumps(message)
            await ws.send(message)
            return True
        except (websockets.exceptions.ConnectionClosed, ConnectionResetError) as e:
            print(f"[SERVER] Connection {connection_id} closed while sending: {e}")
            connection_active = False
            return False
        except Exception as e:
            print(f"[SERVER] Error sending message to {connection_id}: {e}")
            import traceback
            print(traceback.format_exc())
            return False

    # Send initial connection message with tool definitions
    try:
        if not await send_message({
            "event": "connected",
            "connection_id": connection_id,
            "tools": tool_defs
        }):
            # If we can't send the initial message, the connection is already closed
            print(f"[SERVER] Connection {connection_id} closed before initial message")
            return
            
        print(f"[SERVER] Connection {connection_id} ready to receive messages")
        
        # Process incoming messages
        while connection_active:
            try:
                # Add a small sleep to prevent high CPU usage
                await asyncio.sleep(0.1)
                
                # Check if the connection has been inactive for too long
                if time.time() - last_pong > 60:  # 60 seconds without a pong
                    print(f"[SERVER] Connection {connection_id} has been inactive for too long, closing")
                    connection_active = False
                    break
                    
                try:
                    # Use a shorter timeout to detect connection issues faster
                    raw = await asyncio.wait_for(ws.recv(), timeout=30)  # 30 second timeout
                except asyncio.TimeoutError:
                    print(f"[SERVER] Receive timeout for connection {connection_id}, checking connection health")
                    # Don't break immediately, let the ping handler determine if the connection is still alive
                    continue
                except websockets.exceptions.ConnectionClosed as e:
                    print(f"[SERVER] Connection {connection_id} was closed by the client: {e}")
                    connection_active = False
                    break
                except Exception as e:
                    print(f"[SERVER] Unexpected error receiving message from {connection_id}: {str(e)}")
                    # Log the full traceback for debugging
                    import traceback
                    print(traceback.format_exc())
                    # Continue and let the ping handler determine if the connection is still alive
                    continue
                    
                data = json.loads(raw)
                action = data.get('action')
                rid = data.get('request_id', str(uuid.uuid4()))
                
                if action == 'select_tool':
                    # Simple heuristic for tool selection
                    try:
                        msg = data.get('message', '').lower()
                        eff = 'summarize_medical_report' if 'summar' in msg else 'process_medical_report'
                        await send_message({
                            'event': 'tool_selected',
                            'request_id': rid,
                            'tool_name': eff,
                            'confidence': 0.8,
                            'args': {}
                        })
                    except Exception as e:
                        print(f"[SERVER] Error in select_tool for {connection_id}: {e}")
                        await send_message({
                            'event': 'error',
                            'request_id': rid,
                            'error': f'Error in tool selection: {str(e)}'
                        })
                    
                elif action == 'call_tool':
                    tool = data.get('tool_name')
                    args = data.get('args', {})
                    
                    # Validate tool name
                    if tool not in ['summarize_medical_report', 'process_medical_report']:
                        await send_message({
                            'event': 'tool_error',
                            'request_id': rid,
                            'tool': tool,
                            'error': f'Unknown tool: {tool}'
                        })
                        continue
                    
                    # Send acceptance message
                    if not await send_message({
                        'event': 'tool_call_accepted',
                        'request_id': rid,
                        'tool': tool
                    }):
                        continue  # Connection closed, exit early
                    
                    # Define the runner function to execute the tool
                    async def runner():
                        nonlocal connection_active
                        task = asyncio.current_task()
                        active_tasks.add(task)
                        try:
                            # Check if connection is still active before proceeding
                            if not connection_active:
                                print(f"[SERVER] Skipping tool execution for {tool} - connection {connection_id} is closed")
                                return
                                
                            # Use our custom WebSocketContext instead of FastMCP Context
                            ctx = WebSocketContext(websocket=ws)
                            try:
                                # Get the function from the module
                                fn = getattr(sys.modules[__name__], tool)
                                
                                # Execute the function with the provided arguments
                                res = await fn(**args, ctx=ctx)
                                
                                # Include any context messages in the result
                                if hasattr(ctx, 'messages') and ctx.messages and isinstance(res, dict):
                                    res['context_messages'] = ctx.messages
                                
                                # Check connection again before sending result
                                if not connection_active:
                                    print(f"[SERVER] Not sending result for {tool} - connection {connection_id} is closed")
                                    return
                                    
                                # Send the result back to the client
                                if not await send_message({
                                    'event': 'tool_result',
                                    'request_id': rid,
                                    'tool': tool,
                                    'result': res
                                }):
                                    # If we can't send the result, the connection is closed
                                    connection_active = False
                                    return
                                    
                            except Exception as e:
                                error_msg = f"Tool execution failed: {str(e)}"
                                print(f"[ERROR] {error_msg}")
                                import traceback
                                print(traceback.format_exc())
                                
                                # Check connection before sending error
                                if connection_active:
                                    await send_message({
                                        'event': 'tool_error',
                                        'request_id': rid,
                                        'tool': tool,
                                        'error': error_msg
                                    })
                                
                        except asyncio.CancelledError:
                            print(f"[SERVER] Task for {tool} was cancelled")
                            raise
                        except Exception as e:
                            print(f"[SERVER] Unexpected error in task: {e}")
                            import traceback
                            print(traceback.format_exc())
                        finally:
                            active_tasks.discard(task)
                    
                    # Start the runner in a separate task
                    task = asyncio.create_task(runner())
                    # Add a callback to handle task completion
                    task.add_done_callback(lambda t: active_tasks.discard(t) if t in active_tasks else None)
                    
                elif action == 'ping':
                    # Respond to ping with pong
                    await send_message({
                        'event': 'pong',
                        'timestamp': time.time(),
                        'request_id': rid
                    })
                    
                else:
                    # Handle unknown action
                    await send_message({
                        'event': 'error',
                        'request_id': rid,
                        'error': f'Unknown action: {action}'
                    })
                    
            except json.JSONDecodeError:
                print(f"[SERVER] Received invalid JSON from client {connection_id}")
                await send_message({
                    'event': 'error',
                    'error': 'Invalid JSON received'
                })
                
            except asyncio.TimeoutError:
                print(f"[SERVER] Connection {connection_id} timed out")
                if not await send_message({
                    'event': 'error',
                    'error': 'Connection timed out'
                }):
                    # If we can't send the message, the connection is already closed
                    connection_active = False
                    break
                    
                # Keep the connection open but send a ping to check if it's still alive
                try:
                    await ws.ping()
                except:
                    connection_active = False
                    break
                
            except json.JSONDecodeError as e:
                print(f"[SERVER] Received invalid JSON from client {connection_id}: {e}")
                if not await send_message({
                    'event': 'error',
                    'error': 'Invalid JSON received'
                }):
                    connection_active = False
                    break
                
            except Exception as e:
                print(f"[SERVER] Connection {connection_id} handler error: {str(e)}")
                import traceback
                print(traceback.format_exc())
                
                if not await send_message({
                    'event': 'error',
                    'error': f'Internal server error: {str(e)}'
                }):
                    connection_active = False
                    break
                    
    except websockets.exceptions.ConnectionClosed as e:
        print(f"[SERVER] Connection {connection_id} was closed by the client: {e}")
        connection_active = False
    except Exception as e:
        print(f"[SERVER] Connection {connection_id} error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        connection_active = False
        
    finally:
        # Clean up when the connection is closed
        try:
            # Set connection as inactive to prevent further message sends
            connection_active = False
            
            # Cancel the ping task if it's still running
            if 'ping_task' in locals() and not ping_task.done():
                print(f"[SERVER] Cancelling ping task for connection {connection_id}")
                try:
                    ping_task.cancel()
                    # Give the task a short time to clean up
                    try:
                        await asyncio.wait_for(asyncio.shield(ping_task), timeout=1.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                        pass
                except Exception as e:
                    print(f"[SERVER] Error cancelling ping task: {e}")
            
            # Cancel all running tasks for this connection
            if active_tasks:
                print(f"[SERVER] Cancelling {len(active_tasks)} active tasks for connection {connection_id}")
                for task in list(active_tasks):
                    if not task.done():
                        try:
                            task.cancel()
                            # Give the task a short time to clean up
                            try:
                                await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
                            except (asyncio.CancelledError, asyncio.TimeoutError, Exception) as e:
                                pass
                        except Exception as e:
                            print(f"[SERVER] Error cancelling task: {e}")
            
            # Remove from clients set
            if ws in clients:
                clients.remove(ws)
                
            # Try to close the WebSocket if possible
            try:
                await ws.close()
            except Exception as e:
                print(f"[SERVER] Error closing WebSocket: {e}")
                
            print(f"[SERVER] Connection {connection_id} cleaned up ({len(clients)} active connections remaining)")
            
        except Exception as e:
            print(f"[SERVER] Error during connection cleanup: {e}")
            import traceback
            print(traceback.format_exc())

# --- Main Entrypoint ---
async def main():
    # Configure WebSocket server with more robust settings
    ws_srv = await websockets.serve(
        websocket_handler, 
        '0.0.0.0', 
        8765,
        ping_interval=20,  # Send a ping every 20 seconds
        ping_timeout=30,   # Wait 30 seconds for a pong before closing
        close_timeout=10,  # Wait 10 seconds for the close handshake
        max_size=10 * 1024 * 1024  # 10MB max message size
    )
    print('[SERVER] Chat WS on ws://0.0.0.0:8765 with enhanced connection settings')
    # no mcp.run needed — your handler invokes the tools for you
    await ws_srv.wait_closed()


if __name__=='__main__':
    asyncio.run(main())