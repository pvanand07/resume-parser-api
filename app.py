from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import tempfile
import shutil
import uuid
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get OpenRouter API key from environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("Warning: OPENROUTER_API_KEY not found in environment variables. API calls may fail.")

# Import the resume parser functions
from parser_v3 import parse_resume, calculate_file_hash

app = FastAPI(
    title="Resume Parser API",
    description="API for parsing resumes and extracting structured data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create directories if they don't exist
os.makedirs("static", exist_ok=True)

# Store parsed results in memory (for development)
# In production, you would use a database
parsed_results = {}

# Directory to store uploaded resumes temporarily
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Results file to store parsed data
RESULTS_FILE = "resume_results.json"
PROGRESS_FILE = "resume_progress.json"

def load_existing_results():
    """Load existing results from JSON file if it exists."""
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Error loading existing results from {RESULTS_FILE}. Starting with empty results.")
    return {}

def load_progress():
    """Load progress data from JSON file if it exists."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Error loading progress from {PROGRESS_FILE}. Starting with empty progress.")
    return {}

def save_results(results):
    """Save results to JSON file."""
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

def save_progress(progress):
    """Save progress to JSON file."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

# Load existing results and progress on startup
parsed_results = load_existing_results()
processing_progress = load_progress()

def process_resume_background(file_path: str, remove_pii: bool = False):
    """Process a resume in the background and save results."""
    try:
        # Calculate hash first to avoid unbound variable errors
        file_hash = calculate_file_hash(file_path)
        hash_prefix = file_hash[:4]
        
        # Update progress to "processing"
        processing_progress[hash_prefix] = {
            "status": "processing",
            "started_at": datetime.now().isoformat(),
            "filename": os.path.basename(file_path)
        }
        save_progress(processing_progress)
        
        # Call parse_resume function with the API key from env variables
        result = parse_resume(file_path, api_key=OPENROUTER_API_KEY, remove_pii=remove_pii)
        
        # Add filename and full hash for reference
        result["source_file"] = os.path.basename(file_path)
        result["file_hash"] = file_hash
        result["processed_at"] = datetime.now().isoformat()
        
        # Update in-memory results and progress
        parsed_results[hash_prefix] = result
        processing_progress[hash_prefix]["status"] = "completed"
        processing_progress[hash_prefix]["completed_at"] = datetime.now().isoformat()
        
        # Save to files
        save_results(parsed_results)
        save_progress(processing_progress)
        
    except Exception as e:
        # Capture file_hash in a separate variable to avoid unbound errors
        if 'file_hash' in locals():
            hash_prefix = file_hash[:4]
            error_message = f"Processing failed: {str(e)}"
            print(f"Error processing {file_path}: {str(e)}")
            
            # Update progress with error status
            processing_progress[hash_prefix] = {
                "status": "error",
                "error": error_message,
                "failed_at": datetime.now().isoformat(),
                "filename": os.path.basename(file_path)
            }
            save_progress(processing_progress)
            
            parsed_results[hash_prefix] = {"error": error_message}
            save_results(parsed_results)
        else:
            print(f"Critical error processing {file_path}, couldn't calculate hash: {str(e)}")
    
    # Clean up the temporary file
    if os.path.exists(file_path):
        os.remove(file_path)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the HTML frontend."""
    html_file = Path("static/index.html")
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(), status_code=200)
    else:
        # Return HTMLResponse instead of dict to avoid AttributeError with 'encode'
        return HTMLResponse(content="<html><body><h1>Resume Parser API is running</h1><p>Frontend HTML file not found.</p></body></html>")

@app.post("/parse-resume/")
async def parse_resume_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    remove_pii: bool = Form(False)
):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OpenRouter API key not configured")

    file_extension = os.path.splitext(file.filename.lower())[1]
    if file_extension not in ['.pdf', '.docx']:
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")
    
    # Create a temporary file
    temp_dir = tempfile.mkdtemp(dir=UPLOAD_DIR)
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    try:
        # Save the uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Calculate hash immediately to return
        file_hash = calculate_file_hash(temp_file_path)
        hash_prefix = file_hash[:4]
        
        # Process in background
        background_tasks.add_task(
            process_resume_background, 
            temp_file_path,
            remove_pii
        )
        
        return {
            "message": "Resume parsing started",
            "hash_prefix": hash_prefix,
            "status": "processing"
        }
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")

@app.post("/parse-multiple-resumes/")
async def parse_multiple_resumes(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    remove_pii: bool = Form(False)
):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OpenRouter API key not configured")
        
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    hash_prefixes = []
    
    for file in files:
        file_extension = os.path.splitext(file.filename.lower())[1]
        if file_extension not in ['.pdf', '.docx']:
            continue  # Skip unsupported files
        
        # Create a temporary file
        temp_dir = tempfile.mkdtemp(dir=UPLOAD_DIR)
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        try:
            # Save the uploaded file
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Calculate hash
            file_hash = calculate_file_hash(temp_file_path)
            hash_prefix = file_hash[:4]
            hash_prefixes.append(hash_prefix)
            
            # Process in background
            background_tasks.add_task(
                process_resume_background, 
                temp_file_path,
                remove_pii
            )
        
        except Exception as e:
            # Clean up on error but continue with other files
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            print(f"Error processing {file.filename}: {str(e)}")
    
    return {
        "message": f"Processing {len(hash_prefixes)} resumes",
        "hash_prefixes": hash_prefixes,
        "status": "processing"
    }

@app.get("/results/{hash_prefix}")
def get_result(hash_prefix: str):
    # First check if it's in parsed results
    if hash_prefix in parsed_results:
        return parsed_results[hash_prefix]
    
    # If not in parsed results, check progress
    if hash_prefix in processing_progress:
        progress_info = processing_progress[hash_prefix]
        if progress_info["status"] == "processing":
            return {
                "status": "processing",
                "message": f"Resume {hash_prefix} is still being processed",
                "started_at": progress_info["started_at"],
                "filename": progress_info["filename"]
            }
        elif progress_info["status"] == "error":
            return {
                "status": "error",
                "message": progress_info["error"],
                "failed_at": progress_info["failed_at"],
                "filename": progress_info["filename"]
            }
    
    raise HTTPException(status_code=404, detail="Result not found")

@app.get("/results")
def get_all_results():
    # Combine information from both files
    combined_results = []
    
    # Add all items from progress tracking
    for hash_prefix, progress in processing_progress.items():
        if hash_prefix in parsed_results:
            status = "completed"
        else:
            status = progress["status"]
            
        combined_results.append({
            "hash_prefix": hash_prefix,
            "status": status,
            "filename": progress.get("filename", "Unknown"),
            "timestamp": progress.get("started_at", "Unknown")
        })
    
    return {
        "count": len(combined_results),
        "results": combined_results
    }

@app.delete("/results/{hash_prefix}")
def delete_result(hash_prefix: str):
    deleted = False
    
    if hash_prefix in parsed_results:
        del parsed_results[hash_prefix]
        save_results(parsed_results)
        deleted = True
        
    if hash_prefix in processing_progress:
        del processing_progress[hash_prefix]
        save_progress(processing_progress)
        deleted = True
    
    if deleted:
        return {"message": f"Result {hash_prefix} deleted successfully"}
    
    raise HTTPException(status_code=404, detail="Result not found")

@app.get("/resume-results-raw")
def get_resume_results_raw():
    # Load the JSON file
    with open(RESULTS_FILE, 'r') as f:
        parsed_results = json.load(f)
    return parsed_results

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)