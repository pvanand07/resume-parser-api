# Resume Parser API

A FastAPI-based REST API for parsing resume PDFs and extracting structured data.

## Features

- Upload multiple resume PDFs for processing
- Background processing with job status tracking
- Uses MD5 hash prefixes to track processing status and avoid duplicate processing
- Option to remove personal identifiable information (PII)
- RESTful endpoints for all operations
- Results stored in a JSON file for persistence

## Endpoints

- `GET /`: Check if the API is running
- `POST /upload-resumes`: Upload multiple PDF files for processing
- `GET /job-status/{job_id}`: Check the status of a background processing job
- `DELETE /job/{job_id}`: Cancel a running background job
- `GET /results`: Get all processed resume results
- `GET /result/{hash_prefix}`: Get a specific resume result by hash prefix

## Setup and Installation

### Using Docker

1. Build the Docker image:
   ```
   docker build -t resume-parser-api .
   ```

2. Run the container:
   ```
   docker run -p 8000:8000 -v $(pwd)/data:/data -e OPENROUTER_API_KEY=your_api_key resume-parser-api
   ```

### Manual Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set environment variables:
   ```
   export OPENROUTER_API_KEY=your_api_key
   export RESULTS_FILE=path/to/resume_results.json
   ```

3. Run the API:
   ```
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

## Usage Examples

### Upload Resumes

```bash
curl -X POST "http://localhost:8000/upload-resumes" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@resume1.pdf" \
  -F "files=@resume2.pdf" \
  -F "remove_pii=true"
```

### Check Job Status

```bash
curl -X GET "http://localhost:8000/job-status/{job_id}" \
  -H "accept: application/json"
```

### Get Results for a Specific Resume

```bash
curl -X GET "http://localhost:8000/result/{hash_prefix}" \
  -H "accept: application/json"
```

## API Documentation

When the API is running, you can access the auto-generated Swagger documentation at:
http://localhost:8000/docs

## Environment Variables

- `OPENROUTER_API_KEY`: API key for OpenRouter API (required)
- `RESULTS_FILE`: Path to the JSON file for storing results (default: "resume_results.json").

## License

MIT 