# MarkItDown API Endpoint

FastAPI server providing a Wrapper for Microsoft MarkItDown 

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file with:
```
API_KEY=your_api_key_here
OPENAI_API_KEY=your_openai_key_here
```

## Running the Server

Start the server with:
```bash
uvicorn index:app --host 0.0.0.0 --port 8080
```

The server will run on `http://127.1.1.1:8080` with auto-reload enabled for development.

## Deployment

This is a standard FastAPI application that can be deployed to any Python hosting service. Here are some deployment options:

### Local Development
Run the server locally using the instructions in the "Running the Server" section above.

### Cloud Deployment Options

The service can be deployed to various cloud platforms, including:
- Heroku
- Google Cloud Run
- AWS Lambda/EC2
- Azure App Service
- Leapcell
- Any other Python hosting service that supports FastAPI

#### Example: Leapcell Configuration
If using Leapcell, these are the build settings:

- **Framework Preset**: fastapi
- **Branch**: main
- **Root Directory**: ./
- **Runtime**: Python (python3.11 debian)
- **Build Command**: pip install -r requirements.txt
- **Start Command**:uvicorn index:app --host 0.0.0.0 --port 8080
- **Serving Port**: 8080

Remember to configure your environment variables (API_KEY and OPENAI_API_KEY) in your chosen platform's dashboard.

## API Usage

### Converting Files to Markdown

Send a POST request to the root endpoint `/` with:
- Header: `API_KEY: your-endpoint-api-key`
- Body: Form data with a `file` field containing your document

Example using cURL:
```bash
curl -X POST \
     -H "API_KEY: your-endpoint-api-key" \
     -F "file=@your-file.jpg" \
     "https://your-api-url/"
```

## Overview

This endpoint serves as a bridge between various document formats and text-based RAG systems. By converting files to markdown format, it enables:
- Integration with no-code automation platforms
- Building RAG systems without coding knowledge
- Automated document processing pipelines
- Easy integration with vector databases and knowledge bases

## Supported File Types

The service supports converting various file formats to Markdown, including:
- PDF documents
- Microsoft Office files (PowerPoint, Word, Excel)
- Images (with EXIF metadata and OCR)
- Audio files (EXIF metadata and speech transcription)
- HTML documents
- Text-based formats (CSV, JSON, XML)
- ZIP files (iterates over contents)
- EPub documents
- YouTube URLs (transcription)

All conversions preserve important document structure as Markdown, including:
- Headings
- Lists
- Tables
- Links
- And other formatting elements

## Features

- Image to Markdown conversion
- Secure API key handling via environment variables
- Endpoint authentication to prevent unauthorized access
- Temporary file management
- Error handling and logging
- Compatible with n8n and similar automation tools

## Prerequisites

- Python 3.x
- OpenAI API key
- FastAPI
- python-dotenv

## Environment Setup

Create a `.env` file in the root directory with two required keys:
```plaintext
OPENAI_API_KEY=your-openai-api-key
API_KEY=your-custom-endpoint-authentication-key
```

- `OPENAI_API_KEY`: Your OpenAI API key for GPT-4 Vision access
- `API_KEY`: A custom key you create to secure your endpoint from unauthorized access

## Security

The endpoint implements two layers of security:
1. OpenAI API key protection: Keeps your OpenAI credentials secure
2. Endpoint Authentication: Prevents unauthorized access to your conversion service

This dual authentication approach ensures:
- Protection against unauthorized usage and potential abuse
- Separation of concerns between OpenAI authentication and endpoint access
- Prevention of costly unauthorized OpenAI API usage

## Usage

### Deployment

This API can be deployed using [Leapcell](https://leapcell.io). Follow these steps for successful deployment:

1. Create a new Python project in Leapcell
2. Upload your code files:
   - `index.py`
   - `requirements.txt`
   - `.env` (configure in Leapcell environment variables)

3. Configure Environment Variables in Leapcell:
   - Go to your project settings
   - Add the following environment variables:
     ```
     OPENAI_API_KEY=your-openai-api-key
     API_KEY=your-custom-endpoint-authentication-key
     ```
3. Update the build command in Leapcell:
   ```bash
   pip install -r requirements.txt 
   ```
5. Update the start command in Leapcell:
   ```bash
   uvicorn index:app --host 0.0.0.0 --port 8080
   ```

5. Deploy your application

After deployment, your API will be available at your assigned Leapcell domain: `https://[your-project-id].leapcell.dev`

### Important Production Notes
- Ensure all dependencies are listed in `requirements.txt`
- The port must be set to 8080 for Leapcell
- Use the host 0.0.0.0 to allow external connections
- Don't commit your `.env` file; use Leapcell's environment variables instead

### Converting Files to Markdown

**Endpoint:** `POST /convert`

#### Using cURL
```bash
curl -X POST \
     -H "API_KEY: your-endpoint-api-key" \
     -F "file=@your-file.jpg" \
     "https://[your-project-id].leapcell.dev/convert"
```

#### Using Python Requests
```python
import requests

url = "https://[your-project-id].leapcell.dev/convert"
headers = {"API_KEY": "your-endpoint-api-key"}
files = {'file': open('your-file.jpg', 'rb')}

response = requests.post(url, headers=headers, files=files)
print(response.json()['result'])
```

#### Using n8n
1. Add an HTTP Request node
2. Configure as POST request
3. Set URL to your deployment endpoint
4. Add header with either:
   ```json
   {
     "api_key": "your-endpoint-api-key"
   }
   ```
   or
   ```json
   {
     "API_KEY": "your-endpoint-api-key"
   }
   ```
5. Add file content in Binary mode
6. Connect to your next workflow step

# Docker Usage

## Build the Docker image
```sh
docker build -t markitdown-endpoint .
```

## Run with Docker Compose (recommended)
1. Copy `.env.example` to `.env` and fill in your API keys.
2. Run:
```sh
docker-compose up --build
```

## Run with Docker only
```sh
docker run --env-file .env -p 8080:8080 markitdown-endpoint
```

- The API will be available at `http://localhost:8080/` by default.
- You can change the port by setting the `PORT` variable in your `.env` file and updating the Docker run/compose command accordingly.

## Environment Variables
- `API_KEY`: The key required for endpoint authentication.
- `OPENAI_API_KEY`: Your OpenAI API key.
- `PORT`: The port the API will run on (default 8080).