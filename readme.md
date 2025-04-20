# MarkItDown Endpoint

A FastAPI service that converts various file types to markdown text using OpenAI's GPT-4 Vision model and Microsoft's MarkItDown package. This service provides a simple API endpoint that can be integrated with no-code automation tools like n8n, making it perfect for building RAG (Retrieval Augmented Generation) systems in a no-code environment.

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

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install fastapi uvicorn python-dotenv openai markitdown
```

## Running the Server

You can run the server in two ways:

### Option 1: Using Python
```bash
python index.py
```

### Option 2: Using Uvicorn directly
```bash
uvicorn index:app --reload
```

The server will start on `http://localhost:8000`

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

## Deployment

This API is designed to run on [Leapcell](https://leapcell.io). Follow these steps for deployment:

1. Create a new Python project in Leapcell
2. Upload your code files:
   - `index.py`
   - `requirements.txt`

3. Configure Environment Variables in Leapcell:
   - Go to your project settings
   - Add the following environment variables:
     ```
     OPENAI_API_KEY=your-openai-api-key
     API_KEY=your-custom-endpoint-authentication-key
     ```

4. Deploy your application

Your API will be available at your assigned Leapcell domain: `https://[your-project-id].leapcell.dev`

### Important Notes
- Leapcell handles all server infrastructure
- No need for local server configuration
- Environment variables are managed in Leapcell dashboard

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

4. Update the start command in Leapcell:
   ```bash
   pip install -r requirements.txt && uvicorn index:app --host 0.0.0.0 --port 8080
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