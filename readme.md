# MarkItDown API Endpoint

FastAPI server providing a Wrapper for Microsoft MarkItDown

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file with the required variables for your chosen AI provider:

### For Azure OpenAI

```
API_KEY=your_api_key_here
AZURE_OPENAI_API_KEY=your_azure_openai_key_here
AZURE_OPENAI_ENDPOINT=https://your-azure-openai-resource.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o-deployment-name
```

### For AWS Bedrock (Claude 3.5 Sonnet) - Default

```
API_KEY=your_api_key_here
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_REGION=us-east-1
AWS_BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20240620-v1:0
```

**Note:** When using AWS Bedrock, ensure that:

1. Your AWS credentials have permissions to access Amazon Bedrock
2. You have requested access to Claude 3.5 Sonnet model in your AWS region
3. The Claude 3.5 Sonnet model is available in your chosen AWS region
4. `AWS_BEDROCK_MODEL_ID` is optional and defaults to `anthropic.claude-3-5-sonnet-20240620-v1:0`

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

Remember to configure your environment variables (`API_KEY`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_DEPLOYMENT`) in your chosen platform's dashboard.

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

### Enriched PDF Conversion (inline image descriptions)

If you pass the query parameter `enrich_pdf=true` and upload a PDF, the service will:

- Extract the original text per page
- Detect embedded images on each page and generate AI descriptions for images that contain substantive content using your chosen AI provider
- By default, embed each qualifying image as a data URL and place the generated description inline after it
- Control image embedding via the `include_images` query parameter:
  - `include_images=true` (default): include original images + their descriptions
  - `include_images=false`: include only the descriptions without embedding images
- Choose your AI provider via the `model_provider` query parameter:
  - `model_provider=aws_bedrock` (default): use AWS Bedrock with Claude 3.5 Sonnet
  - `model_provider=azure_openai`: use Azure OpenAI (requires GPT-4o or similar multimodal deployment)

This yields a Markdown document containing the original text plus image descriptions, useful for RAG pipelines.

Example cURL (enriched with images, default):

```bash
curl -X POST \
     -H "API_KEY: your-endpoint-api-key" \
     -F "file=@your-file.pdf" \
     "http://localhost:8080/?enrich_pdf=true&include_images=true"

Example cURL (enriched, descriptions only – no images embedded):

```bash
curl -X POST \
     -H "API_KEY: your-endpoint-api-key" \
     -F "file=@your-file.pdf" \
     "http://localhost:8080/?enrich_pdf=true&include_images=false"
```

Example cURL (using AWS Bedrock with Claude 3.5 Sonnet):

```bash
curl -X POST \
     -H "API_KEY: your-endpoint-api-key" \
     -F "file=@your-file.pdf" \
     "http://localhost:8080/?enrich_pdf=true&model_provider=aws_bedrock"
```

Notes:

- When using Azure OpenAI: Requires a multimodal-capable deployment (e.g., GPT-4o)
- When using AWS Bedrock: Requires access to Claude 3.5 Sonnet model in your AWS region
- For local development, `pdf2image` depends on Poppler being installed on your machine
  - macOS: `brew install poppler`
  - Linux (Debian/Ubuntu): `sudo apt-get install poppler-utils`
  - Windows: install Poppler from the official builds

## Prerequisites

- Python 3.x
- Azure OpenAI resource with a model deployment (e.g., GPT-4o)
- FastAPI
- python-dotenv

## Environment Setup

Create a `.env` file in the root directory with required keys:

```plaintext
API_KEY=your-custom-endpoint-authentication-key
AZURE_OPENAI_API_KEY=your-azure-openai-key
AZURE_OPENAI_ENDPOINT=https://your-azure-openai-resource.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o-deployment-name
```

- `API_KEY`: A custom key you create to secure your endpoint from unauthorized access
- `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI endpoint URL
- `AZURE_OPENAI_API_VERSION`: Azure OpenAI API version (default `2024-02-15-preview`)
- `AZURE_OPENAI_DEPLOYMENT`: The deployment name of your model (e.g., `gpt-4o` deployment)

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
- For enriched PDF conversion locally, ensure Poppler is installed (Docker image includes `poppler-utils` already)
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
- `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key.
- `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI endpoint URL.
- `AZURE_OPENAI_API_VERSION`: Azure OpenAI API version.
- `AZURE_OPENAI_DEPLOYMENT`: Azure OpenAI deployment name for the model to use.
- `PORT`: The port the API will run on (default 8080).
