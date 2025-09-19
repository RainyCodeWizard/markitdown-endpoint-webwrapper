import shutil
import base64
import logging
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Request, Query
from uuid import uuid4
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from markitdown import MarkItDown
from PIL import Image, ImageFile
import io
from mangum import Mangum
import boto3
import json
from enum import Enum

# Allow loading of truncated images and increase max pixel limit for large PDFs
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Set a reasonable limit (default is 178 million pixels = ~13k x 13k)
Image.MAX_IMAGE_PIXELS = 500_000_000  # Allow up to ~22k x 22k pixels

# Load environment variables
load_dotenv()

app = FastAPI()

# Model providers enum
class ModelProvider(str, Enum):
    AZURE_OPENAI = "azure_openai"
    AWS_BEDROCK = "aws_bedrock"

# Minimal logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("markitdown-endpoint")

@app.post("/")
async def convert_markdown(
    request: Request,
    api_key: str | None = Header(None, alias="API_KEY", description="API key for endpoint authentication"),
    file: UploadFile = File(None),
    enrich_pdf: bool = Query(False, description="If true and input is a PDF, include inline image descriptions"),
    include_images: bool = Query(True, description="When enrich_pdf=true for PDFs, include original images inline before their descriptions; if false, include only descriptions"),
    model_provider: ModelProvider = Query(ModelProvider.AWS_BEDROCK, description="AI model provider to use for image description")
):
    # Validate endpoint API key
    expected_api_key = os.getenv('API_KEY')
    if not expected_api_key:
        raise HTTPException(status_code=500, detail="Endpoint API key not configured on server")
    # Support multiple header conventions and proxy-safe names
    # Primary: explicit Header param using underscore (may be stripped by some proxies)
    provided_api_key = api_key
    if not provided_api_key:
        # Starlette lower-cases header names; prefer hyphenated variants that survive proxies
        headers = request.headers
        provided_api_key = (
            headers.get('x-api-key')
            or headers.get('api-key')
            or headers.get('api_key')
            or headers.get('x_api_key')
            or headers.get('api_key')
        )
        if not provided_api_key:
            auth_header = headers.get('authorization')
            if auth_header and auth_header.lower().startswith('bearer '):
                provided_api_key = auth_header[7:].strip()

    if provided_api_key != expected_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    # Initialize AI client based on provider
    if model_provider == ModelProvider.AZURE_OPENAI:
        # Get Azure OpenAI configuration from environment variables
        azure_openai_api_key = os.getenv('AZURE_OPENAI_API_KEY')
        azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        azure_openai_api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2025-04-01-preview')
        azure_openai_deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT')

        if not azure_openai_api_key:
            raise HTTPException(status_code=500, detail="AZURE_OPENAI_API_KEY not found in environment variables")
        if not azure_openai_endpoint:
            raise HTTPException(status_code=500, detail="AZURE_OPENAI_ENDPOINT not found in environment variables")
        if not azure_openai_deployment:
            raise HTTPException(status_code=500, detail="AZURE_OPENAI_DEPLOYMENT (deployment name) not found in environment variables")
        
        # Initialize Azure OpenAI client
        ai_client = AzureOpenAI(
            api_key=azure_openai_api_key,
            azure_endpoint=azure_openai_endpoint,
            api_version=azure_openai_api_version
        )
        model_name = azure_openai_deployment
        
    elif model_provider == ModelProvider.AWS_BEDROCK:
        # Get AWS Bedrock configuration from environment variables
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_region = os.getenv('AWS_REGION', 'us-east-1')
        bedrock_model_id = os.getenv('AWS_BEDROCK_MODEL_ID', 'anthropic.claude-3-5-sonnet-20240620-v1:0')
        
        if not aws_access_key_id or not aws_secret_access_key:
            raise HTTPException(status_code=500, detail="AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set for Bedrock")
        
        # Initialize Bedrock client
        ai_client = boto3.client(
            'bedrock-runtime',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region
        )
        model_name = bedrock_model_id
    
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model provider: {model_provider}")

    # Prepare temp folder
    hash = uuid4()
    folder_path = f"/tmp/{hash}"
    try:
        os.makedirs(folder_path, exist_ok=True)
    except OSError as e:
        logger.error("Failed to create temp directory in /tmp: %s", e)
        raise HTTPException(status_code=500, detail="Unable to create temporary storage")

    try:
        if file is not None:
            # Standard multipart/form-data upload
            # Ensure we have a valid filename
            filename = file.filename or f"upload_{hash}"
            # Remove any path separators for security
            filename = os.path.basename(filename)
            if not filename or filename == "." or filename == "..":
                filename = f"upload_{hash}"
            file_path = f"{folder_path}/{filename}"
            with open(file_path, "wb") as f_out:
                shutil.copyfileobj(file.file, f_out)
        else:
            # Raw binary upload
            # Try to get filename from headers, otherwise use a default
            content_type = request.headers.get("content-type", "application/octet-stream")
            filename = request.headers.get("x-filename", f"upload_{hash}")
            # Remove any path separators for security
            filename = os.path.basename(filename)
            if not filename or filename == "." or filename == "..":
                filename = f"upload_{hash}"
            file_path = f"{folder_path}/{filename}"
            body = await request.body()
            with open(file_path, "wb") as f_out:
                f_out.write(body)

        # AI client is already initialized above based on provider

        # If enrichment requested and input is a PDF, run enriched pipeline
        lower_name = os.path.basename(file_path).lower()
        request_id = str(hash)
        logger.info("[%s] Received request enrich_pdf=%s file=%s content_type=%s provider=%s", 
                   request_id, enrich_pdf, os.path.basename(file_path), request.headers.get("content-type"), model_provider.value)
        
        if model_provider == ModelProvider.AZURE_OPENAI:
            logger.info("[%s] Using Azure OpenAI endpoint=%s deployment=%s api_version=%s", 
                       request_id, azure_openai_endpoint, azure_openai_deployment, azure_openai_api_version)
        else:
            logger.info("[%s] Using AWS Bedrock model=%s region=%s", 
                       request_id, bedrock_model_id, aws_region)

        if enrich_pdf and (lower_name.endswith(".pdf") or request.headers.get("content-type", "").startswith("application/pdf")):
            logger.info("[%s] Running OPTIMIZED PDF conversion (image-only to LLM) include_images=%s", request_id, include_images)
            text = convert_pdf_to_markdown_optimized(
                file_path,
                ai_client,
                model_name,
                model_provider,
                request_id=request_id,
                include_images=include_images,
            )
        else:
            # Standard MarkItDown conversion
            if model_provider == ModelProvider.AZURE_OPENAI:
                # For Azure OpenAI, llm_model expects the deployment name
                md_instance = MarkItDown(llm_client=ai_client, llm_model=model_name)
                result = md_instance.convert(file_path)
                text = result.text_content
            else:
                # For Bedrock, MarkItDown may not support it directly, use basic conversion
                md_instance = MarkItDown()
                result = md_instance.convert(file_path)
                text = result.text_content

        return {"result": text}

    except Exception as e:
        logger.error("[error] Unexpected failure: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)

# AWS Lambda handler (via Mangum)
handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("index:app", host="0.0.0.0", port=port, reload=True)

# ----------------------------
# Optimized Enriched PDF conversion
# ----------------------------

def _describe_image_azure(client: AzureOpenAI, deployment: str, image_b64: str, image_mime: str, *, request_id: str, page_index: int) -> str:
    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "If the image contains any text, information, data, or content (including posters, signs, charts, tables, diagrams, forms, screenshots, documents, or any readable material), extract and transcribe ALL visible text and information exactly word-for-word. Output only the raw extracted content without any introductory phrases like 'this image shows' or 'the image contains'. For non-text content like charts or diagrams, provide the exact data, values, labels, and structural information present. If the image is purely decorative (logos, icons, backgrounds, dividers) with no meaningful information, reply exactly with SKIP and nothing else."},
                        {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{image_b64}"}},
                    ],
                }
            ],
            temperature=0.2,
            max_tokens=4000,
        )
        logger.info("[%s] Image description success for page %s", request_id, page_index + 1)
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.error("[%s] Image description failed for page %s: %s", request_id, page_index + 1, exc, exc_info=True)
        return ""


def _describe_image_bedrock(client, model_id: str, image_b64: str, image_mime: str, *, request_id: str, page_index: int) -> str:
    try:
        # Prepare the message for Claude 3.5 Sonnet
        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "If the image contains any text, information, data, or content (including posters, signs, charts, tables, diagrams, forms, screenshots, documents, or any readable material), extract and transcribe ALL visible text and information exactly word-for-word. Output only the raw extracted content without any introductory phrases like 'this image shows' or 'the image contains'. For non-text content like charts or diagrams, provide the exact data, values, labels, and structural information present. If the image is purely decorative (logos, icons, backgrounds, dividers) with no meaningful information, reply exactly with SKIP and nothing else."
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_mime,
                        "data": image_b64
                    }
                }
            ]
        }
        
        # Prepare the request body for Bedrock
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4000,
            "temperature": 0.2,
            "messages": [message]
        }
        
        # Invoke the model
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(body)
        )
        
        # Parse the response
        response_body = json.loads(response['body'].read())
        content = response_body.get('content', [])
        
        if content and len(content) > 0:
            text_content = content[0].get('text', '').strip()
            logger.info("[%s] Bedrock image description success for page %s", request_id, page_index + 1)
            return text_content
        else:
            logger.warning("[%s] Bedrock returned empty content for page %s", request_id, page_index + 1)
            return ""
            
    except Exception as exc:
        logger.error("[%s] Bedrock image description failed for page %s: %s", request_id, page_index + 1, exc, exc_info=True)
        return ""


def convert_pdf_to_markdown_optimized(pdf_path: str, client, model_name: str, model_provider: ModelProvider, *, request_id: str, include_images: bool = True) -> str:
    reader = PdfReader(pdf_path)
    markdown_output: list[str] = []

    for i, page in enumerate(reader.pages):
        page_num = i + 1
        logger.info("[%s] Processing Page %d", request_id, page_num)

        # Extract text from the page locally
        try:
            text = (page.extract_text() or "").strip()
            if text:
                markdown_output.append(text)
        except Exception as e:
            logger.warning("[%s] Could not extract text from page %d: %s", request_id, page_num, e)

        # Extract and describe only embedded images (robust: scan XObjects and only accept JPEG/JP2)
        images_info = _extract_images_from_page(page)
        if images_info:
            logger.info("[%s] Found %d extractable images on page %d", request_id, len(images_info), page_num)
            processed_images = 0
            skipped_images = 0
            
            for j, (image_mime, image_bytes) in enumerate(images_info):
                try:
                    logger.debug("[%s] Processing image %d/%d on page %d (%s, %d bytes)", 
                               request_id, j + 1, len(images_info), page_num, image_mime, len(image_bytes))
                    
                    # Validate and potentially resize image before processing
                    processed_bytes, processed_mime = _validate_and_resize_image_for_azure(
                        image_bytes, image_mime, request_id, j + 1, page_num
                    )
                    if not processed_bytes:
                        skipped_images += 1
                        continue
                    
                    image_b64 = base64.b64encode(processed_bytes).decode("utf-8")
                    # Update mime type to the processed format
                    image_mime = processed_mime
                    
                    # Call appropriate description function based on provider
                    if model_provider == ModelProvider.AZURE_OPENAI:
                        description = _describe_image_azure(
                            client, model_name, image_b64, image_mime, request_id=request_id, page_index=i
                        )
                    elif model_provider == ModelProvider.AWS_BEDROCK:
                        description = _describe_image_bedrock(
                            client, model_name, image_b64, image_mime, request_id=request_id, page_index=i
                        )
                    else:
                        logger.error("[%s] Unsupported model provider: %s", request_id, model_provider)
                        continue
                    
                    # Skip invalid images (no description due to Azure 400 or other issues)
                    if not description:
                        logger.warning("[%s] Skipping image %d on page %d due to invalid data/description", request_id, j + 1, page_num)
                        skipped_images += 1
                        continue
                        
                    # Skip images the model tagged as non-important
                    desc_trimmed = description.strip()
                    if desc_trimmed.upper() == "SKIP" or desc_trimmed.lower().startswith("skip"):
                        logger.info("[%s] AI marked image %d on page %d as non-important (SKIP)", request_id, j + 1, page_num)
                        skipped_images += 1
                        continue
                        
                    # Image successfully processed
                    processed_images += 1
                    logger.debug("[%s] Generated description for image %d on page %d (%d chars)", 
                               request_id, j + 1, page_num, len(description))
                    
                    if include_images:
                        markdown_output.append(
                            f"\n\n![Image {j+1} on Page {page_num}](data:{processed_mime};base64,{image_b64})\n\n{description}"
                        )
                    else:
                        markdown_output.append(
                            f"\n\n{description}"
                        )
                except Exception as e:
                    logger.error("[%s] Failed to process image %d on page %d: %s", request_id, j + 1, page_num, e, exc_info=True)
                    skipped_images += 1
                    
            logger.info("[%s] Page %d image processing complete: %d processed, %d skipped", 
                       request_id, page_num, processed_images, skipped_images)
        else:
            logger.debug("[%s] No extractable images found on page %d", request_id, page_num)

    return "\n\n".join(markdown_output).strip()


def _extract_text_per_page(pdf_path: str) -> list[tuple[int, str]]:
    reader = PdfReader(pdf_path)
    return [(idx + 1, (page.extract_text() or "").strip()) for idx, page in enumerate(reader.pages)]


def _extract_images_from_page(page) -> list[tuple[str, bytes]]:
    """Return list of (mime_type, image_bytes) for image XObjects we can pass directly.

    Only accept images whose filters yield file-encoded bytes (no re-encoding):
    - DCTDecode -> image/jpeg
    - JPXDecode -> image/jp2
    Skip others (e.g., FlateDecode, CCITTFaxDecode) to avoid complex wrapping/decoding.
    """
    results: list[tuple[str, bytes]] = []
    total_xobjects = 0
    image_xobjects = 0
    supported_images = 0
    unsupported_filters = set()
    
    try:
        resources = page.get("/Resources")
        if not resources:
            logger.debug("Page has no /Resources")
            return results
        xobjects = resources.get("/XObject")
        if not xobjects:
            logger.debug("Page has no /XObject resources")
            return results
            
        total_xobjects = len(xobjects)
        logger.debug(f"Found {total_xobjects} XObjects on page")
        
        for obj_name, obj in xobjects.items():
            try:
                xobj = obj.get_object()
                if xobj.get("/Subtype") != "/Image":
                    logger.debug(f"XObject {obj_name} is not an image (subtype: {xobj.get('/Subtype')})")
                    continue
                    
                image_xobjects += 1
                filters = xobj.get("/Filter")
                if isinstance(filters, list):
                    filter_names = [str(f) for f in filters]
                elif filters is not None:
                    filter_names = [str(filters)]
                else:
                    filter_names = []

                logger.debug(f"Image {obj_name} has filters: {filter_names}")

                # Try to extract and convert image data
                try:
                    data: bytes = xobj.get_data()
                    if not data:
                        logger.warning(f"Image {obj_name} has no data")
                        continue

                    # Get image dimensions for validation
                    width = xobj.get("/Width")
                    height = xobj.get("/Height")
                    logger.debug(f"Image {obj_name} dimensions: {width}x{height}")

                    mime_type: str | None = None
                    converted_data: bytes = data

                    # Handle different image encodings
                    if any("DCTDecode" in f for f in filter_names):
                        # JPEG - can use directly
                        mime_type = "image/jpeg"
                    elif any("JPXDecode" in f for f in filter_names):
                        # JPEG 2000 - can use directly
                        mime_type = "image/jp2"
                    elif any("FlateDecode" in f for f in filter_names):
                        # PNG or other compressed format - convert to JPEG
                        converted_data, mime_type = _convert_image_to_jpeg(data, width, height, "FlateDecode")
                        if not converted_data:
                            logger.debug(f"Failed to convert FlateDecode image {obj_name}")
                            continue
                    elif any("CCITTFaxDecode" in f for f in filter_names):
                        # TIFF fax format - convert to JPEG
                        converted_data, mime_type = _convert_image_to_jpeg(data, width, height, "CCITTFaxDecode")
                        if not converted_data:
                            logger.debug(f"Failed to convert CCITTFaxDecode image {obj_name}")
                            continue
                    else:
                        # Try generic conversion for other formats
                        try:
                            converted_data, mime_type = _convert_image_to_jpeg(data, width, height, "Generic")
                            if not converted_data:
                                # Track unsupported filters for logging
                                unsupported_filters.update(filter_names)
                                logger.debug(f"Skipping image {obj_name} - unsupported filters: {filter_names}")
                                continue
                        except Exception:
                            # Track unsupported filters for logging
                            unsupported_filters.update(filter_names)
                            logger.debug(f"Skipping image {obj_name} - unsupported filters: {filter_names}")
                            continue

                    supported_images += 1
                    logger.debug(f"Successfully extracted image {obj_name} ({len(converted_data)} bytes, {mime_type})")
                    results.append((mime_type, converted_data))

                except Exception as conversion_exc:
                    logger.debug(f"Failed to extract/convert image {obj_name}: {conversion_exc}")
                    continue
            except Exception as inner_exc:
                logger.warning("Failed extracting XObject image %s: %s", obj_name, inner_exc)
                
        # Summary logging
        if image_xobjects > 0:
            logger.info(f"Page summary: {total_xobjects} XObjects, {image_xobjects} images, {supported_images} supported, {image_xobjects - supported_images} skipped")
            if unsupported_filters:
                logger.info(f"Unsupported image filters found: {sorted(unsupported_filters)}")
        else:
            logger.debug("No image XObjects found on page")
            
    except Exception as exc:
        logger.warning("Failed scanning page XObjects for images: %s", exc)
    return results


def _convert_image_to_jpeg(image_data: bytes, width: int, height: int, format_type: str) -> tuple[bytes | None, str | None]:
    """Convert various image formats to JPEG for Azure OpenAI compatibility.
    
    Returns tuple of (converted_bytes, mime_type) or (None, None) if conversion fails.
    """
    try:
        if format_type == "FlateDecode":
            # Try to interpret as PNG-like data
            try:
                # First try to load as a direct image
                img = Image.open(io.BytesIO(image_data))
            except Exception:
                # If that fails, try to interpret as raw RGBA/RGB data
                if not width or not height:
                    return None, None
                
                # Try different bit depths and color modes
                for mode, bytes_per_pixel in [("RGBA", 4), ("RGB", 3), ("L", 1)]:
                    expected_size = width * height * bytes_per_pixel
                    if len(image_data) >= expected_size:
                        try:
                            img = Image.frombytes(mode, (width, height), image_data[:expected_size])
                            break
                        except Exception:
                            continue
                else:
                    logger.debug(f"Could not interpret FlateDecode image data (size: {len(image_data)}, expected for {width}x{height})")
                    return None, None
                    
        elif format_type == "CCITTFaxDecode":
            # TIFF fax format - create a black and white image
            if not width or not height:
                return None, None
            try:
                # Try to interpret as 1-bit data
                img = Image.frombytes("1", (width, height), image_data)
            except Exception:
                logger.debug(f"Could not interpret CCITTFaxDecode image data")
                return None, None
                
        else:  # Generic
            try:
                # Try to load as a standard image format
                img = Image.open(io.BytesIO(image_data))
            except Exception:
                return None, None

        # Convert to RGB if necessary (remove alpha channel, handle grayscale)
        if img.mode in ("RGBA", "P"):
            # Create white background for transparency
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
            img = background
        elif img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # Validate image dimensions
        if img.width < 10 or img.height < 10:
            logger.debug(f"Image too small: {img.width}x{img.height}")
            return None, None
            
        if img.width * img.height > 4096 * 4096:  # Reasonable size limit
            logger.debug(f"Image too large: {img.width}x{img.height}")
            return None, None

        # Convert to JPEG
        output_buffer = io.BytesIO()
        img.save(output_buffer, format="JPEG", quality=85, optimize=True)
        jpeg_data = output_buffer.getvalue()
        
        logger.debug(f"Successfully converted {format_type} image to JPEG ({len(jpeg_data)} bytes)")
        return jpeg_data, "image/jpeg"
        
    except Exception as e:
        logger.debug(f"Failed to convert {format_type} image: {e}")
        return None, None


def _validate_and_resize_image_for_azure(image_bytes: bytes, image_mime: str, request_id: str, image_num: int, page_num: int) -> tuple[bytes | None, str | None]:
    """Validate and potentially resize image before sending to Azure OpenAI.
    
    Returns tuple of (processed_image_bytes, mime_type) or (None, None) if invalid.
    """
    # Check file size limits (Azure OpenAI has a 20MB limit, but we'll be more conservative)
    MAX_SIZE_MB = 15
    if len(image_bytes) > MAX_SIZE_MB * 1024 * 1024:
        logger.warning("[%s] Skipping image %d on page %d - file too large (%d MB)", 
                      request_id, image_num, page_num, len(image_bytes) // (1024 * 1024))
        return None, None
    
    # Check minimum size (avoid tiny images that are likely artifacts)
    MIN_SIZE_BYTES = 100
    if len(image_bytes) < MIN_SIZE_BYTES:
        logger.debug("[%s] Skipping image %d on page %d - file too small (%d bytes)", 
                    request_id, image_num, page_num, len(image_bytes))
        return None, None
    
    # Process and potentially resize image using PIL
    try:
        # Load image
        img = Image.open(io.BytesIO(image_bytes))
        original_size = (img.width, img.height)
        
        # Check minimum dimensions
        if img.width < 10 or img.height < 10:
            logger.debug("[%s] Skipping image %d on page %d - dimensions too small (%dx%d)", 
                       request_id, image_num, page_num, img.width, img.height)
            return None, None
        
        # Check aspect ratio to avoid extremely stretched images
        aspect_ratio = max(img.width, img.height) / min(img.width, img.height)
        if aspect_ratio > 50:  # More lenient than before
            logger.debug("[%s] Skipping image %d on page %d - extreme aspect ratio (%.1f)", 
                       request_id, image_num, page_num, aspect_ratio)
            return None, None
        
        # Resize if dimensions are too large (common for vision APIs: 2048x2048 max)
        MAX_DIMENSION = 2048
        needs_resize = img.width > MAX_DIMENSION or img.height > MAX_DIMENSION
        
        if needs_resize:
            # Calculate new size maintaining aspect ratio
            scale_factor = min(MAX_DIMENSION / img.width, MAX_DIMENSION / img.height)
            new_width = int(img.width * scale_factor)
            new_height = int(img.height * scale_factor)
            
            logger.info("[%s] Resizing image %d on page %d from %dx%d to %dx%d (scale: %.2f)", 
                       request_id, image_num, page_num, img.width, img.height, 
                       new_width, new_height, scale_factor)
            
            # Resize with high quality
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            img = img_resized
        
        # Convert to RGB if necessary (remove alpha channel, handle grayscale)
        if img.mode in ("RGBA", "P"):
            # Create white background for transparency
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
            img = background
        elif img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        
        # Convert to JPEG for consistency
        output_buffer = io.BytesIO()
        img.save(output_buffer, format="JPEG", quality=85, optimize=True)
        processed_bytes = output_buffer.getvalue()
        processed_mime = "image/jpeg"
        
        if needs_resize:
            logger.debug("[%s] Image %d on page %d processed: %s -> %s, %d -> %d bytes", 
                        request_id, image_num, page_num, 
                        f"{original_size[0]}x{original_size[1]}", f"{img.width}x{img.height}",
                        len(image_bytes), len(processed_bytes))
        else:
            logger.debug("[%s] Image %d on page %d processed: %dx%d, %d bytes", 
                        request_id, image_num, page_num, img.width, img.height, len(processed_bytes))
        
        return processed_bytes, processed_mime
        
    except Exception as e:
        logger.warning("[%s] Skipping image %d on page %d - processing failed: %s", 
                      request_id, image_num, page_num, e)
        return None, None