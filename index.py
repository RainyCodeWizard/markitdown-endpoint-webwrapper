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

# Load environment variables
load_dotenv()

app = FastAPI()

# Minimal logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("markitdown-endpoint")

@app.post("/")
async def convert_markdown(
    request: Request,
    api_key: str | None = Header(None, alias="API_KEY", description="API key for endpoint authentication"),
    file: UploadFile = File(None),
    enrich: bool = Query(False, description="If true and input is a PDF, include inline image descriptions")
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

    # Prepare temp folder
    hash = uuid4()
    folder_path = f"/tmp/{hash}"
    os.makedirs(folder_path, exist_ok=True)

    try:
        if file is not None:
            # Standard multipart/form-data upload
            file_path = f"{folder_path}/{file.filename}"
            with open(file_path, "wb") as f_out:
                shutil.copyfileobj(file.file, f_out)
        else:
            # Raw binary upload
            # Try to get filename from headers, otherwise use a default
            content_type = request.headers.get("content-type", "application/octet-stream")
            filename = request.headers.get("x-filename", f"upload_{hash}")
            file_path = f"{folder_path}/{filename}"
            body = await request.body()
            with open(file_path, "wb") as f_out:
                f_out.write(body)

        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            api_key=azure_openai_api_key,
            azure_endpoint=azure_openai_endpoint,
            api_version=azure_openai_api_version
        )

        # If enrichment requested and input is a PDF, run enriched pipeline
        lower_name = os.path.basename(file_path).lower()
        request_id = str(hash)
        logger.info("[%s] Received request enrich=%s file=%s content_type=%s", request_id, enrich, os.path.basename(file_path), request.headers.get("content-type"))
        logger.info("[%s] Using Azure OpenAI endpoint=%s deployment=%s api_version=%s", request_id, azure_openai_endpoint, azure_openai_deployment, azure_openai_api_version)

        if enrich and (lower_name.endswith(".pdf") or request.headers.get("content-type", "").startswith("application/pdf")):
            logger.info("[%s] Running OPTIMIZED PDF conversion (image-only to LLM)", request_id)
            text = convert_pdf_to_markdown_optimized(file_path, client, azure_openai_deployment, request_id=request_id)
        else:
            # For Azure OpenAI, llm_model expects the deployment name
            md_instance = MarkItDown(llm_client=client, llm_model=azure_openai_deployment)
            result = md_instance.convert(file_path)
            text = result.text_content

        return {"result": text}

    except Exception as e:
        logger.error("[error] Unexpected failure: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)

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
                        {"type": "text", "text": "Only if the image contains substantive, document-relevant information (e.g., diagrams, charts, tables, data/UI screenshots with context, photos with meaningful content, or any readable text), provide a concise markdown paragraph describing or transcribing it. If the image is decorative or not important (e.g., logos, icons, emojis, watermarks, backgrounds, page furniture, or dividers), reply exactly with SKIP and nothing else."},
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


def convert_pdf_to_markdown_optimized(pdf_path: str, client: AzureOpenAI, deployment: str, *, request_id: str) -> str:
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
            logger.info("[%s] Found %d images on page %d", request_id, len(images_info), page_num)
            for j, (image_mime, image_bytes) in enumerate(images_info):
                try:
                    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                    description = _describe_image_azure(
                        client, deployment, image_b64, image_mime, request_id=request_id, page_index=i
                    )
                    # Skip invalid images (no description due to Azure 400 or other issues)
                    if not description:
                        logger.warning("[%s] Skipping image %d on page %d due to invalid data/description", request_id, j + 1, page_num)
                        continue
                    # Skip images the model tagged as non-important
                    desc_trimmed = description.strip()
                    if desc_trimmed.upper() == "SKIP" or desc_trimmed.lower().startswith("skip"):
                        logger.info("[%s] Skipping non-important image %d on page %d", request_id, j + 1, page_num)
                        continue
                    markdown_output.append(
                        f"\n\n![Image {j+1} on Page {page_num}](data:{image_mime};base64,{image_b64})\n\n{description}"
                    )
                except Exception as e:
                    logger.error("[%s] Failed to process image %d on page %d: %s", request_id, j + 1, page_num, e)

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
    try:
        resources = page.get("/Resources")
        if not resources:
            return results
        xobjects = resources.get("/XObject")
        if not xobjects:
            return results
        for _, obj in xobjects.items():
            try:
                xobj = obj.get_object()
                if xobj.get("/Subtype") != "/Image":
                    continue
                filters = xobj.get("/Filter")
                if isinstance(filters, list):
                    filter_names = [str(f) for f in filters]
                elif filters is not None:
                    filter_names = [str(filters)]
                else:
                    filter_names = []

                mime_type: str | None = None
                if any("DCTDecode" in f for f in filter_names):
                    mime_type = "image/jpeg"
                elif any("JPXDecode" in f for f in filter_names):
                    mime_type = "image/jp2"
                else:
                    # Unsupported without re-encoding
                    continue

                data: bytes = xobj.get_data()
                if data:
                    results.append((mime_type, data))
            except Exception as inner_exc:
                logging.warning("Failed extracting one XObject image: %s", inner_exc)
    except Exception as exc:
        logging.warning("Failed scanning page XObjects for images: %s", exc)
    return results