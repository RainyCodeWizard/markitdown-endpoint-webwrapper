import shutil
from markitdown import MarkItDown
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Request
from uuid import uuid4
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize MarkItDown without OpenAI client
md = MarkItDown()

app = FastAPI()

@app.post("/")
async def convert_markdown(
    request: Request,
    api_key: str | None = Header(None, alias="API_KEY", description="API key for endpoint authentication"),
    file: UploadFile = File(None)
):
    # Validate endpoint API key
    expected_api_key = os.getenv('API_KEY')
    if not expected_api_key:
        raise HTTPException(status_code=500, detail="Endpoint API key not configured on server")
    if api_key != expected_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    # Get OpenAI API key from environment variables
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not found in environment variables")

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

        # Initialize OpenAI client
        client = OpenAI(api_key=openai_api_key)
        md_instance = MarkItDown(llm_client=client, llm_model="gpt-4o")
        result = md_instance.convert(file_path)
        text = result.text_content

        return {"result": text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("index:app", host="0.0.0.0", port=port, reload=True)