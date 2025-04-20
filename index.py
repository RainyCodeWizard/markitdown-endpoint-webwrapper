import shutil
from markitdown import MarkItDown
from fastapi import FastAPI, UploadFile, Header, HTTPException
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
    file: UploadFile,
    api_key: str | None = Header(None, alias="API_KEY", description="API key for endpoint authentication")
):
    # Validate endpoint API key
    expected_api_key = os.getenv('API_KEY')
    if not expected_api_key:
        raise HTTPException(
            status_code=500,
            detail="Endpoint API key not configured on server"
        )
    if api_key != expected_api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )

    # Get OpenAI API key from environment variables
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key not found in environment variables"
        )

    # Debug output
    print("File name:", file.filename)
    
    hash = uuid4()
    folder_path = f"/tmp/{hash}"

    try:
        shutil.os.makedirs(folder_path, exist_ok=True)

        file_path = f"{folder_path}/{file.filename}"
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        print(f"File saved to: {file_path}")
        
        # Initialize OpenAI client with API key from environment
        client = OpenAI(api_key=api_key)
        print("OpenAI client initialized successfully")
        
        # Create new MarkItDown instance for each request
        md_instance = MarkItDown(llm_client=client, llm_model="gpt-4o")
        print("MarkItDown initialized with OpenAI client")
        
        result = md_instance.convert(file_path)
        print("Conversion completed successfully")
        text = result.text_content

        return {"result": text}
    
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temporary files
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"Cleaned up temporary folder: {folder_path}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("index:app", host="0.0.0.0", port=8080, reload=True)