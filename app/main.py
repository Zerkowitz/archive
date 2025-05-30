from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import replicate
import shutil
import os

app = FastAPI()

UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

@app.get("/")
def read_root():
    return HTMLResponse("""
        <h2>Worldarchive MVP</h2>
        <form action='/upload/' method='post' enctype='multipart/form-data'>
            <input name='file' type='file'>
            <input type='submit'>
        </form>
    """)

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Llama a la IA en la nube (Replicate: Real-ESRGAN)
    output_url = replicate.run(
        "xinntao/realesrgan",
        input={"img": open(file_path, "rb"), "scale": 2},
        api_token=REPLICATE_API_TOKEN
    )

    return {"original": file.filename, "restored": output_url}
