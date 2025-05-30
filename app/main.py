from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import replicate
import shutil
import os
import uuid

app = FastAPI()
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "message": None, "result_url": None})

@app.post("/upload/", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return templates.TemplateResponse("index.html", {"request": request, "message": "Error: Solo imágenes JPG o PNG permitidas.", "result_url": None})

    filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        model = replicate.models.get("xinntao/realesrgan")
        version = model.versions.get("v0.2.5")
        output_url = version.predict(img=open(file_path, "rb"), scale=2, api_token=REPLICATE_API_TOKEN)
    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "message": f"Error al procesar la imagen: {e}", "result_url": None})

    return templates.TemplateResponse("index.html", {"request": request, "message": "Imagen restaurada con éxito.", "result_url": output_url})
