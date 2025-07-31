from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

from model import predict_image

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Image Classifier API is running"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    label = predict_image(image)
    return {"prediction": label}