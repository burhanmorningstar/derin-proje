# backend/app/main.py
#uvicorn app.main:app --host 0.0.0.0 --port 8000
#python -m venv .venv
#.\.venv\Scripts\Activate.ps1
import os
import json
import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException, status
import torch
from PIL import UnidentifiedImageError
from .model import load_model
from .predict import preprocess_image, model_inference  # aşağıda tanımlanacak

app = FastAPI(title="Handwritten Symbol Recognition")

# — Model ve map yükleme —
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pth")
model      = load_model(MODEL_PATH, DEVICE)

with open(os.path.join(os.path.dirname(__file__), "idx2symbol.json"), "r", encoding="utf-8") as f:
    raw = json.load(f)
idx2symbol = {int(k): v for k, v in raw.items()}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    # 1) Dosya var mı?
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file sent. Attach an image under the key 'file'."
        )
    # 2) İçerik tipi image/ ile başlıyor mu?
    if not (file.content_type.startswith("image/") or file.content_type == "application/octet-stream"):        
            raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{file.content_type}'. Only images allowed."
        )

    try:
        data = await file.read()
        # sadece tek bir çıktı: (symbol, confidence)
        symbol, conf = model_inference(model, DEVICE, data, idx2symbol)
    except UnidentifiedImageError:
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image file. Could not decode."
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error during prediction: {e}"
        )

    return {"symbol": symbol, "confidence": conf}


@app.post("/predict-debug")
async def predict_debug(file: UploadFile = File(...)):
    """
    Debug endpoint: returns
    - input tensor shape
    - top-5 (symbol, confidence)
    """
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file sent.")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only images allowed.")

    data = await file.read()
    try:
        # 1) Ön işlem
        tensor = preprocess_image(data).unsqueeze(0).to(DEVICE)  # [1,C,H,W]
        # 2) Inference
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)
        top5   = torch.topk(probs, k=5, dim=1)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image.")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {e}")

    # map indices→symbol
    results = []
    for prob, idx in zip(top5.values[0].tolist(), top5.indices[0].tolist()):
        sym = idx2symbol.get(idx, "?")
        results.append({"symbol": sym, "confidence": prob})

    return {
        "tensor_shape": list(tensor.shape),
        "top5": results
    }
