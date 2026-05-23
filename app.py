# app.py

import io
import json
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image

from src.model import CaptionDecoder
from src.extract_features_single import extract_single_image_features
from src.generate_utils import topk_decode

# --------------------------------------------------
# App setup
# --------------------------------------------------
app = FastAPI()     #Creates API server.

app.add_middleware(         #Allows any frontend (React, HTML, etc.) to call this API.
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"     #Uses GPU if available Otherwise CPU

# --------------------------------------------------
# Load vocab
# --------------------------------------------------
with open("data/coco_subset/captions/word2idx.json", "r") as f:
    word2idx = json.load(f)

with open("data/coco_subset/captions/idx2word.json", "r") as f:
    idx2word = json.load(f)

# --------------------------------------------------
# Load trained model ONCE
# --------------------------------------------------
model = CaptionDecoder(    #Load trained model
    vocab_size=len(word2idx),
    pad_idx=word2idx["<pad>"]
).to(DEVICE)

model.load_state_dict(          #Loads trained weights from file
    torch.load(
        "checkpoints_coco_20k/model_coco_subset.pth",
        map_location=DEVICE,
        weights_only=True,
    )
)

model.eval()        #Evaluation mode:-disables dropout,disables training behavior

print("✅ COCO model loaded successfully")

# --------------------------------------------------
# Serve Frontend
# --------------------------------------------------
@app.get("/", response_class=HTMLResponse)      #Root endpoint (frontend)
async def serve_frontend():
    with open("frontend.html", "r", encoding="utf-8") as f:
        return f.read()

# --------------------------------------------------
# Caption Endpoint
# --------------------------------------------------
@app.post("/caption")                                               #input image
async def caption_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()         #read image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Extract real image features
        features, boxes = extract_single_image_features(image, device=DEVICE)  #Feature extraction:-converts image → numerical features
        features = features.unsqueeze(0).to(DEVICE)
        boxes = boxes.unsqueeze(0).to(DEVICE)

        # Generate caption
        caption = topk_decode(  #Uses decoding strategy (top-k sampling)
            model,
            features,
            word2idx,
            idx2word,
            k=7, #At each step → choose from top 7 probable words
        )

        return {"caption": caption}

    except Exception as e:
        print("🔥 ERROR:", e)
        return {"error": str(e)}
    
"""   User uploads image
        ↓
PIL loads image
        ↓
Feature extractor (CNN)
        ↓
Feature vectors + boxes
        ↓
CaptionDecoder (LSTM/Attention)
        ↓
top-k decoding
        ↓
Text caption###"""