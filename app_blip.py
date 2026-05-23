# ---------------------------
# IMPORTS
# ---------------------------
import io
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from PIL import Image

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
    MarianMTModel,
    MarianTokenizer
)

# ---------------------------
# DEVICE
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# TRANSLATION FUNCTIONS
# ---------------------------
def load_translator(src="en", tgt="hi"):
    model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(DEVICE)
    return tokenizer, model


def translate(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(DEVICE)
    output = model.generate(**inputs)
    return tokenizer.decode(output[0], skip_special_tokens=True)


# ---------------------------
# LOAD MODELS
# ---------------------------
print("Loading BLIP caption model...")

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

caption_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    use_safetensors=True
).to(DEVICE)

caption_model.eval()

print("Caption model loaded")

print("Loading VQA model...")

vqa_model = BlipForQuestionAnswering.from_pretrained(
    "Salesforce/blip-vqa-base",
    use_safetensors=True
).to(DEVICE)

vqa_model.eval()

print("VQA model loaded")

print("Loading translators...")

tokenizer_hi, model_hi = load_translator("en", "hi")
tokenizer_ml, model_ml = load_translator("en", "ml")

print("Translators loaded")

# ---------------------------
# FASTAPI APP
# ---------------------------
app = FastAPI()

# ---------------------------
# SERVE FRONTEND
# ---------------------------
@app.get("/", response_class=HTMLResponse)
async def home():
    try:
        with open("frontend.html", "r", encoding="utf-8") as f:
            return f.read()
    except:
        return "<h2>frontend.html not found</h2>"


# ---------------------------
# CAPTION API
# ---------------------------
@app.post("/caption")
async def caption_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        inputs = processor(image, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            output = caption_model.generate(**inputs)

        caption_en = processor.decode(
            output[0],
            skip_special_tokens=True
        )

        caption_hi = translate(caption_en, tokenizer_hi, model_hi)
        caption_ml = translate(caption_en, tokenizer_ml, model_ml)

        return {
            "english": caption_en,
            "hindi": caption_hi,
            "malayalam": caption_ml
        }

    except Exception as e:
        print("CAPTION ERROR:", e)
        return {"error": str(e)}


# ---------------------------
# VQA API
# ---------------------------
@app.post("/vqa")
async def vqa(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        inputs = processor(
            image,
            question,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            output = vqa_model.generate(**inputs)

        answer = processor.decode(
            output[0],
            skip_special_tokens=True
        )

        return {"answer": answer}

    except Exception as e:
        print("VQA ERROR:", e)
        return {"error": str(e)}