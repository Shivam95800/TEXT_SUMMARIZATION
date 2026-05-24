from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re 
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

# --- Model Variables ---
model = None
tokenizer = None

# --- Device Configuration ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# --- Model Load Function ---
def load_ai_model():
    global model, tokenizer
    if model is None or tokenizer is None:
        print("Loading Model...")
        tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False, cache_dir="/tmp/model_cache")
        model = T5ForConditionalGeneration.from_pretrained("t5-small", cache_dir="/tmp/model_cache")
        model.to(device)
        print("Model Loaded Successfully!")

# --- Startup: Server start hote hi model load hoga ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_ai_model()
    yield

# --- App ---
app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0", lifespan=lifespan)

templates = Jinja2Templates(directory=".")

class DialogueInput(BaseModel):
    dialogue: str

# --- Clean Text ---
def clean_data(text):
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip()
    return text

# --- Summarize Logic ---
def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_data(dialogue)
    input_text = "summarize: " + dialogue

    inputs = tokenizer(
        input_text,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        targets = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            early_stopping=True
        )

    summary = tokenizer.decode(targets[0], skip_special_tokens=True)
    return summary

# --- API Endpoints ---
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")
