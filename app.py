from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re 
from fastapi.templating import Jinja2Templates # UI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

# --- Model Variables Initialized to None ---
model = None
tokenizer = None

# --- Device Configuration ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

templates = Jinja2Templates(directory=".")

class DialogueInput(BaseModel):
    dialogue: str

# --- Lazy Loading Function ---
def load_ai_model():
    global model, tokenizer
    if model is None or tokenizer is None:
        print("Downloading & Loading Model... First request mein thoda time lagega.")
        tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
        model.to(device)
        print("Model Loaded Successfully!")

def clean_data(text):
    # HTML tags hatayenge, lekin Capital letters aur New Lines nahi
    text = re.sub(r"<.*?>", " ", text) 
    text = text.strip()
    return text

def summarize_dialogue(dialogue: str) -> str:
    # API hit hone par sabse pehle model load hoga (agar pehle se nahi hai toh)
    load_ai_model()
    
    dialogue = clean_data(dialogue)
    
    # AI ko instruction dena zaroori hai
    input_text = "summarize: " + dialogue

 # Baki upar ka code same rahega...

    # Tokenize
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
    # -----------------------------
    
    # Decode the output
    summary = tokenizer.decode(targets[0], skip_special_tokens=True)
    return summary

# API endpoints
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")
