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
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from pydantic import BaseModel
import anthropic
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY env variable se automatically lega
    yield

app = FastAPI(title="Text Summarizer App", description="Text Summarization using Claude", version="1.0", lifespan=lifespan)

templates = Jinja2Templates(directory=".")

class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip()
    return text

def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_data(dialogue)
    
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",  # Sabse fast & cheap model
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": f"Please summarize the following text in 2-3 sentences:\n\n{dialogue}"
            }
        ]
    )
    return message.content[0].text

@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")
