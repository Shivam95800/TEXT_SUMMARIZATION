from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from pydantic import BaseModel
from groq import Groq
import os
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    print("Groq Client Ready!")
    yield

app = FastAPI(title="Text Summarizer App", description="Text Summarization using Groq", version="1.0", lifespan=lifespan)

templates = Jinja2Templates(directory=".")

class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text = re.sub(r"<.*?>", " ", text)
    text = text.strip()
    return text

def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_data(dialogue)
    
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "user",
                "content": f"Summarize the following text in 2-3 sentences. Only return the summary, nothing else:\n\n{dialogue}"
            }
        ],
        max_tokens=300
    )
    return response.choices[0].message.content

@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")
