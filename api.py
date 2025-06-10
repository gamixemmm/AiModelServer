from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from main import initialize_model, generate_response, load_knowledge_base
import re

# Initialize FastAPI app
app = FastAPI(
    title="SAP AI Assistant API",
    description="API for interacting with the SAP AI Assistant",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def is_greeting(message: str) -> bool:
    """Check if the message is a greeting."""
    greetings = r'\b(hello|hi|hey|greetings|good morning|good afternoon|good evening)\b'
    return bool(re.search(greetings, message.lower()))

# Initialize model and knowledge base at startup
print("Initializing model and knowledge base...")
knowledge_base = load_knowledge_base()
model, tokenizer = initialize_model()
print("Initialization complete!")

# Store conversation histories for different sessions
conversation_histories = {}

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[tuple]] = []

class ChatResponse(BaseModel):
    response: str

@app.get("/")
async def root():
    return {"status": "ok", "message": "SAP AI Assistant API is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            user_input=request.message,
            knowledge_base=knowledge_base,
            conversation_history=request.conversation_history
        )
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False) 