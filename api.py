from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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
print("Loading model and knowledge base...")
model, tokenizer = initialize_model()
knowledge_base = load_knowledge_base()

# Store conversation histories for different sessions
conversation_histories = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    response: str
    session_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Get or create conversation history for this session
        if request.session_id not in conversation_histories:
            conversation_histories[request.session_id] = []
        
        # Get conversation history
        conversation_history = conversation_histories[request.session_id]
        
        # Check if this is a greeting
        if is_greeting(request.message):
            # Add special instruction for greeting
            user_input = f"{request.message} [This is a greeting, respond warmly and introduce yourself as an SAP AI Assistant]"
        else:
            user_input = request.message
        
        # Generate response
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            user_input=user_input,
            knowledge_base=knowledge_base,
            conversation_history=conversation_history
        )
        
        # Update conversation history
        conversation_history.append((request.message, response))
        if len(conversation_history) > 5:
            conversation_history.pop(0)
        
        return ChatResponse(
            response=response,
            session_id=request.session_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False) 