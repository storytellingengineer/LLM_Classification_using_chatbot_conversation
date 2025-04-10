from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from .classifier import ConversationClassifier

app = FastAPI(
    title="Chatbot Conversation Classifier",
    description="API for classifying chatbot conversations based on sentiment and context",
    version="1.0.0"
)

class ConversationMessage(BaseModel):
    role: str
    content: str

class ClassificationRequest(BaseModel):
    conversation: List[ConversationMessage]

class ClassificationResponse(BaseModel):
    sentiment: str
    context: str
    confidence_score: float

classifier = ConversationClassifier()

@app.post("/classify", response_model=ClassificationResponse)
async def classify_conversation(request: ClassificationRequest):
    try:
        # Convert conversation to the format expected by the classifier
        conversation_text = "\n".join([
            f"{msg.role}: {msg.content}" for msg in request.conversation
        ])
        
        # Get classification results
        result = classifier.classify(conversation_text)
        
        return ClassificationResponse(
            sentiment=result["sentiment"],
            context=result["context"],
            confidence_score=result["confidence_score"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 