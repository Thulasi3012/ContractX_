from fastapi import APIRouter
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from app.services.chatbot_service import ChatbotService


router = APIRouter(
    prefix="/api/ChatAgent",
    tags=["Chat Bot Services"]
)

# Pydantic models for request validation
class ChatRequest(BaseModel):
    question: str
    include_rag: bool = True
    include_kg: bool = True
    n_results: int = 8

class ChatResponse(BaseModel):
    document_id: str
    question: str
    answer: str
    sources: dict
    context_used: dict
    confidence: str

chatbot_service = ChatbotService()


@router.post("chatbot_query/{document_id}")
async def chatbot_query(document_id: str, request: ChatRequest):
    """
    AI Chatbot endpoint - Ask questions about a specific document
    
    Combines:
    - RAG (vector similarity search)
    - Knowledge Graph (structured data retrieval)
    - Gemini LLM (answer generation)
    
    Example:
    ```
    POST /api/v1/chatbot/abcd-1234-xyz-987
    {
        "question": "What is the payment term?",
        "include_rag": true,
        "include_kg": true,
        "n_results": 8
    }
    ```
    """
    try:
        result = chatbot_service.chat(
            document_id=document_id,
            question=request.question,
            include_rag=request.include_rag,
            include_kg=request.include_kg,
            n_rag_results=request.n_results
        )
        
        return JSONResponse(content=result, status_code=200)
    
    except Exception as e:
        return JSONResponse(
            content={
                "error": str(e),
                "document_id": document_id,
                "question": request.question
            },
            status_code=500
        )

@router.get("get_chatbot/{document_id}/info")
async def get_chatbot_info(document_id: str):
    """
    Get document information for chatbot context
    Shows what data is available in RAG and KG
    """
    try:
        info = chatbot_service.get_conversation_summary(document_id)
        return JSONResponse(content=info, status_code=200)
    except Exception as e:
        return JSONResponse(
            content={"error": str(e), "document_id": document_id},
            status_code=500
        )
