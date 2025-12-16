from fastapi import APIRouter
from app.services.knowledge_graph_builder import KnowledgeGraphBuilder
from app.config.config import Config

kg_builder = KnowledgeGraphBuilder()


router = APIRouter(
    prefix="/api/System_Health",
    tags=["System Performance"]
)

@router.get("/health")
async def health_check():
    """Check system health and model status"""
    return {
        "status": "healthy",
        "models": {
            "text_analysis": Config.GEMINI_MODEL,
            "table_extraction": Config.GEMINI_MODEL,
            "embeddings": "text-embedding-004",
            "chatbot": Config.GEMINI_MODEL
        },
        "version": "1.0.0",
        "services": {
            "database": "connected",
            "knowledge_graph": "connected" if kg_builder.driver else "disconnected",
            "rag": "active",
            "chatbot": "ready"
        }
    }


@router.get("/")
async def root():
    return {
        "service": "ContractX Complete Document Analysis with AI Chatbot",
        "version": "1.0.0",
        "endpoints": {
            "extraction": {
                "extract": "/api/v1/extract-document",
                "extract_full": "/api/v1/extract-and-build-kg"
            },
            "chatbot": {
                "query": "/api/v1/chatbot/{document_id}",
                "info": "/api/v1/chatbot/{document_id}/info"
            },
            "documents": {
                "list": "/api/v1/documents",
                "get": "/api/v1/documents/{document_id}",
                "delete": "/api/v1/documents/{document_id}"
            },
            "knowledge_graph": {
                "query": "/api/v1/query-kg",
                "clear": "/api/v1/clear-kg"
            },
            "rag": {
                "stats": "/api/v1/rag/stats/{document_id}",
                "search": "/api/v1/rag/search",
                "clear": "/api/v1/rag/clear"
            },
            "system": {
                "health": "/health",
                "docs": "/docs"
            }
        },
        "features": [
            "Document extraction (text, tables, visuals)",
            "Database storage with UUID",
            "Knowledge Graph (Neo4j)",
            "RAG vector indexing (ChromaDB)",
            "AI Chatbot (RAG + KG combined)",
            "Multi-document support",
            "Semantic search",
            "Entity extraction",
            "Automatic retry logic"
        ],
        "usage": {
            "1_upload": "POST /api/v1/extract-and-build-kg (with PDF file)",
            "2_chat": "POST /api/v1/chatbot/{document_id} with question",
            "3_example": {
                "question": "What is the payment term?",
                "include_rag": True,
                "include_kg": True
            }
        }
    }
