from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from app.services.rag_service import RAGService


router = APIRouter(
    prefix="/api/RAG",
    tags=["RAG services"]
)

rag_service = RAGService()


@router.get("Get_stats/{document_id}")
async def get_rag_stats(document_id: str):
    """Get RAG indexing statistics for a document"""
    try:
        stats = rag_service.get_document_stats(document_id)
        return JSONResponse(content=stats, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.post("/search")
async def rag_search(
    query: str = Body(...),
    document_id: str = Body(None),
    n_results: int = Body(8)
):
    """Direct RAG search endpoint (without LLM)"""
    try:
        results = rag_service.search(
            query=query,
            document_id=document_id,
            n_results=n_results
        )
        return JSONResponse(content={"results": results, "count": len(results)})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.delete("/clear_rag")
async def clear_rag():
    """Clear entire RAG vector store (use with caution!)"""
    try:
        rag_service.clear_all()
        return JSONResponse(content={"status": "RAG store cleared successfully"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
