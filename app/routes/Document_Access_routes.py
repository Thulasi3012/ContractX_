
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.services.knowledge_graph_builder import KnowledgeGraphBuilder
from app.services.database_service import DatabaseService
from app.services.rag_service import RAGService

# Initialize services
kg_builder = KnowledgeGraphBuilder()
db_service = DatabaseService()
rag_service = RAGService()


router = APIRouter(
    prefix="/api/Database",
    tags=["Database Services"]
)

@router.get("Get_all/documents")
async def get_all_documents(skip: int = 0, limit: int = 100):
    """Retrieve all documents from database"""
    try:
        documents = db_service.get_all_documents(skip=skip, limit=limit)
        return JSONResponse(content={
            "documents": [
                {
                    "id": doc.id,
                    "document_name": doc.document_name,
                    "uploaded_on": doc.uploaded_on.isoformat(),
                    "document_type": doc.document_type,
                    "page_count": doc.page_count,
                    "buyer": doc.buyer,
                    "seller": doc.seller
                }
                for doc in documents
            ],
            "total": len(documents)
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.get("Get/documents/{document_id}")
async def get_document(document_id: str):
    """Retrieve specific document by ID"""
    try:
        document = db_service.get_document_by_id(document_id)
        if not document:
            return JSONResponse(
                content={"error": "Document not found"}, 
                status_code=404
            )
        
        return JSONResponse(content={
            "id": document.id,
            "document_name": document.document_name,
            "uploaded_on": document.uploaded_on.isoformat(),
            "summary": document.summary,
            "document_type": document.document_type,
            "buyer": document.buyer,
            "seller": document.seller,
            "parties": document.parties_json,
            "deadlines": document.deadlines,
            "alerts": document.alerts,
            "obligations": document.obligations,
            "page_count": document.page_count,
            "text_as_json": document.text_as_json
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.delete("Delete/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete document from all systems (DB, KG, RAG)"""
    try:
        print(f"\n[DELETE] Removing document: {document_id}")
        print("=" * 80)
        
        # Delete from database
        print("[1/3] Deleting from Database...")
        db_success = db_service.delete_document(document_id)
        if db_success:
            print(f"[OK] Database: Deleted successfully")
        else:
            print(f"[WARNING] Database: Document not found")
        
        # Delete from Knowledge Graph
        print("[2/3] Deleting from Knowledge Graph...")
        kg_success = kg_builder.delete_document(document_id)
        if kg_success:
            print(f"[OK] Knowledge Graph: Deleted successfully")
        else:
            print(f"[WARNING] Knowledge Graph: Document not found or deletion failed")
        
        # Delete from RAG
        print("[3/3] Deleting from RAG Vector Store...")
        rag_success = rag_service.delete_document(document_id)
        if rag_success:
            print(f"[OK] RAG: Deleted successfully")
        else:
            print(f"[WARNING] RAG: Document not found")
        
        print("=" * 80)
        
        # Return success if deleted from at least one system
        if db_success or kg_success or rag_success:
            return JSONResponse(content={
                "status": "success",
                "message": "Document deleted from all systems",
                "document_id": document_id,
                "deleted_from": {
                    "database": db_success,
                    "knowledge_graph": kg_success,
                    "rag": rag_success
                }
            }, status_code=200)
        else:
            return JSONResponse(
                content={
                    "status": "failed",
                    "error": "Document not found in any system",
                    "document_id": document_id
                }, 
                status_code=404
            )
            
    except Exception as e:
        print(f"[ERROR] Delete operation failed: {str(e)}")
        print("=" * 80)
        
        return JSONResponse(
            content={
                "error": str(e),
                "document_id": document_id
            }, 
            status_code=500
        )
