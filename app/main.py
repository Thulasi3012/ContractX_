from fastapi import FastAPI, UploadFile, File, Query, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import asyncio
import os

from datetime import datetime
from dotenv import load_dotenv
from app.config.config import Config

load_dotenv()

from app.services.pdf_processor import PDFProcessor
from app.services.text_analyzer import TextAnalyzer
from app.services.image_detector import ImageDetector
from app.services.knowledge_graph_builder import KnowledgeGraphBuilder
from app.services.database_service import DatabaseService
from app.services.rag_service import RAGService
from app.services.chatbot_service import ChatbotService
from app.utils.file_handler import FileHandler
from app.services.LLM_tracker import LLMUsageManager, LLMUsageTracker

app = FastAPI(
    title="ContractX - Complete Document Analysis with AI Chatbot", 
    version="4.0.0",
    description="Page-by-page analysis with RAG+KG powered chatbot"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
file_handler = FileHandler()
pdf_processor = PDFProcessor()
text_analyzer = TextAnalyzer()
image_detector = ImageDetector()
kg_builder = KnowledgeGraphBuilder()
db_service = DatabaseService()
rag_service = RAGService()
chatbot_service = ChatbotService()

print("=" * 80)
print("ContractX - Complete Document Analysis System v4.0")
print("=" * 80)
print("Features:")
print("  - Page-by-page processing")
print("  - Text analysis (sections, entities, clauses)")
print("  - Advanced multi-scale table detection")
print("  - Gemini table extraction with merged cell handling")
print("  - Image/visual detection (charts, graphs, diagrams)")
print("  - Knowledge Graph construction (Neo4j)")
print("  - Database storage (PostgreSQL/MySQL)")
print("  - RAG (Vector embeddings with ChromaDB)")
print("  - AI Chatbot (RAG + KG combined retrieval)")
print("=" * 80)

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


@app.post("/api/v1/extract-and-build-kg")
async def extract_and_build_kg(
    file: UploadFile = File(...),
    dpi: int = Query(350, ge=100, le=600, description="Image resolution for table detection")
):
    """Complete pipeline with LLM usage tracking"""
    request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("\n" + "=" * 80)
    print(f"NEW REQUEST (Full Pipeline): {request_id}")
    print(f"File: {file.filename}")
    print(f"DPI: {dpi}")
    print("=" * 80)
    
    # STEP 0: Initialize LLM Usage Manager
    
    llm_manager = LLMUsageManager()  # Document ID will be set after DB save
    
    try:
        # Step 1: Save and convert PDF
        print("\n[Step 1/7] Saving and converting PDF...")
        file_path = await file_handler.save_upload(file)
        pages = await pdf_processor.extract_pages(file_path, dpi=dpi)
        total_pages = len(pages)
        print(f"[OK] Extracted {total_pages} pages at {dpi} DPI")
        
        # Step 2: Process each page with tracking
        print(f"\n[Step 2/7] Processing {total_pages} pages...")
        print("=" * 80)
        
        all_page_results = []
        total_tables_found = 0
        total_visuals_found = 0
        
        # Get trackers for page processing
        text_tracker = llm_manager.get_tracker("text", Config.GEMINI_MODEL)
        table_tracker = llm_manager.get_tracker("table", Config.GEMINI_MODEL)
        image_tracker = llm_manager.get_tracker("image", Config.GEMINI_MODEL)
        
        for page_num, page_data in enumerate(pages, start=1):
            print(f"\n{'='*60}")
            print(f"Processing Page {page_num}/{total_pages}")
            print(f"{'='*60}")
            
            page_result = await process_single_page(
                page_number=page_num,
                page_data=page_data,
                total_pages=total_pages,
                pdf_path=file_path,
                text_tracker=text_tracker,      # PASS TRACKERS
                table_tracker=table_tracker,
                image_tracker=image_tracker
            )
            
            all_page_results.append(page_result)
            
            tables_on_page = len(page_result.get('tables', []))
            visuals_on_page = len(page_result.get('visuals', []))
            total_tables_found += tables_on_page
            total_visuals_found += visuals_on_page
            
            print(f"[OK] Page {page_num} complete:")
            print(f"  - Text sections: {page_result['text_analysis']['sections_count']}")
            print(f"  - Tables found: {tables_on_page}")
            print(f"  - Visuals found: {visuals_on_page}")
        
        # Step 3: Create final response
        print("\n" + "=" * 80)
        print("[Step 3/7] Consolidating results...")
        
        final_result = {
            "request_id": request_id,
            "filename": file.filename,
            "total_pages": total_pages,
            "pages": all_page_results,
            "summary": create_summary(all_page_results),
            "metadata": {
                "dpi": dpi,
                "extraction_method": "gemini-table-extractor",
                "text_model": "gemini-2.0-flash",
                "table_extraction_model": Config.GEMINI_MODEL,                
                "version": "4.0.0",
                "features": [
                    "Multi-scale table detection",
                    "Merged cell handling",
                    "Page-by-page processing",
                    "Image/visual detection",
                    "Knowledge graph construction",
                    "Database storage",
                    "RAG vector indexing",
                    "AI Chatbot ready",
                    "LLM usage tracking"  # NEW
                ]
            },
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }
        
        # Step 3.5: Generate overall summary
        print("\n[Step 3.5/7] Generating overall document summary...")
        overall_summary = db_service._generate_overall_summary(all_page_results)
        final_result['overall_summary'] = overall_summary
        print(f"[OK] Overall summary generated")
        
        # Step 4: Save to Database (Get document_id FIRST)
        print("\n" + "=" * 80)
        print("[Step 4/7] Saving to database...")
        document_id = None
        try:
            document_id = db_service.store_document(final_result)
            final_result['database'] = {
                "status": "success",
                "document_id": document_id,
                "message": "Document stored successfully"
            }
            print(f"[OK] Document saved with ID: {document_id}")
            
            # CRITICAL: Set document_id in LLM manager NOW
            llm_manager.set_document_id(document_id)
            
        except Exception as db_error:
            final_result['database'] = {
                "status": "failed",
                "error": str(db_error),
                "message": "Failed to store document in database"
            }
            print(f"[ERROR] Database storage failed: {str(db_error)}")
            raise
        
        # Step 5: Build Knowledge Graph
        print("\n" + "=" * 80)
        print("[Step 5/7] Building Knowledge Graph...")
        try:
            kg_result = kg_builder.build_graph(final_result, db_document_id=document_id)
            final_result['knowledge_graph'] = kg_result
            
            if kg_result.get('status') == 'success':
                print(f"[OK] Knowledge Graph built")
                print(f"  - Nodes: {kg_result.get('total_nodes', 0)}")
                print(f"  - Relationships: {kg_result.get('total_relationships', 0)}")
        except Exception as kg_error:
            final_result['knowledge_graph'] = {
                "status": "failed",
                "error": str(kg_error)
            }
            print(f"[ERROR] Knowledge Graph build failed: {str(kg_error)}")
            raise
        
        # Step 6: Index in RAG
        print("\n" + "=" * 80)
        print("[Step 6/7] Indexing in RAG vector store...")
        try:
            rag_result = rag_service.index_document(document_id, final_result)
            final_result['rag_indexing'] = rag_result
            print(f"[OK] RAG indexing complete: {rag_result['total_chunks']} chunks")
            
            rag_stats = rag_service.get_document_stats(document_id)
            print(f"  - Sections: {rag_stats.get('chunk_types', {}).get('section', 0)}")
            print(f"  - Tables: {rag_stats.get('chunk_types', {}).get('table', 0)}")
            print(f"  - Visuals: {rag_stats.get('chunk_types', {}).get('visual', 0)}")
        except Exception as rag_error:
            final_result['rag_indexing'] = {
                "status": "failed",
                "error": str(rag_error)
            }
            print(f"[ERROR] RAG indexing failed: {str(rag_error)}")
            raise
        
        # Step 7: Save LLM Usage to Database
        print("\n" + "=" * 80)
        print("[Step 7/7] Saving LLM usage statistics...")
        try:
            llm_manager.save_all_to_db()  # This will print summary too
            
            # Add LLM stats to response
            final_result['llm_usage'] = llm_manager.get_overall_stats()
            
        except Exception as llm_error:
            print(f"[ERROR] Failed to save LLM usage: {str(llm_error)}")
            # Don't fail the request, just log
        
        print("\n" + "=" * 80)
        print("✓ DOCUMENT PROCESSING COMPLETE!")
        print("=" * 80)
        print(f"Document ID: {document_id}")
        print(f"✓ Database: {final_result['database']['status']}")
        print(f"✓ Knowledge Graph: {final_result['knowledge_graph']['status']}")
        print(f"✓ RAG Indexing: {final_result['rag_indexing']['status']}")
        print(f"✓ LLM Usage Tracked: Yes")
        print("\n📝 You can now ask questions using:")
        print(f"   POST /api/v1/chatbot/{document_id}")
        print("=" * 80)
        
        # Cleanup
        await file_handler.cleanup(file_path)
        
        return JSONResponse(content=final_result, status_code=200)
        
    except Exception as e:
        print(f"\n[ERROR] Request {request_id} failed: {str(e)}")
        print("=" * 80)
        
        return JSONResponse(
            content={
                "error": str(e),
                "request_id": request_id,
                "status": "failed"
            },
            status_code=500
        )


async def process_single_page(
    page_number: int, 
    page_data: dict, 
    total_pages: int, 
    pdf_path: str = None,
    text_tracker: LLMUsageTracker = None,     
    table_tracker: LLMUsageTracker = None,     
    image_tracker: LLMUsageTracker = None  
) -> dict:
    """Process a single page with LLM tracking"""
    result = {
        "page_number": page_number,
        "text_analysis": {},
        "tables": [],
        "visuals": [],
        "image_detection": {}
    }
    
    # Text analysis with tracking
    result["text_analysis"] = await text_analyzer.analyze_text(
        page_number=page_number,
        page_data=page_data,
        usage_tracker=text_tracker  # PASS TRACKER
    )
    
    # Table detection with tracking (you'll need to update table extractor similarly)
    table_result = await pdf_processor.detect_tables_in_page(
        page_image=page_data['pil_image'],
        page_num=page_number,
        total_pages=total_pages,
        usage_tracker=table_tracker  # PASS TRACKER
    )
    result["tables"] = table_result.get("tables", [])
    
    # Image detection with tracking (you'll need to update image detector similarly)
    detected_visuals = await image_detector.detect_images(
        page_image=page_data['pil_image'],
        page_number=page_number,
        pdf_path=pdf_path,
        usage_tracker=image_tracker  # PASS TRACKER
    )
    
    result['visuals'] = detected_visuals
    result["image_detection"] = {
        "has_visuals": len(detected_visuals) > 0,
        "visual_count": len(detected_visuals),
        "types": list(set([v.get('type') for v in detected_visuals if v.get('type') and v.get('type') is not None])),
        "with_summaries": True,
        "visuals_data": detected_visuals
    }
    
    return result

def create_summary(page_results):
    """Create summary from all pages"""
    summary = {
        "total_sections": 0,
        "total_tables": 0,
        "pages_with_tables": 0,
        "total_visuals": 0,
        "pages_with_visuals": 0,
        "visual_types": {},
        "entities": {
            "buyer_name": None,
            "seller_name": None,
            "dates": [],
            "deadlines": [],
            "alerts": [],
            "obligations": [],
            "addresses": [],
            "contact_info": {}
        },
        "document_type": None
    }
    
    for page in page_results:
        summary['total_sections'] += page['text_analysis']['sections_count']
        
        table_count = len(page['tables'])
        if table_count > 0:
            summary['pages_with_tables'] += 1
            summary['total_tables'] += table_count
        
        visual_count = len(page['visuals'])
        if visual_count > 0:
            summary['pages_with_visuals'] += 1
            summary['total_visuals'] += visual_count
            
            for visual in page['visuals']:
                vtype = visual['type']
                summary['visual_types'][vtype] = summary['visual_types'].get(vtype, 0) + 1
        
        entities = page['text_analysis']['entities']
        if entities.get('buyer_name') and not summary['entities']['buyer_name']:
            summary['entities']['buyer_name'] = entities['buyer_name']
        if entities.get('seller_name') and not summary['entities']['seller_name']:
            summary['entities']['seller_name'] = entities['seller_name']
        
        summary['entities']['dates'].extend(entities.get('dates', []))
        summary['entities']['deadlines'].extend(entities.get('deadlines', []))
        summary['entities']['alerts'].extend(entities.get('alerts', []))
        summary['entities']['addresses'].extend(entities.get('addresses', []))
        
        if entities.get('obligations'):
            summary['entities']['obligations'].extend(entities.get('obligations', []))
        
        if entities.get('contact_info'):
            summary['entities']['contact_info'].update(entities.get('contact_info', {}))
    
    summary['entities']['dates'] = list(set(filter(None, summary['entities']['dates'])))
    summary['entities']['deadlines'] = list(set(filter(None, summary['entities']['deadlines'])))
    summary['entities']['alerts'] = list(set(filter(None, summary['entities']['alerts'])))
    summary['entities']['addresses'] = list(set(filter(None, summary['entities']['addresses'])))
    
    return summary


# ============================================================================
# CHATBOT ENDPOINTS - RAG + KG Combined
# ============================================================================

@app.post("/api/v1/chatbot/{document_id}")
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


@app.get("/api/v1/chatbot/{document_id}/info")
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


# ============================================================================
# DATABASE ENDPOINTS
# ============================================================================

@app.get("/api/v1/documents")
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


@app.get("/api/v1/documents/{document_id}")
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


@app.delete("/api/v1/documents/{document_id}")
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


# ============================================================================
# KNOWLEDGE GRAPH ENDPOINTS
# ============================================================================

@app.post("/api/v1/query-kg")
async def query_knowledge_graph(cypher_query: str = Query(..., description="Cypher query to execute")):
    """Query the knowledge graph using Cypher"""
    try:
        results = kg_builder.query_graph(cypher_query)
        return JSONResponse(content={"results": results, "count": len(results)})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.delete("/api/v1/clear-kg")
async def clear_knowledge_graph():
    """Clear the entire knowledge graph (use with caution!)"""
    try:
        kg_builder.clear_graph()
        return JSONResponse(content={"status": "Graph cleared successfully"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ============================================================================
# RAG ENDPOINTS
# ============================================================================

@app.get("/api/v1/rag/stats/{document_id}")
async def get_rag_stats(document_id: str):
    """Get RAG indexing statistics for a document"""
    try:
        stats = rag_service.get_document_stats(document_id)
        return JSONResponse(content=stats, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/api/v1/rag/search")
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


@app.delete("/api/v1/rag/clear")
async def clear_rag():
    """Clear entire RAG vector store (use with caution!)"""
    try:
        rag_service.clear_all()
        return JSONResponse(content={"status": "RAG store cleared successfully"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ============================================================================
# SYSTEM ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Check system health and model status"""
    return {
        "status": "healthy",
        "models": {
            "text_analysis": "gemini-2.0-flash-exp",
            "table_extraction": Config.GEMINI_MODEL,
            "embeddings": "text-embedding-004",
            "chatbot": "gemini-2.0-flash-exp"
        },
        "version": "4.0.0",
        "services": {
            "database": "connected",
            "knowledge_graph": "connected" if kg_builder.driver else "disconnected",
            "rag": "active",
            "chatbot": "ready"
        }
    }


@app.get("/")
async def root():
    return {
        "service": "ContractX Complete Document Analysis with AI Chatbot",
        "version": "4.0.0",
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


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)