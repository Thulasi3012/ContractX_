from fastapi import APIRouter
from fastapi import UploadFile, File, Query
from fastapi.responses import JSONResponse

from app.services.pdf_processor import PDFProcessor
from app.services.text_analyzer import TextAnalyzer
from app.services.image_detector import ImageDetector
from app.services.database_service import DatabaseService
from app.utils.file_handler import FileHandler
from app.services.LLM_tracker import LLMUsageManager, LLMUsageTracker
from app.services.knowledge_graph_builder import KnowledgeGraphBuilder
from app.services.rag_service import RAGService
from datetime import datetime
from app.config.config import Config

router = APIRouter(
    prefix="/api/documents",
    tags=["Document Extraction"]
)

file_handler = FileHandler()
pdf_processor = PDFProcessor()
text_analyzer = TextAnalyzer()
image_detector = ImageDetector()
db_service = DatabaseService()
kg_builder = KnowledgeGraphBuilder()
rag_service = RAGService()
llm_manager = LLMUsageManager()


@router.post("/Document_Extraction")
async def extract_and_build_kg(
    file: UploadFile = File(...),
    dpi: int = Query(400, ge=100, le=600, description="Image resolution for table detection")
):
    """Complete pipeline with LLM usage tracking"""
    request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("\n" + "=" * 80)
    print(f"NEW REQUEST (Full Pipeline): {request_id}")
    print(f"File: {file.filename}")
    print(f"DPI: {dpi}")
    print("=" * 80)
    
    # STEP 0: Initialize LLM Usage Manager
    
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
        # 🔥 BUILD PURE DOCUMENT CONTENT (TEXT + TABLES + VISUALS ONLY)
        document_content = {
            "filename": file.filename,
            "total_pages": total_pages,
            "pages": []
        }

        for page in all_page_results:
            document_content["pages"].append({
                "page_number": page["page_number"],
                "text": {
                    "sections": page["text_analysis"].get("sections", []),
                    "summary": page["text_analysis"].get("summary", ""),
                    "entities": page["text_analysis"].get("entities", {}),
                    "sections_count": page["text_analysis"].get("sections_count", 0)
                },
                "tables": page.get("tables", []),
                "visuals": page.get("visuals", [])
            })

        document_id = None
        try:
            document_id = db_service.store_document({
                "filename": file.filename,
                "total_pages": total_pages,
                "pages": all_page_results,
                "document_content": document_content
            })
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
