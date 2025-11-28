from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from app.services.pdf_processor import PDFProcessor
from app.services.text_analyzer import TextAnalyzer
from app.services.advanced_table_detector import AdvancedTableDetector
from app.services.gemini_table_extractor import GeminiTableExtractor
from app.services.image_detector import ImageDetector
from app.services.knowledge_graph_builder import KnowledgeGraphBuilder
from app.utils.file_handler import FileHandler

app = FastAPI(
    title="ContractX - Complete Document Analysis", 
    version="3.0.0",
    description="Page-by-page analysis with advanced table detection"
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
table_detector = AdvancedTableDetector()
table_extractor = GeminiTableExtractor()
image_detector = ImageDetector()
kg_builder = KnowledgeGraphBuilder()

print("=" * 80)
print("ContractX - Complete Document Analysis System v3.0")
print("=" * 80)
print("Features:")
print("  - Page-by-page processing")
print("  - Text analysis (sections, entities, clauses)")
print("  - Advanced multi-scale table detection")
print("  - Gemini table extraction with merged cell handling")
print("  - Image/visual detection (charts, graphs, diagrams)")
print("  - Knowledge Graph construction (Neo4j)")
print("=" * 80)

@app.post("/api/v1/extract-and-build-kg")
async def extract_and_build_kg(
    file: UploadFile = File(...),
    dpi: int = Query(350, ge=100, le=600, description="Image resolution for table detection"),
    build_kg: bool = Query(True, description="Build knowledge graph in Neo4j")
):
    """
    Complete pipeline: Extract document + Build Knowledge Graph
    
    1. Extract document (text, tables, visuals)
    2. Build knowledge graph in Neo4j
    3. Return both extraction results and KG stats
    """
    request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("\n" + "=" * 80)
    print(f"NEW REQUEST (with KG): {request_id}")
    print(f"File: {file.filename}")
    print(f"Build KG: {build_kg}")
    print("=" * 80)
    
    try:
        # Step 1: Save and convert PDF
        print("\n[Step 1] Saving and converting PDF...")
        file_path = await file_handler.save_upload(file)
        pages = await pdf_processor.extract_pages(file_path, dpi=dpi)
        total_pages = len(pages)
        print(f"[OK] Extracted {total_pages} pages at {dpi} DPI")
        
        # Step 2: Process each page
        print(f"\n[Step 2] Processing {total_pages} pages...")
        print("=" * 80)
        
        all_page_results = []
        total_tables_found = 0
        total_visuals_found = 0
        
        for page_num, page_data in enumerate(pages, start=1):
            print(f"\n{'='*60}")
            print(f"Processing Page {page_num}/{total_pages}")
            print(f"{'='*60}")
            
            page_result = await process_single_page(
                page_number=page_num,
                page_data=page_data,
                total_pages=total_pages,
                pdf_path=file_path
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
            if visuals_on_page > 0:
                types = page_result['image_detection']['types']
                print(f"  - Visual types: {', '.join(types)}")
        
        # Step 3: Create final response
        print("\n" + "=" * 80)
        print("[Step 3] Consolidating results...")
        
        final_result = {
            "request_id": request_id,
            "filename": file.filename,
            "total_pages": total_pages,
            "pages": all_page_results,
            "summary": create_summary(all_page_results),
            "metadata": {
                "dpi": dpi,
                "extraction_method": "table-transformer + gemini-2.5-flash",
                "text_model": "gemini-2.0-flash-exp",
                "table_detection_model": "microsoft/table-transformer-detection",
                "version": "3.0.0",
                "features": [
                    "Multi-scale table detection",
                    "Merged cell handling",
                    "Page-by-page processing",
                    "Automatic retry on failure",
                    "Image/visual detection",
                    "Knowledge graph construction"
                ]
            },
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"\n[SUCCESS] Document analysis complete!")
        print(f"  Total pages: {total_pages}")
        print(f"  Total tables: {total_tables_found}")
        print(f"  Total visuals: {total_visuals_found}")
        print(f"  Total sections: {final_result['summary']['total_sections']}")
        
        # Step 4: Build Knowledge Graph (if requested)
        if build_kg:
            print("\n" + "=" * 80)
            print("[Step 4] Building Knowledge Graph...")
            kg_result = kg_builder.build_graph(final_result)
            final_result['knowledge_graph'] = kg_result
            print(f"[OK] Knowledge Graph built: {kg_result['total_nodes']} nodes, {kg_result['total_relationships']} relationships")
        else:
            final_result['knowledge_graph'] = {"status": "skipped"}
        
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


@app.post("/api/v1/extract-document")
async def extract_document(
    file: UploadFile = File(...),
    dpi: int = Query(350, ge=100, le=600, description="Image resolution for table detection")
):
    """
    Complete document extraction:
    1. Extract pages from PDF
    2. For each page:
       - Analyze text (sections, entities, clauses)
       - Detect tables (multi-scale detection)
       - Extract table structure (Gemini with merged cells)
    3. Return complete structured data
    """
    request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("\n" + "=" * 80)
    print(f"NEW REQUEST: {request_id}")
    print(f"File: {file.filename}")
    print(f"DPI: {dpi}")
    print("=" * 80)
    
    try:
        # Step 1: Save and convert PDF
        print("\n[Step 1] Saving and converting PDF...")
        file_path = await file_handler.save_upload(file)
        pages = await pdf_processor.extract_pages(file_path, dpi=dpi)
        total_pages = len(pages)
        print(f"[OK] Extracted {total_pages} pages at {dpi} DPI")
        
        # Step 2: Process each page
        print(f"\n[Step 2] Processing {total_pages} pages...")
        print("=" * 80)
        
        all_page_results = []
        total_tables_found = 0
        total_visuals_found = 0
        
        for page_num, page_data in enumerate(pages, start=1):
            print(f"\n{'='*60}")
            print(f"Processing Page {page_num}/{total_pages}")
            print(f"{'='*60}")
            
            page_result = await process_single_page(
                page_number=page_num,
                page_data=page_data,
                total_pages=total_pages,
                pdf_path=file_path
            )
            
            all_page_results.append(page_result)
            
            # Count tables and visuals
            tables_on_page = len(page_result.get('tables', []))
            visuals_on_page = len(page_result.get('visuals', []))
            total_tables_found += tables_on_page
            total_visuals_found += visuals_on_page
            
            print(f"[OK] Page {page_num} complete:")
            print(f"  - Text sections: {page_result['text_analysis']['sections_count']}")
            print(f"  - Tables found: {tables_on_page}")
            print(f"  - Visuals found: {visuals_on_page}")
            if visuals_on_page > 0:
                types = page_result['image_detection']['types']
                print(f"  - Visual types: {', '.join(types)}")
        
        # Step 3: Create final response
        print("\n" + "=" * 80)
        print("[Step 3] Consolidating results...")
        
        final_result = {
            "request_id": request_id,
            "filename": file.filename,
            "total_pages": total_pages,
            "pages": all_page_results,
            "summary": create_summary(all_page_results),
            "metadata": {
                "dpi": dpi,
                "extraction_method": "table-transformer + gemini-2.5-flash",
                "text_model": "gemini-2.0-flash-exp",
                "table_detection_model": "microsoft/table-transformer-detection",
                "version": "3.0.0",
                "features": [
                    "Multi-scale table detection",
                    "Merged cell handling",
                    "Page-by-page processing",
                    "Automatic retry on failure"
                ]
            },
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"\n[SUCCESS] Document analysis complete!")
        print(f"  Total pages: {total_pages}")
        print(f"  Total tables: {total_tables_found}")
        print(f"  Total visuals: {total_visuals_found}")
        print(f"  Total sections: {final_result['summary']['total_sections']}")
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


async def process_single_page(page_number: int, page_data: dict, total_pages: int, pdf_path: str = None) -> dict:
    """
    Process a single page completely:
    1. Text analysis
    2. Table detection & extraction
    3. Image/visual detection
    """
    result = {
        "page_number": page_number,
        "text_analysis": {},
        "tables": [],
        "visuals": [],
        "image_detection": {}
    }
    
    # Step 1: Text Analysis
    print(f"\n  [1/4] Analyzing text content...")
    result["text_analysis"] = await text_analyzer.analyze_text(
        page_number=page_number,
        page_data=page_data
    )
    print(f"  [OK] Found {result['text_analysis']['sections_count']} sections")
    
    # Step 2: Advanced Table Detection
    print(f"\n  [2/4] Detecting tables (multi-scale)...")
    detected_tables = table_detector.detect_tables(
        page_image=page_data['pil_image'],
        page_number=page_number
    )
    
    if detected_tables:
        print(f"  [OK] Detected {len(detected_tables)} table(s)")
        
        # Extract each table with Gemini
        print(f"\n  [3/4] Extracting table structures...")
        for idx, table_bbox in enumerate(detected_tables, start=1):
            print(f"    [i] Extracting table {idx}/{len(detected_tables)}...")
            
            table_data = await table_extractor.extract_table(
                page_image=page_data['pil_image'],
                bbox=table_bbox['bbox'],
                confidence=table_bbox['confidence'],
                page_number=page_number,
                table_index=idx
            )
            
            result['tables'].append(table_data)
            print(f"    [OK] Table {idx}: {table_data['total_rows']}x{table_data['total_columns']}")
    else:
        print(f"  [i] No tables detected on this page")
    
    # Step 4: Image/Visual Detection
    print(f"\n  [4/4] Detecting images and visuals...")
    detected_visuals = image_detector.detect_images(
        page_image=page_data['pil_image'],
        page_number=page_number,
        pdf_path=pdf_path
    )
    
    result['visuals'] = detected_visuals
    
    # Image detection summary
    result["image_detection"] = {
        "has_visuals": len(detected_visuals) > 0,
        "visual_count": len(detected_visuals),
        "types": list(set([v['type'] for v in detected_visuals])) if detected_visuals else []
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
            "alerts": []
        }
    }
    
    for page in page_results:
        # Sections
        summary['total_sections'] += page['text_analysis']['sections_count']
        
        # Tables
        table_count = len(page['tables'])
        if table_count > 0:
            summary['pages_with_tables'] += 1
            summary['total_tables'] += table_count
        
        # Visuals
        visual_count = len(page['visuals'])
        if visual_count > 0:
            summary['pages_with_visuals'] += 1
            summary['total_visuals'] += visual_count
            
            # Count visual types
            for visual in page['visuals']:
                vtype = visual['type']
                summary['visual_types'][vtype] = summary['visual_types'].get(vtype, 0) + 1
        
        # Entities
        entities = page['text_analysis']['entities']
        if entities.get('buyer_name') and not summary['entities']['buyer_name']:
            summary['entities']['buyer_name'] = entities['buyer_name']
        if entities.get('seller_name') and not summary['entities']['seller_name']:
            summary['entities']['seller_name'] = entities['seller_name']
        
        summary['entities']['dates'].extend(entities.get('dates', []))
        summary['entities']['deadlines'].extend(entities.get('deadlines', []))
        summary['entities']['alerts'].extend(entities.get('alerts', []))
    
    # Deduplicate
    summary['entities']['dates'] = list(set(summary['entities']['dates']))
    summary['entities']['deadlines'] = list(set(summary['entities']['deadlines']))
    summary['entities']['alerts'] = list(set(summary['entities']['alerts']))
    
    return summary


@app.get("/health")
async def health_check():
    """Check system health and model status"""
    return {
        "status": "healthy",
        "models": {
            "text_analysis": "gemini-2.0-flash-exp",
            "table_detection": table_detector.get_status(),
            "table_extraction": table_extractor.get_status()
        },
        "version": "3.0.0"
    }


@app.get("/")
async def root():
    return {
        "service": "ContractX Complete Document Analysis",
        "version": "3.0.0",
        "endpoints": {
            "extract": "/api/v1/extract-document",
            "health": "/health",
            "docs": "/docs"
        },
        "features": [
            "Page-by-page processing",
            "Multi-scale table detection",
            "Gemini table extraction",
            "Merged cell handling",
            "Text analysis (sections, entities)",
            "Automatic retry logic"
        ]
    }


@app.get("/")
async def root():
    return {
        "service": "ContractX Complete Document Analysis",
        "version": "3.0.0",
        "endpoints": {
            "extract": "/api/v1/extract-document",
            "extract_and_kg": "/api/v1/extract-and-build-kg",
            "query_kg": "/api/v1/query-kg",
            "health": "/health",
            "docs": "/docs"
        },
        "features": [
            "Page-by-page processing",
            "Multi-scale table detection",
            "Gemini table extraction",
            "Merged cell handling",
            "Text analysis (sections, entities)",
            "Image/visual detection",
            "Knowledge graph construction",
            "Automatic retry logic"
        ]
    }


@app.post("/api/v1/query-kg")
async def query_knowledge_graph(cypher_query: str = Query(..., description="Cypher query to execute")):
    """
    Query the knowledge graph using Cypher
    
    Example queries:
    - Get all sections: MATCH (s:Section) RETURN s LIMIT 10
    - Find buyer: MATCH (b:Buyer) RETURN b
    - Get tables with merged cells: MATCH (t:Table {has_merged_cells: true}) RETURN t
    - Get page structure: MATCH (p:Page {page_number: 1})-[r]->(n) RETURN p, r, n
    """
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


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)