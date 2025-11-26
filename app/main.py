from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Optional
import os
from pathlib import Path
import logging
from datetime import datetime

from app.services.document_processor import DocumentProcessor
from app.services.text_llm_service import TextLLMService
from app.services.table_detection_service import TableDetectionService
from app.services.table_llm_service import TableLLMService
from app.services.merger_service import MergerService
from app.utils.file_handler import FileHandler
from app.database.schemas import DocumentAnalysisResponse
from app.config import config
from app.services.chunking_service import ChunkingService

# Configure main application logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/main_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ContractX Document Analysis API", 
    version="1.0.0",
    description="Advanced document analysis with AI-powered table detection and extraction"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
logger.info("="*80)
logger.info("Initializing ContractX Services")
logger.info("="*80)

file_handler = FileHandler()
document_processor = DocumentProcessor()
chunking_service = ChunkingService()
text_llm_service = TextLLMService()
table_detection_service = TableDetectionService()
table_llm_service = TableLLMService()
merger_service = MergerService()

logger.info("✓ All services initialized successfully")
logger.info("="*80)

@app.post("/api/v1/analyze-document", response_model=DocumentAnalysisResponse)
async def analyze_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(5),
    overlap: int = Form(1)
):
    """
    Main endpoint for document analysis
    
    Args:
        file: PDF or DOC file
        chunk_size: Number of pages per chunk (default: 5)
        overlap: Number of overlapping pages between chunks (default: 1)
    
    Returns:
        Complete document analysis with sections, entities, and tables
    """
    request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    logger.info("="*80)
    logger.info(f"NEW REQUEST: {request_id}")
    logger.info(f"File: {file.filename}")
    logger.info(f"Chunk Size: {chunk_size}")
    logger.info(f"Overlap: {overlap}")
    logger.info("="*80)
    
    try:
        # Step 1: Save and validate uploaded file
        logger.info("Step 1: Saving and validating file...")
        file_path = await file_handler.save_upload(file)
        logger.info(f"✓ File saved: {file_path}")
        
        # Step 2: Convert document to pages (text + images + tables)
        logger.info("Step 2: Converting document to pages...")
        pages = await document_processor.convert_to_pages(file_path)
        logger.info(f"✓ Extracted {len(pages)} pages")
        
        # Step 3: Create chunks with metadata
        logger.info("Step 3: Creating chunks...")
        chunks = chunking_service.create_chunks(
            pages=pages,
            chunk_size=chunk_size,
            overlap=overlap
        )
        logger.info(f"✓ Created {len(chunks)} chunks")
        
        # Log chunk statistics
        stats = chunking_service.get_chunk_statistics(chunks)
        logger.info(f"Chunk Statistics: {stats}")
        
        # Step 4: Process each chunk
        logger.info("="*80)
        logger.info("Step 4: Processing Chunks with Text LLM and Table Detection")
        logger.info("="*80)
        
        all_text_results = []
        all_table_detections = []
        
        for idx, chunk in enumerate(chunks, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing Chunk {idx}/{len(chunks)}")
            logger.info(f"{'='*80}")
            
            # Step 4a: Text LLM processing
            logger.info("4a: Text LLM Analysis...")
            text_result = await text_llm_service.process_chunk(chunk)
            all_text_results.append(text_result)
            logger.info(f"✓ Text LLM completed for chunk {idx}")
            
            # Step 4b: Table detection using HuggingFace model
            logger.info("4b: HuggingFace Table Detection...")
            table_detection = await table_detection_service.detect_tables(chunk)
            all_table_detections.append(table_detection)
            logger.info(f"✓ Table Detection completed for chunk {idx}")
        
        logger.info(f"\n{'='*80}")
        logger.info("✓ All chunks processed successfully")
        logger.info("="*80)
        
        # Step 5: Compare and resolve table conflicts
        logger.info("Step 5: Resolving table detection conflicts...")
        resolved_tables = merger_service.resolve_table_conflicts(
            text_results=all_text_results,
            table_detections=all_table_detections
        )
        logger.info(f"✓ Resolved {len(resolved_tables)} tables")
        
        # Step 6: Process confirmed tables with Table LLM
        logger.info("="*80)
        logger.info("Step 6: Extracting Table Structures with Table LLM")
        logger.info("="*80)
        
        final_tables = []
        for idx, table_info in enumerate(resolved_tables, 1):
            logger.info(f"\nProcessing table {idx}/{len(resolved_tables)}")
            table_result = await table_llm_service.extract_table_structure(
                chunk=table_info['chunk'],
                table_metadata=table_info['metadata']
            )
            final_tables.append(table_result)
        
        logger.info(f"✓ Extracted structures for {len(final_tables)} tables")
        
        # Step 7: Merge all results into final output
        logger.info("="*80)
        logger.info("Step 7: Merging all results...")
        logger.info("="*80)
        
        final_result = merger_service.merge_results(
            text_results=all_text_results,
            final_tables=final_tables,
            chunks=chunks
        )
        
        logger.info("✓ Results merged successfully")
        
        # Cleanup temporary files
        await file_handler.cleanup(file_path)
        logger.info("✓ Temporary files cleaned up")
        
        logger.info("="*80)
        logger.info(f"REQUEST {request_id} COMPLETED SUCCESSFULLY")
        logger.info(f"Summary:")
        logger.info(f"  - Total Pages: {final_result['metadata']['total_pages']}")
        logger.info(f"  - Total Sections: {final_result['metadata']['total_sections']}")
        logger.info(f"  - Total Tables: {final_result['metadata']['total_tables']}")
        logger.info(f"  - Total Images: {final_result['metadata']['total_images']}")
        logger.info("="*80)
        
        return JSONResponse(content=final_result, status_code=200)
        
    except Exception as e:
        logger.error("="*80)
        logger.error(f"✗ REQUEST {request_id} FAILED")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("="*80)
        
        return JSONResponse(
            content={
                "error": str(e), 
                "status": "failed",
                "request_id": request_id
            },
            status_code=500
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ContractX Document Analysis"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)