import os
import json
import base64
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from pdf2image import convert_from_bytes
import io
from app.config.config import Config
from app.services.LLM_tracker import LLMUsageTracker  # NEW IMPORT

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=Config.GEMINI_API_KEY)


class PDFProcessor:
    """Class for processing PDF files and extracting tables"""
    
    def __init__(self):
        self.output_dir = Path("extracted_tables")
        self.log_dir = Path("logs")
        self.output_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
    
    async def extract_pages(self, pdf_path: str, dpi: int = None) -> List[Dict[str, Any]]:
        """
        Convert PDF file to image pages only
        
        Args:
            pdf_path: Path to PDF file
            dpi: DPI for image conversion (default None)
            
        Returns:
            List of page data dictionaries with only 'pil_image'
        """
        try:
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            
            images = convert_from_bytes(pdf_bytes, dpi=dpi)
            
            page_data_list = [{"pil_image": page_image, "text": ""} for page_image in images]
            
            logger.info(f"Converted {len(images)} pages from {pdf_path}")
            return page_data_list
        
        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            raise
    
    async def detect_tables_in_page(
        self, 
        page_image, 
        page_num: int, 
        total_pages: int,
        usage_tracker: Optional[LLMUsageTracker] = None  # NEW PARAMETER
    ) -> Dict[str, Any]:
        """
        Detect and extract tables from a single page using Gemini API
        
        Args:
            page_image: PIL Image object
            page_num: Current page number
            total_pages: Total number of pages
            usage_tracker: LLMUsageTracker instance for table LLM (optional)
            
        Returns:
            Dictionary with table detection results
        """
        return detect_tables_in_page(
            page_image, 
            page_num, 
            total_pages,
            usage_tracker=usage_tracker  # PASS TRACKER
        )


# Create output directory
OUTPUT_DIR = Path("extracted_tables")
LOG_DIR = Path("logs")
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Configure logging
log_filename = LOG_DIR / f"extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)

# Table extraction instruction prompt
INSTRUCTION_TEXT = """TABLE EXTRACTION (COMPLETE STRUCTURE)
CONSIDER THE ATTACHMENT AS A MAINFRAME SPOOL REPORT GENERATED FROM JCL AS A PDS FILE AND EXTRACT THE SAME AS PER THE BELOW PROMPT. SOME ROWS MAY CONTAIN A NOTATION BEFORE EACH ROW (EXAMPLE. "R"), KEEP THAT AS A SEPERATE ROW LEVEL ATTRIBUTE.
SAME THING FOR ANY ATTACHMENT LIKE EXCELSHEET, BORDERLESS TABLE FROM THE PDF FILES AND EXTRACT THE SAME AS PER THE BELOW PROMPT.
DETECTION STRATEGY:
Scan ENTIRE page for:
Do NOT miss ANY tables or inside the table rows, including headers rows detect properly in flattern layer:
• Explicit grid lines (bordered tables)
• Financial tables (numbers, currency, percentages)
• Schedule tables (dates, timelines, deliverables)
• Comparison tables (features, specifications)
• Nested tables (tables within tables)
• Tables in margins, headers, footers
• Tables embedded in screenshots

⚠️ CRITICAL: DETECT TABLE CONTINUATION
1. Extract ALL rows (no truncation!)
2. Extract ALL columns
3. Handle merged cells correctly for both rows and columns
4. Preserve exact cell values
• If a table is cut off at the bottom of the page, mark "continues_to_next_page": true
• If a table starts at the top without headers (continuing from previous page), mark "continued_from_previous_page": true
• Look for indicators like:
  - Table ends at page bottom without closing line
  - No clear table footer or totals row
  - Content appears truncated
  - Page starts with table rows but no headers

FOR EVERY TABLE:
1. STRUCTURE EXTRACTION
   • table_id: Unique identifier (e.g., T1, T2, T3...)
   • table_title: Title or caption (if present, otherwise empty string)
   • position: Location on page (e.g., "top", "center", "bottom")
   • size: Coverage area (small/medium/large/full-width)
   • table_type: "financial", "schedule", "comparison", "data", "specification"
   • continues_to_next_page: true/false (if table is cut off at bottom)
   • continued_from_previous_page: true/false (if table continues from previous page)

2. HEADERS (Compulsory check properly)
If headers are in one row: Use flat array format and preserve order and structured format do not miss any headers.
   • Extract ALL header rows (may be multiple rows)
   • Preserve column header hierarchy
   • Example: ["Deliverables", "Estimated Sign-off", "Status"]
   • If continuing from previous page, headers may be empty: []

3. DATA ROWS - COMPLETE EXTRACTION
   • Extract EVERY row
   • Extract EVERY cell value
   • NO truncation, NO "..." placeholders
   • Preserve cell data types (text, number, currency, date, percentage)

4. MERGED CELLS DETECTION & HANDLING ⚠️ CRITICAL
   
   VISUAL INDICATORS OF MERGED CELLS:
   • Cell borders that span multiple rows/columns
   • Single text value positioned over multiple row/column spaces
   • Large whitespace in bordered areas
   • Content logically shared across rows
   
   MERGED CELL EXTRACTION RULES:
   • If cell spans multiple rows → REPEAT value in each row
   • If cell spans multiple columns → REPEAT value in each column
   • Mark merged regions in "merged_cells" field
   • NEVER output "(blank)" where merged cells exist
   
   EXAMPLE INPUT (Visual):
   ┌────────────────────────────┬──────────────┐
   │ Task 1                     │              │
   ├────────────────────────────┤  Aug 2024    │
   │ Task 2                     │              │
   ├────────────────────────────┼──────────────┤

   ANOTHER EXAMPLE INPUT (Visual):
-------------------------------------------------------------------------------
FIN   | FUNCTIONAL DESIGNATION  | PANEL | ZONE | ACCESS DOOR | ATA REF.       |
-------------------------------------------------------------------------------
19FP1 | ADM-L TOTAL PRESSURE    | NONE  | 125  |   812       |  34-11-17      |
19FP2 | ADM-R TOTAL PRESSURE    |       | 126  |   822       |  34-11-17      |
19FP3 | ADM-STBY TOTAL PRESSURE |       | 125  |   812       |  34-11-17      |

    ANOTHER EXAMPLE INPUT (Visual):
------------------------------------------------------------------
| EQUIPMENT | 28VDC             | 115VAC            | 26VAC      |
|           |-------------------|-------------------|            |
|           | Typical | Maximum | Typical | Maximum |            |
------------------------------------------------------------------

   REQUIRED OUTPUT:
   {
     "rows": [
       ["Task 1"], ["Aug 2024"]],
       ["Task 2"], ["Aug 2024"]]
     ],
     "has_merged_cells": true,
     "merged_cells": "Column 2: 'Aug 2024' spans rows 1-2"
   }

5. TABLE METADATA
   • total_rows: Row count (excluding headers)
   • total_columns: Column count
   • has_merged_cells: true/false
   • merged_cells: Description of merged regions (string or null)
   • continues_to_next_page: true/false
   • continued_from_previous_page: true/false
   • data_types: Type of each column (optional)
   • notes: Any footnotes or table notes (optional)

REQUIRED JSON OUTPUT SCHEMA:
{
  "table_id": "T<idx>",
  "table_title": "<title or empty string>",
  "position": "<position on page>",
  "size": "<small/medium/large/full-width>",
  "table_type": "<financial/schedule/comparison/data/specification>",
  "headers": [
     ["Column1"],
     ["Column2"],
     ["Column3"]
  ],
  "rows": [
   [
     ["Value1"],
     ["Value2"],
     ["Value3"]
   ],
   [
     ["Value4"],
     ["Value5"],
     ["Value6"]
   ]
  ],
  "total_rows": <number>,
  "total_columns": <number>,
  "has_merged_cells": <true/false>,
  "merged_cells": "<description or null>",
  "data_types": ["text", "number", "date"],
  "notes": "<any footnotes or empty string>"
}

IMPORTANT RULES:
- Return ONLY valid JSON with no commentary or markdown
- Extract ALL rows completely, no truncation
- Handle merged cells by repeating values as shown above
- Always check if table continues to next page or continues from previous page
- If a field is optional and not applicable, you may omit it or set to null/empty
- Ensure headers are distinct from data rows
- If NO tables found on page, return: {"has_tables": false, "tables": []}
- If tables found, return: {"has_tables": true, "tables": [<table1>, <table2>, ...]}
- Headers MUST be returned as an array of arrays, where each column header is wrapped in its own array
- Each row MUST be an array of cells
- Each cell MUST be wrapped inside its own array (even if it contains only one value)
- Never return plain strings for headers or row cells
"""


class TableExtractionResponse(BaseModel):
    filename: str
    total_pages: int
    pages_with_tables: int
    total_tables_extracted: int
    processing_time_seconds: float
    extraction_results: List[Dict[str, Any]]
    output_file: str
    log_file: str


def encode_image_to_base64(image) -> str:
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def detect_tables_in_page(
    page_image, 
    page_num: int, 
    total_pages: int,
    usage_tracker: Optional[LLMUsageTracker] = None  # NEW PARAMETER
) -> Dict[str, Any]:
    """
    Detect and extract tables from a single page using Gemini API
    
    Args:
        page_image: PIL Image object
        page_num: Current page number
        total_pages: Total number of pages
        usage_tracker: LLMUsageTracker instance for tracking table extraction LLM calls
    """
    page_start_time = time.time()
    logger.info(f"Starting page {page_num}/{total_pages}")

    model_names = [
        Config.GEMINI_MODEL
    ]

    last_error = None
    model_used = None

    try:
        # Try model names one by one
        for model_name in model_names:
            try:
                logger.info(f"  Trying model: {model_name}")
                model_start = time.time()
                
                model = genai.GenerativeModel(model_name)

                prompt = f"""Analyze this PDF page (Page {page_num}) and extract ALL tables found.

{INSTRUCTION_TEXT}

Respond with ONLY valid JSON, no markdown formatting, no code blocks."""

                # TRACK LLM USAGE: Start request
                if usage_tracker:
                    usage_tracker.start_request(prompt)
                
                response = model.generate_content([prompt, page_image])
                
                # TRACK LLM USAGE: End request
                if usage_tracker:
                    usage_tracker.end_request(response)
                
                # Check if response was blocked by safety filters
                if not response.parts:
                    if hasattr(response, 'prompt_feedback'):
                        block_reason = response.prompt_feedback
                        logger.warning(f"  Response blocked by safety filters: {block_reason}")
                    else:
                        logger.warning(f"  Response blocked - no content returned")
                    
                    # Return empty result for this page
                    page_time = time.time() - page_start_time
                    return {
                        "page_number": page_num,
                        "model_used": model_name,
                        "processing_time_seconds": round(page_time, 2),
                        "has_tables": False,
                        "tables": [],
                        "table_count": 0,
                        "error": "Response blocked by safety filters"
                    }

                response_text = response.text.strip()
                model_time = time.time() - model_start
                logger.info(f"  Success with {model_name} in {model_time:.2f}s")
                model_used = model_name
                break  # success

            except Exception as e:
                logger.warning(f"  Failed with {model_name}: {str(e)[:100]}")
                last_error = e
                if "404" in str(e) or "not found" in str(e).lower():
                    continue  # try next model
                else:
                    raise

        else:
            # After loop, no model succeeded
            error_msg = f"No available Gemini models found. Last error: {last_error}"
            logger.error(f"  {error_msg}")
            raise Exception(error_msg)

        # ---------------------------
        # Parse JSON response
        # ---------------------------
        logger.info(f"  Parsing response...")

        # Remove code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        response_text = response_text.strip()

        result = json.loads(response_text)

        page_time = time.time() - page_start_time
        
        has_tables = result.get("has_tables", False)
        table_count = len(result.get("tables", []))
        
        if has_tables:
            logger.info(f"  Found {table_count} table(s) on page {page_num}")
            
            # Log continuation info
            for idx, table in enumerate(result.get("tables", []), 1):
                continued_from = table.get("continued_from_previous_page", False)
                continues_to = table.get("continues_to_next_page", False)
                
                if continued_from:
                    logger.info(f"     Table T{idx}: Continued from previous page")
                if continues_to:
                    logger.info(f"     Table T{idx}: Continues to next page")
        else:
            logger.info(f"  No tables found on page {page_num}")
        
        logger.info(f"  Page {page_num} completed in {page_time:.2f}s")

        return {
            "page_number": page_num,
            "model_used": model_used, 
            "processing_time_seconds": round(page_time, 2),
            "has_tables": result.get("has_tables", False),
            "tables": result.get("tables", []),
            "table_count": len(result.get("tables", []))
        }

    except json.JSONDecodeError as e:
        page_time = time.time() - page_start_time
        error_msg = f"Failed to parse JSON response: {str(e)}"
        logger.error(f"  {error_msg}")
        logger.debug(f"  Response text: {response_text[:500]}")

        return {
            "page_number": page_num,
            "processing_time_seconds": round(page_time, 2),
            "has_tables": False,
            "tables": [],
            "table_count": 0,
            "error": f"Failed to parse JSON response: {str(e)}"
        }

    except Exception as e:
        page_time = time.time() - page_start_time
        error_msg = f"Error processing page: {str(e)}"
        logger.error(f"  {error_msg}")
        
        return {
            "page_number": page_num,
            "processing_time_seconds": round(page_time, 2),
            "has_tables": False,
            "tables": [],
            "table_count": 0,
            "error": error_msg
        }

def merge_continued_tables(extraction_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge tables that span multiple pages
    """
    logger.info("Checking for multi-page table continuations...")
    
    merged_results = []
    
    for i, page_result in enumerate(extraction_results):
        if not page_result.get("has_tables"):
            merged_results.append(page_result)
            continue
        
        tables = page_result.get("tables", [])
        merged_tables = []
        
        for table in tables:
            # Check if this table continues from previous page
            if table.get("continued_from_previous_page") and i > 0:
                # Find the table from previous page that continues here
                prev_page = merged_results[-1]
                if prev_page.get("has_tables"):
                    prev_tables = prev_page.get("tables", [])
                    
                    # Find the last table that continues to next page
                    for prev_table in reversed(prev_tables):
                        if prev_table.get("continues_to_next_page"):
                            # Merge rows
                            logger.info(f"Merging table from page {i} to page {i+1}")
                            prev_table["rows"].extend(table["rows"])
                            prev_table["total_rows"] = len(prev_table["rows"])
                            prev_table["continues_to_next_page"] = table.get("continues_to_next_page", False)
                            
                            # Add note about merge
                            if "notes" in prev_table and prev_table["notes"]:
                                prev_table["notes"] += f" | Continued to page {i+1}"
                            else:
                                prev_table["notes"] = f"Table spans pages {i} to {i+1}"
                            
                            # Don't add this table separately
                            break
                    else:
                        # No matching previous table found, add as new
                        merged_tables.append(table)
                else:
                    merged_tables.append(table)
            else:
                merged_tables.append(table)
        
        page_result["tables"] = merged_tables
        page_result["table_count"] = len(merged_tables)
        merged_results.append(page_result)
    
    return merged_results

def process_pdf(pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Process entire PDF and extract tables from all pages
    """
    start_time = time.time()
    logger.info("="*80)
    logger.info(f"Starting PDF extraction: {filename}")
    logger.info("="*80)

    try:
        # Convert PDF to images
        logger.info("Converting PDF to images...")
        convert_start = time.time()
        images = convert_from_bytes(pdf_bytes, dpi=None)
        convert_time = time.time() - convert_start
        
        total_pages = len(images)
        logger.info(f"Converted {total_pages} pages in {convert_time:.2f}s")
        
        all_results = []
        total_tables = 0
        pages_with_tables = 0
        
        # Process each page
        for page_num, image in enumerate(images, start=1):
            print(f"Processing page {page_num}/{total_pages}...")
            
            page_result = detect_tables_in_page(image, page_num, total_pages)
            all_results.append(page_result)
            
            if page_result["has_tables"]:
                pages_with_tables += 1
                total_tables += page_result["table_count"]
        
         # Merge tables that span multiple pages
        all_results = merge_continued_tables(all_results)
        
        # Recalculate totals after merging
        total_tables = sum(r.get("table_count", 0) for r in all_results)
        pages_with_tables = sum(1 for r in all_results if r.get("has_tables"))
        
        total_time = time.time() - start_time
        
        logger.info("="*80)
        logger.info(f"PDF Processing Complete")
        logger.info(f"  Total pages: {total_pages}")
        logger.info(f"  Pages with tables: {pages_with_tables}")
        logger.info(f"  Total tables extracted: {total_tables}")
        logger.info(f"  Total processing time: {total_time:.2f}s")
        logger.info(f"  Average time per page: {total_time/total_pages:.2f}s")
        logger.info("="*80)
        
        # Create output structure
        output_data = {
            "document_name": filename,
            "extraction_timestamp": datetime.now().isoformat(),
            "total_pages": total_pages,
            "pages_with_tables": pages_with_tables,
            "total_tables_extracted": total_tables,
            "processing_time_seconds": round(total_time, 2),
            "average_time_per_page": round(total_time/total_pages, 2),
            "extraction_results": all_results
        }
        
        # Save to JSON file
        output_filename = f"{Path(filename).stem}_tables.json"
        output_path = OUTPUT_DIR / output_filename
        
        logger.info(f"Saving results to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        return {
            "filename": filename,
            "total_pages": total_pages,
            "pages_with_tables": pages_with_tables,
            "total_tables_extracted": total_tables,
            "processing_time_seconds": round(total_time, 2),
            "extraction_results": all_results,
            "output_file": str(output_path),
            "log_file": str(log_filename)
        }
        
    except Exception as e:
        logger.error(f"Fatal error processing PDF: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")



# @app.post("/extract-tables/", response_model=TableExtractionResponse)
# async def extract_tables(file: UploadFile = File(...)):
#     """
#     Extract tables from uploaded PDF file
    
#     - Detects all tables in the PDF
#     - Extracts complete table structure including merged cells
#     - Handles tables that span multiple pages
#     - Returns JSON with all extracted tables
#     - Saves results to extracted_tables directory
#     - Creates detailed log file for debugging
#     """
#     # Validate file type
#     if not file.filename.endswith('.pdf'):
#         raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
#     # Check API key
#     if not os.getenv("GEMINI_API_KEY"):
#         raise HTTPException(
#             status_code=500, 
#             detail="GEMINI_API_KEY not found in environment variables"
#         )
    
#     try:
#         # Read PDF file
#         pdf_bytes = await file.read()
        
#         # Process PDF
#         result = process_pdf(pdf_bytes, file.filename)
        
#         return JSONResponse(content=result)
        
#     except Exception as e:
#         logger.error(f"API Error: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))
