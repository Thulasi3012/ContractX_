# from openai import AsyncOpenAI
# import os
# import json
# import time
# import asyncio
# from typing import Dict, Any, List
# from app.database.schemas import Chunk, FinalTable, TableStructure
# from PIL import Image
# import io
# import base64
# import logging
# from datetime import datetime

# # Configure detailed logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(f'logs/table_llm_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# class TableLLMService:
#     """Extracts detailed table structure using OpenAI GPT with vision capabilities"""
    
#     def __init__(self):
#         # Configure OpenAI API
#         api_key = os.getenv("OPENAI_API_KEY")
#         if not api_key:
#             raise ValueError("OPENAI_API_KEY environment variable not set")
        
#         self.client = AsyncOpenAI(api_key=api_key)
#         self.model = "gpt-4o"  # Use gpt-4o for vision + text, or "gpt-4-turbo" as alternative
        
#         # Rate limiting configuration
#         self.request_delay = float(os.getenv("OPENAI_REQUEST_DELAY", "2.5"))  # Slightly longer for table processing
#         self.max_retries = int(os.getenv("MAX_RETRIES", "5"))
#         self.retry_delay = int(os.getenv("RETRY_DELAY", "60"))
#         self.exponential_backoff = True
        
#         logger.info("="*80)
#         logger.info("TableLLMService Initialized")
#         logger.info(f"Model: {self.model}")
#         logger.info(f"Request Delay: {self.request_delay}s")
#         logger.info(f"Max Retries: {self.max_retries}")
#         logger.info(f"Retry Delay: {self.retry_delay}s")
#         logger.info("="*80)
    
#     async def extract_table_structure(
#         self, 
#         chunk: Chunk, 
#         table_metadata: Dict[str, Any]
#     ) -> FinalTable:
#         """
#         Extract detailed table structure from detected table region
        
#         Args:
#             chunk: Document chunk containing the table
#             table_metadata: Metadata about detected table (page, bbox, etc.)
            
#         Returns:
#             Structured table data with rows, columns, and headers
#         """
#         page_num = table_metadata.get('page')
#         table_id = table_metadata.get('table_id', f"table_{page_num}")
        
#         logger.info("="*80)
#         logger.info(f"Extracting Table Structure: {table_id} (Page {page_num})")
#         logger.info(f"Source: {table_metadata.get('source', 'unknown')}")
#         logger.info(f"Confidence: {table_metadata.get('confidence', 'N/A')}")
#         logger.info("="*80)
        
#         # Apply rate limiting
#         logger.info(f"Applying rate limit: waiting {self.request_delay}s before API call...")
#         await asyncio.sleep(self.request_delay)
        
#         try:
#             # Get the relevant text context around the table
#             context_text = self._extract_table_context(chunk, page_num)
#             logger.info(f"Context text length: {len(context_text)} characters")
            
#             # Get table image if available
#             table_image = self._get_table_image(chunk, table_metadata)
#             if table_image:
#                 logger.info(f"Table image extracted: {table_image.size[0]}x{table_image.size[1]}")
#             else:
#                 logger.warning("⚠ No table image available, using text context only")
            
#             # Create prompt for table extraction
#             prompt = self._create_table_extraction_prompt(context_text, table_metadata)
            
#             # Retry logic with exponential backoff
#             for attempt in range(1, self.max_retries + 1):
#                 try:
#                     logger.info(f"Attempt {attempt}/{self.max_retries} - Sending request to OpenAI API")
#                     start_time = time.time()
                    
#                     # Build messages array
#                     messages = []
                    
#                     if table_image:
#                         # Convert image to base64 for OpenAI
#                         base64_image = self._image_to_base64(table_image)
#                         messages.append({
#                             "role": "user",
#                             "content": [
#                                 {"type": "text", "text": prompt},
#                                 {
#                                     "type": "image_url",
#                                     "image_url": {
#                                         "url": f"data:image/png;base64,{base64_image}"
#                                     }
#                                 }
#                             ]
#                         })
#                         logger.info("Request sent with table image (multimodal)")
#                     else:
#                         messages.append({
#                             "role": "user",
#                             "content": prompt
#                         })
#                         logger.info("Request sent with text context only")
                    
#                     # Generate response
#                     response = await self.client.chat.completions.create(
#                         model=self.model,
#                         messages=messages,
#                         temperature=0.0,
#                         response_format={"type": "json_object"}
#                     )
                    
#                     elapsed_time = time.time() - start_time
#                     logger.info(f"✓ API Response received in {elapsed_time:.2f}s")
                    
#                     # Parse response
#                     result_text = response.choices[0].message.content.strip()
#                     logger.info(f"Response length: {len(result_text)} characters")
                    
#                     # Clean markdown blocks (in case response_format doesn't work perfectly)
#                     if result_text.startswith("```json"):
#                         result_text = result_text[7:]
#                     if result_text.startswith("```"):
#                         result_text = result_text[3:]
#                     if result_text.endswith("```"):
#                         result_text = result_text[:-3]
                    
#                     table_data = json.loads(result_text.strip())
#                     logger.info("✓ JSON parsed successfully")
                    
#                     # Log extracted table info
#                     rows = table_data.get('rows', 0)
#                     cols = table_data.get('columns', 0)
#                     headers = table_data.get('headers', [])
#                     data_rows = len(table_data.get('data', []))
                    
#                     logger.info(f"Table Structure Extracted:")
#                     logger.info(f"  - Rows: {rows}")
#                     logger.info(f"  - Columns: {cols}")
#                     logger.info(f"  - Headers: {headers}")
#                     logger.info(f"  - Data Rows: {data_rows}")
                    
#                     # Validate data completeness
#                     if data_rows != rows:
#                         logger.warning(f"⚠ Row count mismatch: declared={rows}, actual={data_rows}")
                    
#                     # Create structured output
#                     structure = TableStructure(
#                         page=page_num,
#                         table_id=table_id,
#                         rows=table_data.get('rows', 0),
#                         columns=table_data.get('columns', 0),
#                         headers=table_data.get('headers', []),
#                         data=table_data.get('data', []),
#                         metadata=table_data.get('metadata', {})
#                     )
                    
#                     final_table = FinalTable(
#                         table_id=table_id,
#                         page=page_num,
#                         structure=structure,
#                         source=table_metadata.get('source', 'unknown'),
#                         confidence=table_metadata.get('confidence', 0.0)
#                     )
                    
#                     logger.info(f"✓ Table {table_id} processed successfully")
#                     logger.info("="*80)
#                     return final_table
                    
#                 except json.JSONDecodeError as e:
#                     logger.error(f"✗ JSON parsing error for table {table_id} (Attempt {attempt})")
#                     logger.error(f"Error: {str(e)}")
#                     logger.error(f"Response text (first 500 chars): {result_text[:500]}")
                    
#                     if attempt < self.max_retries:
#                         retry_wait = self._calculate_retry_delay(attempt)
#                         logger.warning(f"Retrying in {retry_wait}s...")
#                         await asyncio.sleep(retry_wait)
#                     else:
#                         logger.error(f"✗ Max retries reached for table {table_id}")
#                         return self._create_empty_table(table_metadata)
                
#                 except Exception as e:
#                     error_msg = str(e)
#                     logger.error(f"✗ Error extracting table {table_id} (Attempt {attempt})")
#                     logger.error(f"Error type: {type(e).__name__}")
#                     logger.error(f"Error message: {error_msg}")
                    
#                     # Check for quota/rate limit errors
#                     if any(keyword in error_msg.lower() for keyword in ['quota', 'rate limit', 'resource exhausted', '429']):
#                         logger.warning("⚠ QUOTA/RATE LIMIT ERROR DETECTED")
                        
#                         if attempt < self.max_retries:
#                             retry_wait = self._calculate_retry_delay(attempt, is_quota_error=True)
#                             logger.warning(f"Pausing for {retry_wait}s before retry...")
#                             await asyncio.sleep(retry_wait)
#                         else:
#                             logger.error(f"✗ Max retries reached for table {table_id} after quota errors")
#                             return self._create_empty_table(table_metadata)
#                     else:
#                         if attempt < self.max_retries:
#                             retry_wait = self._calculate_retry_delay(attempt)
#                             logger.warning(f"Retrying in {retry_wait}s...")
#                             await asyncio.sleep(retry_wait)
#                         else:
#                             logger.error(f"✗ Max retries reached for table {table_id}")
#                             return self._create_empty_table(table_metadata)
            
#         except Exception as e:
#             logger.error(f"✗ Unexpected error extracting table structure: {e}")
#             return self._create_empty_table(table_metadata)
    
#     def _calculate_retry_delay(self, attempt: int, is_quota_error: bool = False) -> int:
#         """Calculate retry delay with exponential backoff"""
#         if is_quota_error:
#             base_delay = self.retry_delay * 3  # Even longer for quota errors on tables
#         else:
#             base_delay = self.retry_delay
        
#         if self.exponential_backoff:
#             delay = base_delay * (2 ** (attempt - 1))
#             logger.info(f"Exponential backoff: attempt {attempt} -> {delay}s delay")
#         else:
#             delay = base_delay
        
#         return min(delay, 600)  # Cap at 10 minutes
    
#     def _extract_table_context(self, chunk: Chunk, page_num: int) -> str:
#         """Extract text context around the table from the chunk"""
#         for page_data in chunk.raw_pages:
#             if page_data['page_number'] == page_num:
#                 return page_data.get('text', '')
#         return chunk.text
    
#     def _get_table_image(self, chunk: Chunk, table_metadata: Dict[str, Any]) -> Image.Image:
#         """Extract table region as image for vision-based analysis"""
#         try:
#             page_num = table_metadata.get('page')
#             bbox = table_metadata.get('bounding_box')
            
#             if not bbox:
#                 logger.warning("No bounding box provided")
#                 return None
            
#             # Find raw page
#             raw_page = None
#             for page_data in chunk.raw_pages:
#                 if page_data['page_number'] == page_num:
#                     raw_page = page_data.get('raw_page')
#                     break
            
#             if raw_page is None:
#                 logger.warning(f"No raw page found for page {page_num}")
#                 return None
            
#             # Convert page to image
#             import fitz
#             mat = fitz.Matrix(2.0, 2.0)
#             pix = raw_page.get_pixmap(matrix=mat)
#             img_data = pix.tobytes("png")
#             page_image = Image.open(io.BytesIO(img_data))
            
#             # Crop to table region
#             table_image = page_image.crop((
#                 bbox['x_min'],
#                 bbox['y_min'],
#                 bbox['x_max'],
#                 bbox['y_max']
#             ))
            
#             return table_image
            
#         except Exception as e:
#             logger.error(f"Error extracting table image: {e}")
#             return None
    
#     def _image_to_base64(self, image: Image.Image) -> str:
#         """Convert PIL Image to base64 string for OpenAI API"""
#         buffered = io.BytesIO()
#         image.save(buffered, format="PNG")
#         return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
#     def _create_table_extraction_prompt(self, context_text: str, table_metadata: Dict[str, Any]) -> str:
#         """Create detailed prompt for table structure extraction"""
        
#         prompt = f"""You are an expert at extracting structured data from tables in documents. 

# ⚠️ CRITICAL: Extract EVERY row, column, and cell. NO OMISSIONS ALLOWED.
# ⚠️ This table contains financial, legal, or business-critical data. Missing even one cell is unacceptable.

# **CONTEXT:**
# Page Number: {table_metadata.get('page')}
# Table ID: {table_metadata.get('table_id')}
# Detection Source: {table_metadata.get('source', 'unknown')}

# **TEXT CONTEXT FROM PAGE (for reference):**
# {context_text[:3000]}

# **YOUR TASK:**
# Extract the COMPLETE table structure with ALL content. 

# **EXTRACTION REQUIREMENTS:**
# 1. ✓ Extract EVERY single row (don't skip any)
# 2. ✓ Extract EVERY single column (don't skip any)
# 3. ✓ Extract EVERY cell value exactly as shown
# 4. ✓ Preserve numbers, currency symbols, percentages, special characters
# 5. ✓ Identify column headers accurately
# 6. ✓ Handle merged cells appropriately
# 7. ✓ Maintain row-column alignment
# 8. ✓ Extract table title/caption if present
# 9. ✓ Preserve formatting indicators (bold, italic) in metadata if important
# 10. ✓ Handle multi-line cell content

# **WHAT TO PRESERVE:**
# - Exact numerical values (including decimals)
# - Currency symbols ($, €, £, etc.)
# - Percentages (%)
# - Dates and time formats
# - Special characters and punctuation
# - Empty cells (represent as empty string "")

# **OUTPUT FORMAT:**
# Return ONLY a valid JSON object (no markdown, no explanations):

# {{
#   "rows": <total number of DATA rows excluding headers>,
#   "columns": <total number of columns>,
#   "headers": ["Column 1 Header", "Column 2 Header", "Column 3 Header", ...],
#   "data": [
#     ["row1_col1", "row1_col2", "row1_col3", ...],
#     ["row2_col1", "row2_col2", "row2_col3", ...],
#     ["row3_col1", "row3_col2", "row3_col3", ...],
#     ...
#   ],
#   "metadata": {{
#     "title": "table title/caption or null",
#     "has_merged_cells": true/false,
#     "notes": "any additional context",
#     "table_type": "pricing/transition/financial/etc"
#   }}
# }}

# **EXAMPLE FOR FINANCIAL TABLE:**
# Input: A pricing table with headers "Role", "Rate USD", "Rate EUR" and 5 data rows

# Output:
# {{
#   "rows": 5,
#   "columns": 3,
#   "headers": ["Role", "Rate USD", "Rate EUR"],
#   "data": [
#     ["Software Engineer", "$120.00", "€105.00"],
#     ["Senior Engineer", "$150.00", "€132.00"],
#     ["Project Manager", "$180.00", "€158.00"],
#     ["QA Analyst", "$100.00", "€88.00"],
#     ["Database Admin", "$140.00", "€123.00"]
#   ],
#   "metadata": {{
#     "title": "Daily Rate Card",
#     "has_merged_cells": false,
#     "notes": null,
#     "table_type": "pricing"
#   }}
# }}

# **VALIDATION CHECKLIST:**
# Before returning, verify:
# ☐ Row count matches actual data rows in your output
# ☐ Column count matches actual columns in your output
# ☐ Every cell has a value (even if empty string)
# ☐ No rows or columns are skipped
# ☐ All numbers and special characters are preserved

# **CRITICAL REMINDER:** 
# This data may be used for financial calculations, legal agreements, or business decisions.
# Accuracy is MANDATORY. Extract 100% of the table content.

# Return ONLY the JSON object."""

#         return prompt
    
#     def _create_empty_table(self, table_metadata: Dict[str, Any]) -> FinalTable:
#         """Create empty table structure for error cases"""
#         logger.warning(f"Creating empty table for {table_metadata.get('table_id', 'unknown')}")
        
#         structure = TableStructure(
#             page=table_metadata.get('page', 0),
#             table_id=table_metadata.get('table_id', 'unknown'),
#             rows=0,
#             columns=0,
#             headers=[],
#             data=[],
#             metadata={"error": "Failed to extract table structure", "extraction_failed": True}
#         )
        
#         return FinalTable(
#             table_id=table_metadata.get('table_id', 'unknown'),
#             page=table_metadata.get('page', 0),
#             structure=structure,
#             source=table_metadata.get('source', 'unknown'),
#             confidence=0.0
#         )