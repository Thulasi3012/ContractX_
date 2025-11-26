import google.generativeai as genai
import os
import json
import time
import asyncio
from typing import Dict, Any
from app.database.schemas import Chunk, TextLLMOutput, Section, TableInfo, ImageInfo, Entities
import logging
from datetime import datetime

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/text_llm_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TextLLMService:
    """Processes document chunks using Gemini 2.5 Flash for structure extraction (TEXT ONLY - NO TABLES/IMAGES)"""
    
    def __init__(self):
        # Configure Gemini API
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Rate limiting configuration
        self.request_delay = float(os.getenv("GEMINI_REQUEST_DELAY", "2.0"))  # 2 seconds between requests
        self.max_retries = int(os.getenv("MAX_RETRIES", "5"))
        self.retry_delay = int(os.getenv("RETRY_DELAY", "60"))  # 60 seconds initial retry delay
        self.exponential_backoff = True
        
        logger.info("="*80)
        logger.info("TextLLMService Initialized")
        logger.info(f"Model: gemini-2.0-flash-exp")
        logger.info(f"Request Delay: {self.request_delay}s")
        logger.info(f"Max Retries: {self.max_retries}")
        logger.info(f"Retry Delay: {self.retry_delay}s")
        logger.info("="*80)
    
    async def process_chunk(self, chunk: Chunk) -> TextLLMOutput:
        """
        Process a document chunk to extract structure and entities (TEXT ONLY)
        
        Args:
            chunk: Document chunk with text and metadata
            
        Returns:
            Structured output with sections, entities, and table/image indicators
        """
        chunk_id = chunk.metadata.chunk_id
        page_range = f"{chunk.metadata.page_start}-{chunk.metadata.page_end}"
        
        logger.info("="*80)
        logger.info(f"Processing Chunk {chunk_id} (Pages {page_range})")
        logger.info("="*80)
        
        # Apply rate limiting
        logger.info(f"Applying rate limit: waiting {self.request_delay}s before API call...")
        await asyncio.sleep(self.request_delay)
        
        # Create detailed prompt for structure extraction
        prompt = self._create_prompt(chunk)
        
        # Retry logic with exponential backoff
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Attempt {attempt}/{self.max_retries} - Sending request to Gemini API")
                start_time = time.time()
                
                # Generate response from Gemini
                response = self.model.generate_content(prompt)
                
                elapsed_time = time.time() - start_time
                logger.info(f"✓ API Response received in {elapsed_time:.2f}s")
                
                # Parse JSON response
                result_text = response.text.strip()
                logger.info(f"Response length: {len(result_text)} characters")
                
                # Clean markdown code blocks if present
                if result_text.startswith("```json"):
                    result_text = result_text[7:]
                    logger.info("Removed ```json prefix")
                if result_text.startswith("```"):
                    result_text = result_text[3:]
                    logger.info("Removed ``` prefix")
                if result_text.endswith("```"):
                    result_text = result_text[:-3]
                    logger.info("Removed ``` suffix")
                
                result_data = json.loads(result_text.strip())
                logger.info("✓ JSON parsed successfully")
                
                # Log extracted data summary
                sections_count = len(result_data.get('sections', []))
                tables_detected = result_data.get('tables', {}).get('contains_table', False)
                images_detected = result_data.get('images', {}).get('contains_image', False)
                
                logger.info(f"Extracted Data Summary:")
                logger.info(f"  - Sections: {sections_count}")
                logger.info(f"  - Tables Detected: {tables_detected}")
                logger.info(f"  - Images Detected: {images_detected}")
                logger.info(f"  - Buyer: {result_data.get('entities', {}).get('buyer_name', 'Not found')}")
                logger.info(f"  - Seller: {result_data.get('entities', {}).get('seller_name', 'Not found')}")
                
                # Convert to TextLLMOutput model
                output = TextLLMOutput(
                    sections=result_data.get('sections', []),
                    tables=TableInfo(**result_data.get('tables', {})),
                    images=ImageInfo(**result_data.get('images', {})),
                    entities=Entities(**result_data.get('entities', {})),
                    chunk_metadata=chunk.metadata
                )
                
                logger.info(f"✓ Chunk {chunk_id} processed successfully")
                logger.info("="*80)
                return output
                
            except json.JSONDecodeError as e:
                logger.error(f"✗ JSON parsing error for chunk {chunk_id} (Attempt {attempt})")
                logger.error(f"Error: {str(e)}")
                logger.error(f"Response text (first 500 chars): {result_text[:500]}")
                
                if attempt < self.max_retries:
                    retry_wait = self._calculate_retry_delay(attempt)
                    logger.warning(f"Retrying in {retry_wait}s...")
                    await asyncio.sleep(retry_wait)
                else:
                    logger.error(f"✗ Max retries reached for chunk {chunk_id}")
                    return self._create_empty_output(chunk)
            
            except Exception as e:
                error_msg = str(e)
                logger.error(f"✗ Error processing chunk {chunk_id} (Attempt {attempt})")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Error message: {error_msg}")
                
                # Check for quota/rate limit errors
                if any(keyword in error_msg.lower() for keyword in ['quota', 'rate limit', 'resource exhausted', '429']):
                    logger.warning("⚠ QUOTA/RATE LIMIT ERROR DETECTED")
                    
                    if attempt < self.max_retries:
                        retry_wait = self._calculate_retry_delay(attempt, is_quota_error=True)
                        logger.warning(f"Pausing for {retry_wait}s before retry...")
                        await asyncio.sleep(retry_wait)
                    else:
                        logger.error(f"✗ Max retries reached for chunk {chunk_id} after quota errors")
                        return self._create_empty_output(chunk)
                else:
                    # Non-quota error
                    if attempt < self.max_retries:
                        retry_wait = self._calculate_retry_delay(attempt)
                        logger.warning(f"Retrying in {retry_wait}s...")
                        await asyncio.sleep(retry_wait)
                    else:
                        logger.error(f"✗ Max retries reached for chunk {chunk_id}")
                        return self._create_empty_output(chunk)
        
        # Should not reach here, but return empty output as fallback
        logger.error(f"✗ Unexpected state - returning empty output for chunk {chunk_id}")
        return self._create_empty_output(chunk)
    
    def _calculate_retry_delay(self, attempt: int, is_quota_error: bool = False) -> int:
        """Calculate retry delay with exponential backoff"""
        if is_quota_error:
            # Longer delays for quota errors
            base_delay = self.retry_delay * 2
        else:
            base_delay = self.retry_delay
        
        if self.exponential_backoff:
            delay = base_delay * (2 ** (attempt - 1))
            logger.info(f"Exponential backoff: attempt {attempt} -> {delay}s delay")
        else:
            delay = base_delay
            logger.info(f"Fixed delay: {delay}s")
        
        return min(delay, 600)  # Cap at 10 minutes
    
    def _create_prompt(self, chunk: Chunk) -> str:
        """Create detailed prompt for Gemini - TEXT ONLY EXTRACTION"""
        
        prompt = f"""You are an expert document analyzer specializing in legal contracts and business documents. 

⚠️ CRITICAL INSTRUCTION: Analyze ONLY the plain text, paragraphs, and textual content. 
⚠️ DO NOT extract or process content from TABLES or IMAGES.
⚠️ Simply DETECT if tables/images are present, but don't extract their content.

**DOCUMENT CHUNK (Pages {chunk.metadata.page_start}-{chunk.metadata.page_end}):**

{chunk.text}

**YOUR TASK:**

1. **Extract Document Structure (TEXT ONLY):**
   - Identify headings, sub-headings, clauses, and sub-clauses from PLAIN TEXT
   - SKIP any content that appears in table format
   - Assign hierarchical IDs (heading_id: 1, sub_heading_id: 1.1, clause_id: 1.1.1)
   - Preserve exact text from paragraphs and text blocks only

2. **Detect Tables (DO NOT EXTRACT CONTENT):**
   - Simply identify IF tables are present in the document
   - Note page numbers where tables appear
   - Count how many tables you see
   - DO NOT extract table data, headers, or cell contents
   - Just answer: Are there tables? Yes/No, Which pages?, How many?

3. **Detect Images (DO NOT EXTRACT CONTENT):**
   - Identify references to images, figures, charts, or diagrams
   - Note page numbers where images appear or are mentioned
   - Count total references
   - DO NOT describe image content

4. **Extract Entities (FROM TEXT ONLY):**
   - Buyer name(s) and seller name(s) from paragraphs
   - All dates mentioned in text
   - Important deadlines from text
   - Addresses from text
   - Alert items or critical clauses from text
   - Objection level or risk level from text
   - DO NOT extract entities from within tables

**WHAT TO SKIP:**
- Any content organized in rows and columns (tables)
- Any content within table structures
- Image descriptions or captions within images
- Numerical data organized in tabular format

**WHAT TO EXTRACT:**
- Headings and section titles
- Paragraph text
- Numbered or bulleted lists (as text)
- Definitions and clauses written as prose
- Names, dates, and entities from plain text

**OUTPUT FORMAT:**

Return ONLY a valid JSON object (no markdown, no explanations):

{{
  "sections": [
    {{
      "heading": "Section heading text from paragraphs",
      "heading_id": "1",
      "sub_headings": [
        {{
          "sub_heading": "Sub-heading text from paragraphs",
          "sub_heading_id": "1.1",
          "clauses": [
            {{
              "clause": "Clause text from paragraphs",
              "clause_id": "1.1.1",
              "sub_clauses": [
                {{
                  "sub_clause": "Sub-clause text",
                  "sub_clause_id": "1.1.1.1"
                }}
              ]
            }}
          ]
        }}
      ]
    }}
  ],
  "tables": {{
    "contains_table": true or false,
    "pages": [list of page numbers where tables exist],
    "count": number of tables detected (just count, don't extract)
  }},
  "images": {{
    "contains_image": true or false,
    "pages": [list of page numbers],
    "count": number of images
  }},
  "entities": {{
    "buyer_name": "extracted from TEXT ONLY or null",
    "seller_name": "extracted from TEXT ONLY or null",
    "objection_level": "risk level from TEXT or null",
    "dates": ["date1 from text", "date2 from text"],
    "alerts": ["critical item 1 from text", "critical item 2"],
    "deadlines": ["deadline 1 from text", "deadline 2"],
    "addresses": ["address 1 from text", "address 2 from text"]
  }}
}}

**EXAMPLE:**

If the document has:
- A title "SERVICE AGREEMENT" 
- A paragraph "This agreement is between Buyer Corp and Seller Inc effective January 1, 2024"
- A pricing table with 10 rows
- An appendix section

Your output should extract:
- The title and paragraph text
- Detect that a table exists (contains_table: true, count: 1)
- Extract "Buyer Corp" and "Seller Inc" from the paragraph
- Extract "January 1, 2024" from the paragraph
- Extract appendix section heading
- BUT NOT extract any data from the pricing table itself

**CRITICAL REMINDER:** 
- Extract structure and entities from TEXT ONLY
- Do NOT process table content
- Do NOT process image content
- Just DETECT if tables/images exist
- Return ONLY the JSON object"""

        return prompt
    
    def _create_empty_output(self, chunk: Chunk) -> TextLLMOutput:
        """Create empty output structure for error cases"""
        logger.warning(f"Creating empty output for chunk {chunk.metadata.chunk_id}")
        return TextLLMOutput(
            sections=[],
            tables=TableInfo(contains_table=False, pages=[], count=0),
            images=ImageInfo(contains_image=False, pages=[], count=0),
            entities=Entities(),
            chunk_metadata=chunk.metadata
        )