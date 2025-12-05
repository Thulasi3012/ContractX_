"""
Text Analysis using Gemini
Extracts sections, clauses, entities, and page summary (TEXT ONLY - no tables)
"""

import google.generativeai as genai
import os
import json
import asyncio
from typing import Dict, Any
from app.config.config import Config

class TextAnalyzer:
    """Analyze document text for structure, entities, and summary"""
    
    def __init__(self):
        api_key = Config.GEMINI_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set!")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        self.request_delay = float(os.getenv("GEMINI_REQUEST_DELAY", "3.0"))
        self.max_retries = int(os.getenv("MAX_RETRIES", "5"))
        self.retry_delay = int(os.getenv("RETRY_DELAY", "60"))
        
        print("[OK] TextAnalyzer initialized")
    
    async def analyze_text(self, page_number: int, page_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze text content with page summary (no tables)
        
        Returns structure with sections, entities, and summary
        """
        text = page_data['text']
        
        for attempt in range(1, self.max_retries + 1):
            try:
                await asyncio.sleep(self.request_delay)
                
                prompt = self._create_prompt(page_number, text)
                response = self.model.generate_content(prompt)
                result_text = response.text.strip()
                
                # Clean JSON
                if result_text.startswith("```json"):
                    result_text = result_text[7:]
                if result_text.startswith("```"):
                    result_text = result_text[3:]
                if result_text.endswith("```"):
                    result_text = result_text[:-3]
                
                data = json.loads(result_text.strip())
                
                # Add metadata
                data['page_number'] = page_number
                data['sections_count'] = len(data.get('sections', []))
                
                # Ensure summary exists
                if not data.get('summary'):
                    data['summary'] = self._generate_fallback_summary(text, data)
                
                return data
                
            except json.JSONDecodeError as e:
                print(f"    [!] JSON error on page {page_number} (attempt {attempt})")
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** (attempt - 1)))
                else:
                    return self._empty_result(page_number, text)
            
            except Exception as e:
                error_msg = str(e).lower()
                
                if any(kw in error_msg for kw in ['quota', 'rate limit', '429', 'resource exhausted']):
                    print(f"    [!] Quota error on page {page_number} (attempt {attempt})")
                    if attempt < self.max_retries:
                        wait = self.retry_delay * 3 * (2 ** (attempt - 1))
                        print(f"    [i] Waiting {wait}s...")
                        await asyncio.sleep(wait)
                    else:
                        return self._empty_result(page_number, text)
                else:
                    print(f"    [!] Error on page {page_number}: {e}")
                    if attempt < self.max_retries:
                        await asyncio.sleep(self.retry_delay * (2 ** (attempt - 1)))
                    else:
                        return self._empty_result(page_number, text)
        
        return self._empty_result(page_number, text)
    
    def _create_prompt(self, page_number: int, text: str) -> str:
        """Create detailed prompt for Gemini - TEXT ONLY EXTRACTION"""
        
        prompt = f"""You are an expert document analyzer specializing in legal contracts and business documents. 

⚠️ CRITICAL INSTRUCTION: Analyze ONLY the plain text, paragraphs, and textual content. 
⚠️ DO NOT extract or process content from TABLES or IMAGES.
⚠️ Simply DETECT if tables/images are present, but don't extract their content.

PAGE {page_number}:
{text[:5000]}

**YOUR TASK:**

1. **Extract Document Structure (TEXT ONLY):**
   - Identify headings, sub-headings, clauses, and sub-clauses from PLAIN TEXT
   - SKIP any content that appears in table format
   - Assign hierarchical IDs (heading_id: 1, sub_heading_id: 1.1, clause_id: 1.1.1)
   - Preserve exact text from paragraphs and text blocks only

2. **Detect Tables (DO NOT EXTRACT CONTENT):**
   - DO NOT extract table data, headers, or cell contents
   - don't describe table content
   
3. **Detect Images (DO NOT EXTRACT CONTENT):**
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
- headers and footers that are part of page layout

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
  "summary": "A detailed summary of the document text only who is the buyer, seller, important dates, deadlines mentioned in the text and is there any risk or objection level mentioned in the text and amount of tables present in the document or any important alert mentioned in the text. **note** the summary should be in text format only not in json format and short and precise",
  "entities": {{
    "document_type": "Type of document (Contract, Agreement, Invoice, Lease, Purchase Order, Loan Agreement, etc.) extracted from TEXT ONLY",
    
    "buyer_name": "Buyer / Client / Purchaser extracted from TEXT ONLY or null",
    "seller_name": "Seller / Provider / Contractor extracted from TEXT ONLY or null",
    
    "addresses": ["All postal addresses mentioned in the text"],
    "dates": ["All important dates found in the text"],
    "deadlines": ["All deadlines, due dates, expiry dates"],
    
    "alerts": ["All risk points or critical issues from the document"],
    
    "obligations": [
        {{
            "party": "Buyer / Seller / Third Party",
            "description": "The obligation / responsibility extracted from the text",
            "page": "page number if available"
        }}
    ],
    
    "payment_terms": "Extract payment terms if available",
    "contract_effective_date": "Start date of contract if available",
    "contract_end_date": "End/termination date if available"
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
    
    def _generate_fallback_summary(self, text: str, data: Dict[str, Any]) -> str:
        """Generate a basic summary if API doesn't provide one"""
        entities = data.get('entities', {})
        sections = data.get('sections', [])
        
        summary_parts = []
        
        # Parties
        buyer = entities.get('buyer_name')
        seller = entities.get('seller_name')
        if buyer and seller:
            summary_parts.append(f"Agreement between buyer {buyer} and seller {seller}.")
        elif buyer:
            summary_parts.append(f"Buyer: {buyer}.")
        elif seller:
            summary_parts.append(f"Seller: {seller}.")
        
        # Dates
        dates = entities.get('dates', [])
        if dates:
            summary_parts.append(f"Key dates: {', '.join(dates[:3])}.")
        
        # Deadlines
        deadlines = entities.get('deadlines', [])
        if deadlines:
            summary_parts.append(f"Deadlines: {', '.join(deadlines[:2])}.")
        
        # Sections
        if sections:
            summary_parts.append(f"Contains {len(sections)} main sections.")
        
        # Alerts
        alerts = entities.get('alerts', [])
        if alerts:
            summary_parts.append(f"Important alerts: {', '.join(alerts[:2])}.")
        
        # Risk level
        risk = entities.get('objection_level')
        if risk:
            summary_parts.append(f"Risk level: {risk}.")
        
        return ' '.join(summary_parts) if summary_parts else "Page contains textual content."
    
    def _empty_result(self, page_number: int, text: str = "") -> Dict[str, Any]:
        """Empty result on failure with fallback summary"""
        return {
            "page_number": page_number,
            "sections": [],
            "sections_count": 0,
            "summary": f"Page {page_number}: Analysis failed. Text preview: {text[:100]}..." if text else f"Page {page_number}: Analysis failed.",
            "entities": {
                "buyer_name": None,
                "seller_name": None,
                "objection_level": None,
                "dates": [],
                "deadlines": [],
                "alerts": [],
                "addresses": []
            },
            "error": "Analysis failed"
        }