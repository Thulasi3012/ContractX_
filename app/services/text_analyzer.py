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
        """
        Create detailed prompt for Gemini
        NO HALLUCINATED HEADINGS - NO FAKE IDs - NO BROKEN ADDRESSES
        """
        
        prompt = f"""You are a precision document extraction system. Extract EXACTLY what is in the PDF.

PAGE {page_number}:
{text[:6000]}

═════════════════════════════════════════════════════════════════════════════════

⚠️ CRITICAL RULES - FOLLOW 100% STRICTLY:

1. **Extract ALL Headings (including small/footer text):**
   ✓ Main section headings (bold, large)
   ✓ Sub-section headings (smaller)
   ✓ Page headers ("Confidential", "Page X of Y")
   ✓ Contact email lines (e.g., "christopher.tapping@thomsonreuters.com")
   ✓ Footer text lines
   ✓ Standalone single-line text boxes
   ✓ Tiny/small font headings that might be easy to miss
   ✗ DO NOT skip any heading just because it's small

2. **Extract ALL Lines of Text Without Skipping:**
   ✓ Long paragraphs
   ✓ Short one-line statements (e.g., "Thank you.")
   ✓ Small font text
   ✓ Isolated footer/header lines
   ✓ Contact information
   ✓ Disclaimers and notes
   ✗ NEVER omit text because it appears small or isolated

3. **Handle Bullet Lists and Enumerations:**
   ✓ Each bullet point = separate item (NOT merged into one heading)
   ✓ Example: If text shows:
      • Alteryx
      • Experian
      • K-1 Analyzer
     Then extract 3 separate sub_headings, NOT one combined heading
   ✓ Preserve exact bullet formatting
   ✗ NEVER merge multiple bullets into one

4. **Preserve Numeric Values EXACTLY:**
   ✓ Extract numbers, Ammount ,Price, IDs, order codes EXACTLY as printed
   ✓ Example: Order ID "Q-01020320" must be "Q-01020320" not "Q-01020330"
   ✓ If unsure, leave empty—NEVER guess or modify numbers
   ✗ NEVER hallucinate or change numeric values

5. **Extract Text Content EXACTLY:**
   ✓ Preserve exact wording, spacing, capitalization, punctuation
   ✓ Keep abbreviations as-is
   ✓ Keep acronyms as-is
   ✗ NEVER reword, simplify, or modify text

6. **Table Detection (DO NOT EXTRACT TABLE DATA):**
   ✓ Detect if tables exist on page
   ✓ Count tables
   ✗ DO NOT extract table cell contents, headers, or row data

7. **Entity Extraction (FROM TEXT ONLY):**
   ✓ Buyer/Seller names from paragraphs
   ✓ Dates from text (e.g., "3/27/2021", "January 1, 2024")
   ✓ Deadlines from text (e.g., "within 12 months","Payment due within 30 days")
   ✓ Addresses from text paragraphs
   ✓ Email addresses from text
   ✓ Phone numbers from text

8. **Image/Visual Detection:**
   ✓ Only mark as visual if it contains actual image/diagram content
   ✗ NEVER mark plain text boxes as visuals
   ✗ Text that appears in boxes is still TEXT, not visual 

9. **HEADINGS vs PARAGRAPHS - STRICT DISTINCTION:**
   ✓ HEADING: Bold/large text that acts as a section title
     • "Account Address"
     • "Terms and Conditions"
     • "Miscellaneous"
   ✓ PARAGRAPH/CONTENT: Normal sentences and continuous text
     • "This Order Form, once accepted by Thomson Reuters..."
     • Address blocks (multi-line)
     • Descriptions and explanations
   ✗ NEVER classify normal text as a heading
   ✗ NEVER split an address block into multiple fake headings

10. **ADDRESS BLOCKS - KEEP THEM TOGETHER:**
   ✓ If you see a multi-line address block:
     Account #: 12345
     Huafon INTL Biomaterials LLC
     1105 N MARKET ST FL 11
     WILMINGTON, DE 19801-1216 US
   
   CORRECT OUTPUT:
   {{
     "heading": "Account Address",
     "content": "Account #: 12345\\nHuafon INTL Biomaterials LLC\\n1105 N MARKET ST FL 11\\nWILMINGTON, DE 19801-1216 US"
   }}
   
   ✗ WRONG (DO NOT DO THIS):
   {{
     "sub_heading": "Huafon INTL Biomaterials LLC",
     "sub_heading_id": "1.1"
   }},
   {{
     "sub_heading": "1105 N MARKET ST FL 11",
     "sub_heading_id": "1.2"
   }}
   
   ✗ NEVER break an address into multiple sub-headings
   ✗ NEVER assign fake IDs to address lines

11. **HEADING IDs - ONLY IF EXPLICITLY IN DOCUMENT:**
   ✓ Extract heading_id ONLY if the document explicitly shows numbering like:
     "1. Introduction"
     "2. Terms"
     "3. Conditions"
   ✓ Preserve exact numbering as shown
   
   CORRECT OUTPUT:
   {{
     "heading": "Introduction",
     "heading_id": "1",
     "content": "..."
   }}
   
   ✗ WRONG (DO NOT AUTO-GENERATE):
   {{
     "heading": "Introduction",
     "heading_id": "1",  ← HALLUCINATED! Document has no numbering
     "content": "..."
   }}
   
   ✓ IF NO NUMBERING IN DOCUMENT: Omit heading_id entirely
   {{
     "heading": "Introduction",
     "content": "..."
   }}
*NOTE* : Do NOT classify "Page X of Y" as a heading.  
Treat it as footer text and place it under "content" of the nearest section.

12. **NO SUB-HEADINGS WITHOUT VISUAL SEPARATION:**
   ✓ Use sub-headings ONLY if there are clearly separate, visually distinct sections
   ✓ Each sub-heading should be bold/indented/visually distinct
   *Example:* 1.1 Service A
               1.2 Service B
               1.3 Service C
   
   ✗ NEVER create sub-headings from:
     • Bullet points that are just list items
     • Names in an address block
     • Sequential lines of continuous text
     • Part of a longer paragraph
   
   ✓ CORRECT - Bullet items should be in "content" under a single heading:
   {{
     "heading": "Services Provided",
     "content": "• Service A\\n• Service B\\n• Service C"
   }}
   
   ✗ WRONG - DO NOT CREATE SUB-HEADINGS:
   {{
     "sub_heading": "Service A",
     "sub_heading_id": "1.1"
   }},
   {{
     "sub_heading": "Service B",
     "sub_heading_id": "1.2"
   }}

13. **CLAUSES - ONLY ACTUAL CONTRACT LANGUAGE:**
   ✓ Extract clauses from legal contract sections
   ✓ Clauses should be substantial legal text, not single lines
   Example:
   clause_id: "2.1"
   clause_content : "The Buyer agrees to pay the Seller within 30 days of invoice receipt..."

   ✗ DO NOT create clauses from:
     • Addresses
     • Names
     • Contact information
     • Single descriptive lines

═════════════════════════════════════════════════════════════════════════════════

**OUTPUT FORMAT - NO HALLUCINATIONS:**

Return ONLY valid JSON (no markdown):

{{
  "sections": [
    {{
      "heading": "Section Title OR null",
      "content": "Full text content (addresses, paragraphs, etc.)",
      "subsections": [
        {{
          "heading": "Sub-section title (only if visually distinct)",
          "content": "Content text"
        }}
      ],
      "clauses": [
        {{
          "clause_id": "1.1 (only if in document)",
          "clause": "Full clause text from contract"
        }}
      ]
    }}
  ],
  "summary": "Concise summary: Who is buyer/seller, important dates, deadlines, alerts, and count of tables. Keep to 2-3 sentences max.",
  "entities": {{
    "document_type": "Type (Order Form, Agreement, Invoice, etc.) from TEXT",
    "buyer_name": "Buyer/Client name from text or null",
    "seller_name": "Seller/Provider name from text or null",
    "addresses": ["All postal addresses from text"],
    "dates": ["All dates mentioned in text (e.g., '3/27/2021')"],
    "deadlines": ["All deadlines/due dates from text"],
    "alerts": ["Risk points, important warnings from text"],
    "obligations": [
      {{
        "party": "Who has obligation",
        "description": "What they must do"
      }}
    ],
    "payment_terms": "Payment text or null",
    "contract_effective_date": "Start date or null",
    "contract_end_date": "End date or null"
  }}
}}

═════════════════════════════════════════════════════════════════════════════════

**CRITICAL EXAMPLES:**


❌ WRONG - Hallucinated heading IDs and broken address:
{{
  "heading": "Account Address",
  "heading_id": "2",  ← NO, not in document
  "subsections": [
    {{
      "heading": "Huafon INTL Biomaterials LLC",
      "heading_id": "2.1"  ← NO, this is an address line!
    }},
    {{
      "heading": "1105 N MARKET ST FL 11",
      "heading_id": "2.2"  ← NO, hallucinated!
    }}
  ]
}}

✅ CORRECT - Keep address together, no fake IDs:
{{
  "heading": "Account Address",
  "content": "Account #: 12345\\nHuafon INTL Biomaterials LLC\\n1105 N MARKET ST FL 11\\nWILMINGTON, DE 19801-1216 US"
}}

❌ WRONG - Paragraph classified as heading:
{{
  "heading": "This Order Form, once accepted by Thomson Reuters...",
  "content": ""
}}

✅ CORRECT - Paragraph should be content, not heading:
{{
  "heading": null,
  "content": "This Order Form, once accepted by Thomson Reuters, constitutes a binding agreement..."
}}

❌ WRONG - Bullet items as fake sub-headings:
{{
  "subsections": [
    {{"heading": "Alteryx", "heading_id": "1.1"}},
    {{"heading": "Experian", "heading_id": "1.2"}},
    {{"heading": "K-1 Analyzer", "heading_id": "1.3"}}
  ]
}}

✅ CORRECT - Bullets in content:
{{
  "heading": "Available Data Sets",
  "content": "• Alteryx\\n• Experian\\n• K-1 Analyzer"
}}

═════════════════════════════════════════════════════════════════════════════════

NOW: Extract this page EXACTLY following ALL rules above.
NEVER hallucinate headings, IDs, or structure.
NEVER break addresses or names into fake sub-headings.
Return ONLY the JSON object."""

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
            summary_parts.append(f"Agreement between {buyer} (buyer) and {seller} (seller).")
        elif buyer:
            summary_parts.append(f"Document for buyer: {buyer}.")
        elif seller:
            summary_parts.append(f"Document from seller: {seller}.")
        
        # Dates
        dates = entities.get('dates', [])
        if dates:
            date_str = ', '.join([str(d) for d in dates if d][:3])
            summary_parts.append(f"Key dates: {date_str}.")
        
        # Deadlines
        deadlines = entities.get('deadlines', [])
        if deadlines:
            deadline_str = ', '.join([str(d) for d in deadlines if d][:2])
            summary_parts.append(f"Deadlines: {deadline_str}.")
        
        # Sections
        if sections:
            summary_parts.append(f"Contains {len(sections)} main sections.")
        
        # Alerts
        alerts = entities.get('alerts', [])
        if alerts:
            alert_str = ', '.join([str(a) for a in alerts if a][:2])
            summary_parts.append(f"Important alerts: {alert_str}.")
        
        return ' '.join(summary_parts) if summary_parts else "Page contains document text."
    
    def _empty_result(self, page_number: int, text: str = "") -> Dict[str, Any]:
        """Empty result on failure with fallback summary"""
        return {
            "page_number": page_number,
            "sections": [],
            "sections_count": 0,
            "summary": f"Page {page_number}: Analysis failed. Preview: {text[:100]}..." if text else f"Page {page_number}: Analysis failed.",
            "entities": {
                "document_type": None,
                "buyer_name": None,
                "seller_name": None,
                "dates": [],
                "deadlines": [],
                "alerts": [],
                "obligations": [],
                "addresses": [],
                "payment_terms": None,
                "contract_effective_date": None,
                "contract_end_date": None
            },
            "error": "Analysis failed"
        }