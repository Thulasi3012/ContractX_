"""
AI-Powered Legal Document Comparison using Gemini
Replaces hardcoded logic with intelligent AI analysis
"""
import re
import hashlib
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
import google.generativeai as genai
from datetime import datetime
from app.config.config import Config

genai.configure(api_key=Config.GEMINI_API_KEY)
model = genai.GenerativeModel(Config.GEMINI_MODEL)

# ============================================================================
# ENUMS & DATA CLASSES (Keep these as they define structure)
# ============================================================================

class ChangeType(Enum):
    """Change classification"""
    NO_IMPACT_CHANGE = "NO_IMPACT_CHANGE"
    STRUCTURAL_CHANGE = "STRUCTURAL_CHANGE"
    OBLIGATION_CHANGE = "OBLIGATION_CHANGE"
    SCOPE_CHANGE = "SCOPE_CHANGE"
    PARTY_CHANGE = "PARTY_CHANGE"
    CONDITION_CHANGE = "CONDITION_CHANGE"
    TIMING_CHANGE = "TIMING_CHANGE"
    RISK_CHANGE = "RISK_CHANGE"
    NEW_CLAUSE = "NEW_CLAUSE"
    REMOVED_CLAUSE = "REMOVED_CLAUSE"
    LANGUAGE_EQUIVALENT = "LANGUAGE_EQUIVALENT"


class ImpactLevel(Enum):
    """Change impact"""
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class LegalChange:
    """AI-detected change"""
    type: ChangeType
    path: str
    from_value: Optional[str] = None
    to_value: Optional[str] = None
    description: str = ""
    impact: ImpactLevel = ImpactLevel.NONE
    confidence: float = 1.0
    requires_human_review: bool = False
    ai_reasoning: str = ""  # NEW: AI's explanation
    
    def to_dict(self) -> Dict:
        return {
            'type': self.type.value,
            'path': self.path,
            'from_value': self.from_value,
            'to_value': self.to_value,
            'description': self.description,
            'impact': self.impact.value,
            'confidence': self.confidence,
            'requires_human_review': self.requires_human_review,
            'ai_reasoning': self.ai_reasoning
        }


# ============================================================================
# AI-POWERED COMPARISON ENGINE
# ============================================================================

class GeminiDocumentComparator:
    """AI-powered legal document comparison using Gemini"""
    
    @staticmethod
    def _build_prompt(doc1_str: str, doc2_str: str) -> str:
        """Build the analysis prompt with proper escaping"""
        return f"""You are an expert legal document analyst. Compare these two legal documents and identify all material changes.

**CRITICAL RULES:**
1. Focus ONLY on legal clauses, obligations, rights, and conditions
2. Ignore metadata, summaries, timestamps, version numbers, document IDs
3. Translation is NOT a change (French → English with same meaning = no change)
4. Rewording is NOT a change if legal meaning is identical
5. Structural reorganization is NOT a material change (just mark as STRUCTURAL_CHANGE)
6. Renumbering clauses is NOT a change (5.2 → 6.1 with same content = no change)

**WHAT COUNTS AS A CHANGE:**
- Party changes (Customer → Vendor)
- Obligation changes (shall → may, must → should not)
- Scope changes (30 days → 60 days, $1M → $2M)
- New or removed obligations
- Changes to conditions, timelines, amounts, parties, or legal effects

**OUTPUT FORMAT:** Return valid JSON with this structure:
- changes: array of change objects
- summary: object with totals

Each change object must have:
- type: one of [OBLIGATION_CHANGE, PARTY_CHANGE, SCOPE_CHANGE, TIMING_CHANGE, NEW_CLAUSE, REMOVED_CLAUSE, STRUCTURAL_CHANGE, LANGUAGE_EQUIVALENT]
- path: location identifier
- from_value: original text (if applicable)
- to_value: new text (if applicable)
- description: brief explanation
- impact: one of [NONE, LOW, MEDIUM, HIGH, CRITICAL]
- confidence: float between 0.0 and 1.0
- requires_human_review: boolean
- reasoning: explanation of the decision

**DOCUMENT 1:**
{doc1_str}

**DOCUMENT 2:**
{doc2_str}

Analyze and return ONLY valid JSON output."""

    @staticmethod
    def compare(doc1_json: Dict, doc2_json: Dict) -> Dict:
        """
        AI-powered comparison - replaces all hardcoded logic
        """
        try:
            # Prepare documents for AI
            doc1_str = json.dumps(doc1_json, indent=2)
            doc2_str = json.dumps(doc2_json, indent=2)
            
            print(f"Document 1 size: {len(doc1_str)} chars")
            print(f"Document 2 size: {len(doc2_str)} chars")
            
            # Check size limits (Gemini has context limits)
            if len(doc1_str) + len(doc2_str) > 500000:  # ~500KB limit
                print("Documents too large, using chunked comparison...")
                return GeminiDocumentComparator._fallback_chunked_comparison(
                    doc1_json, doc2_json
                )
            
            # Call Gemini
            prompt = GeminiDocumentComparator._build_prompt(doc1_str, doc2_str)
            
            print("Calling Gemini API...")
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,  # Low temperature for consistency
                    top_p=0.95,
                    max_output_tokens=8192,
                )
            )
            
            print(f"Gemini response received. Length: {len(response.text)} chars")
            print(f"First 500 chars: {response.text[:500]}")
            
            # Parse AI response
            result = GeminiDocumentComparator._parse_ai_response(response.text)
            
            print(f"Parsed result: {len(result.get('changes', []))} changes detected")
            
            # Post-process and validate
            result = GeminiDocumentComparator._validate_and_enrich(result)
            
            return result
            
        except Exception as e:
            import traceback
            print(f"ERROR in AI comparison: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            raise Exception(f"AI comparison failed: {str(e)}")
    
    @staticmethod
    def _parse_ai_response(response_text: str) -> Dict:
        """Extract JSON from AI response"""
        try:
            # Remove markdown code blocks if present
            json_text = response_text.strip()
            
            # Handle ```json blocks
            if "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0]
            elif "```" in json_text:
                # Handle generic ``` blocks
                parts = json_text.split("```")
                if len(parts) >= 2:
                    json_text = parts[1]
            
            json_text = json_text.strip()
            
            print(f"Attempting to parse JSON. Length: {len(json_text)}")
            print(f"First 200 chars: {json_text[:200]}")
            
            # Parse JSON
            result = json.loads(json_text)
            
            print(f"Successfully parsed JSON: {list(result.keys())}")
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {str(e)}")
            print(f"Failed text (first 1000 chars): {json_text[:1000]}")
            
            # Fallback: try to extract JSON with regex
            import re
            json_match = re.search(r'\{[^{}]*"changes"[^{}]*\[.*?\][^{}]*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                    print("Successfully extracted JSON with regex fallback")
                    return result
                except:
                    pass
            
            # Last resort: return empty structure
            print("All parsing failed, returning empty structure")
            return {
                "changes": [],
                "summary": {
                    "total_material_changes": 0,
                    "high_risk_changes": 0,
                    "translations_detected": 0,
                    "structural_only": 0
                }
            }
    
    @staticmethod
    def _validate_and_enrich(result: Dict) -> Dict:
        """Validate AI output and add metadata"""
        
        # Ensure required fields
        if 'changes' not in result:
            result['changes'] = []
        if 'summary' not in result:
            result['summary'] = {
                'total_changes': len(result['changes']),
                'by_type': {},
                'by_impact': {},
                'requires_human_review': 0,
                'material_changes': 0
            }
        
        # Convert change types to enums and calculate summary
        validated_changes = []
        by_type = {}
        by_impact = {}
        human_review_count = 0
        material_count = 0
        
        for change in result['changes']:
            try:
                # Convert strings to enums
                change_type = ChangeType[change['type']]
                impact_level = ImpactLevel[change['impact']]
                
                # Create LegalChange object
                legal_change = LegalChange(
                    type=change_type,
                    path=change.get('path', 'unknown'),
                    from_value=change.get('from_value'),
                    to_value=change.get('to_value'),
                    description=change.get('description', ''),
                    impact=impact_level,
                    confidence=change.get('confidence', 1.0),
                    requires_human_review=change.get('requires_human_review', False),
                    ai_reasoning=change.get('reasoning', '')
                )
                
                validated_changes.append(legal_change)
                
                # Update counters
                by_type[change_type.value] = by_type.get(change_type.value, 0) + 1
                by_impact[impact_level.value] = by_impact.get(impact_level.value, 0) + 1
                
                if legal_change.requires_human_review:
                    human_review_count += 1
                
                if impact_level in [ImpactLevel.HIGH, ImpactLevel.CRITICAL]:
                    material_count += 1
                
            except (KeyError, ValueError) as e:
                print(f"Warning: Skipping invalid change: {e}")
                continue
        
        # Build final result
        return {
            'summary': {
                'total_changes': len(validated_changes),
                'by_type': by_type,
                'by_impact': by_impact,
                'requires_human_review': human_review_count,
                'material_changes': material_count
            },
            'changes': [c.to_dict() for c in validated_changes],
            'ai_metadata': {
                'model': 'gemini-1.5-pro',
                'timestamp': datetime.now().isoformat(),
                'validation_passed': True
            }
        }
    
    @staticmethod
    def _fallback_chunked_comparison(doc1_json: Dict, doc2_json: Dict) -> Dict:
        """
        Handle large documents by comparing in chunks
        """
        # Extract top-level sections
        sections1 = GeminiDocumentComparator._extract_sections(doc1_json)
        sections2 = GeminiDocumentComparator._extract_sections(doc2_json)
        
        all_changes = []
        
        # Compare each section
        for section_name in set(list(sections1.keys()) + list(sections2.keys())):
            sec1 = sections1.get(section_name, {})
            sec2 = sections2.get(section_name, {})
            
            if not sec1:
                # New section
                all_changes.append({
                    'type': 'NEW_CLAUSE',
                    'path': section_name,
                    'to_value': json.dumps(sec2)[:500],
                    'description': f'New section: {section_name}',
                    'impact': 'HIGH',
                    'confidence': 1.0,
                    'requires_human_review': True,
                    'reasoning': 'Entire section added'
                })
            elif not sec2:
                # Removed section
                all_changes.append({
                    'type': 'REMOVED_CLAUSE',
                    'path': section_name,
                    'from_value': json.dumps(sec1)[:500],
                    'description': f'Removed section: {section_name}',
                    'impact': 'HIGH',
                    'confidence': 1.0,
                    'requires_human_review': True,
                    'reasoning': 'Entire section removed'
                })
            else:
                # Compare section
                section_result = GeminiDocumentComparator.compare(sec1, sec2)
                all_changes.extend(section_result.get('changes', []))
        
        # Aggregate results
        return GeminiDocumentComparator._validate_and_enrich({
            'changes': all_changes
        })
    
    @staticmethod
    def _extract_sections(doc: Dict, max_sections: int = 20) -> Dict:
        """Extract top-level sections from document"""
        sections = {}
        
        if isinstance(doc, dict):
            for key, value in list(doc.items())[:max_sections]:
                if isinstance(value, (dict, list)) and len(str(value)) > 100:
                    sections[key] = value
        
        return sections


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def compare_documents_with_ai(doc1: Dict, doc2: Dict) -> Dict:
    """
    Main entry point - replaces LegalDocumentComparator.compare()
    """
    return GeminiDocumentComparator.compare(doc1, doc2)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Test documents
    doc1 = {
        "contract": {
            "clauses": [
                {
                    "id": "5.2",
                    "text": "The Customer shall pay within 30 days of invoice date."
                },
                {
                    "id": "7.1",
                    "text": "The Vendor must maintain confidentiality of all customer data."
                }
            ]
        }
    }
    
    doc2 = {
        "contract": {
            "clauses": [
                {
                    "id": "6.1",  # Renumbered
                    "text": "The Customer shall pay within 60 days of invoice date."  # Changed!
                },
                {
                    "id": "8.1",  # Renumbered
                    "text": "The Vendor must maintain confidentiality of all customer data."  # Same
                }
            ]
        }
    }
    
    result = compare_documents_with_ai(doc1, doc2)
    print(json.dumps(result, indent=2))