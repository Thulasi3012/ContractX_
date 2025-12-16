"""
Document Comparison API Routes - AI-Powered with Gemini
"""
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session
from typing import Dict, Optional, List
import logging
import os

from app.services.AI_comparasion import GeminiDocumentComparator
from app.database.database import get_db
from app.database.models import Document

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/AI_comparasion",
    tags=["AI Document Comparison"]
)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ComparisonRequest(BaseModel):
    """Request model for document comparison"""
    document_1_id: str = Field(..., description="First document ID")
    document_2_id: str = Field(..., description="Second document ID")
    
    @validator('document_2_id')
    def documents_must_differ(cls, v, values):
        if 'document_1_id' in values and v == values['document_1_id']:
            raise ValueError('Documents must be different')
        return v


class ChangeSummary(BaseModel):
    """Change summary statistics"""
    total_changes: int
    by_type: Dict[str, int]
    by_impact: Dict[str, int]
    requires_human_review: int
    material_changes: int


class LegalChangeDetail(BaseModel):
    """Individual change detail"""
    type: str
    path: str
    from_value: Optional[str] = None
    to_value: Optional[str] = None
    description: str
    impact: str
    confidence: float
    requires_human_review: bool
    ai_reasoning: Optional[str] = None  # NEW: AI's explanation


class ComparisonResponse(BaseModel):
    """AI-powered comparison response"""
    document_1_id: str
    document_2_id: str
    analysis_method: str = "AI-Powered (Gemini)"
    summary: ChangeSummary
    changes: List[LegalChangeDetail]
    recommendations: List[str] = []
    ai_metadata: Optional[Dict] = None


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post(
    "/compare",
    response_model=ComparisonResponse,
    status_code=status.HTTP_200_OK,
    summary="AI-powered legal document comparison",
    description="Compare documents using Gemini AI for intelligent legal analysis",
    responses={
        200: {"description": "AI analysis successful"},
        404: {"model": ErrorResponse, "description": "Document not found"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def compare_legal_documents(
    request: ComparisonRequest,
    db: Session = Depends(get_db)
):
    """
    AI-powered legal document comparison using Gemini
    
    Features:
    - Intelligent change detection (no hardcoded rules)
    - Understands legal context and intent
    - Distinguishes translations from real changes
    - Ignores structural reorganization
    - Provides confidence scores and reasoning
    - Flags high-risk changes for human review
    
    Returns:
        ComparisonResponse with AI-powered analysis
    """
    
    try:
        logger.info(f"AI comparison: {request.document_1_id} vs {request.document_2_id}")
        
        # Fetch documents
        doc1 = db.query(Document).filter(Document.id == request.document_1_id).first()
        if not doc1:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{request.document_1_id}' not found"
            )
        
        doc2 = db.query(Document).filter(Document.id == request.document_2_id).first()
        if not doc2:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{request.document_2_id}' not found"
            )
        
        # Validate JSON content
        if not doc1.text_as_json:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Document '{request.document_1_id}' has no JSON content"
            )
        
        if not doc2.text_as_json:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Document '{request.document_2_id}' has no JSON content"
            )
        
        logger.info("Documents fetched. Starting AI analysis...")
        
        # Execute AI-powered comparison
        result = GeminiDocumentComparator.compare(
            doc1.text_as_json,
            doc2.text_as_json
        )
        
        logger.info(f"AI analysis complete. Material changes: {result['summary']['material_changes']}")
        
        # Generate recommendations
        recommendations = _generate_recommendations(result)
        
        # Build response
        response = ComparisonResponse(
            document_1_id=request.document_1_id,
            document_2_id=request.document_2_id,
            analysis_method="AI-Powered (Gemini 1.5 Pro)",
            summary=ChangeSummary(**result['summary']),
            changes=[LegalChangeDetail(**c) for c in result['changes']],
            recommendations=recommendations,
            ai_metadata=result.get('ai_metadata')
        )
        
        return response
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Comparison error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI comparison failed: {str(e)}"
        )


@router.get(
    "/compare/{document_1_id}/{document_2_id}",
    response_model=ComparisonResponse,
    status_code=status.HTTP_200_OK,
    summary="AI comparison (GET method)"
)
async def compare_legal_documents_get(
    document_1_id: str,
    document_2_id: str,
    db: Session = Depends(get_db)
):
    """GET method for AI document comparison"""
    
    if document_1_id == document_2_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Documents must be different"
        )
    
    request = ComparisonRequest(
        document_1_id=document_1_id,
        document_2_id=document_2_id
    )
    
    return await compare_legal_documents(request, db)


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Health check"
)
async def health_check():
    """Service health check"""
    return {
        "status": "healthy",
        "service": "ai-document-comparison",
        "version": "3.0.0",
        "analysis_method": "AI-Powered (Gemini)",
        "features": [
            "Intelligent change detection",
            "Legal context understanding",
            "Translation detection",
            "Structural reorganization handling",
            "Confidence scoring",
            "AI reasoning provided",
            "Human review flagging"
        ]
    }


@router.get(
    "/capabilities",
    summary="Get AI capabilities",
    description="Returns information about AI analysis capabilities"
)
async def ai_capabilities():
    """Return AI capabilities information"""
    return {
        "analysis_method": "AI-Powered (Gemini 1.5 Pro)",
        "advantages_over_hardcoded": {
            "context_awareness": "Understands legal context and intent",
            "translation_detection": "Automatically detects translations vs changes",
            "flexibility": "No hardcoded rules - adapts to any document type",
            "reasoning": "Provides explanations for every decision",
            "confidence": "Assigns confidence scores to each finding",
            "accuracy": "Learns from patterns, not just rules"
        },
        "supported_change_types": [
            "NO_IMPACT_CHANGE",
            "STRUCTURAL_CHANGE",
            "OBLIGATION_CHANGE",
            "SCOPE_CHANGE",
            "PARTY_CHANGE",
            "CONDITION_CHANGE",
            "TIMING_CHANGE",
            "RISK_CHANGE",
            "NEW_CLAUSE",
            "REMOVED_CLAUSE",
            "LANGUAGE_EQUIVALENT"
        ],
        "impact_levels": [
            "NONE",
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL"
        ],
        "limitations": {
            "context_window": "~500KB per comparison",
            "large_documents": "Automatically chunked for processing",
            "api_costs": "Usage-based pricing applies"
        }
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _generate_recommendations(result: Dict) -> List[str]:
    """Generate actionable recommendations based on AI analysis"""
    recommendations = []
    
    summary = result['summary']
    changes = result['changes']
    
    # Check for high-risk changes
    if summary['requires_human_review'] > 0:
        recommendations.append(
            f"⚠️ {summary['requires_human_review']} change(s) require human legal review "
            f"(AI flagged as high-risk or low-confidence)"
        )
    
    # Check for critical changes
    critical_count = summary['by_impact'].get('CRITICAL', 0)
    if critical_count > 0:
        recommendations.append(
            f"🚨 {critical_count} CRITICAL change(s) detected - immediate review required"
        )
    
    # Check for party changes
    party_changes = sum(1 for c in changes if c['type'] == 'PARTY_CHANGE')
    if party_changes > 0:
        recommendations.append(
            f"👥 {party_changes} party change(s) - verify all stakeholders are aware"
        )
    
    # Check for obligation changes
    obligation_changes = sum(1 for c in changes if c['type'] == 'OBLIGATION_CHANGE')
    if obligation_changes > 0:
        recommendations.append(
            f"📋 {obligation_changes} obligation change(s) - review compliance implications"
        )
    
    # Check for timing changes
    timing_changes = sum(1 for c in changes if c['type'] == 'TIMING_CHANGE')
    if timing_changes > 0:
        recommendations.append(
            f"⏰ {timing_changes} timing change(s) - update project schedules"
        )
    
    # Language equivalence (AI detected translations)
    lang_equiv = sum(1 for c in changes if c['type'] == 'LANGUAGE_EQUIVALENT')
    if lang_equiv > 0:
        recommendations.append(
            f"🌐 {lang_equiv} translation(s) detected by AI - no legal change"
        )
    
    # Low-impact changes
    no_impact = summary['by_impact'].get('NONE', 0)
    if no_impact > 0:
        recommendations.append(
            f"✅ {no_impact} change(s) have no legal impact (per AI analysis)"
        )
    
    # AI-specific recommendations
    low_confidence = sum(1 for c in changes if c.get('confidence', 1.0) < 0.7)
    if low_confidence > 0:
        recommendations.append(
            f"🤖 {low_confidence} change(s) with low AI confidence - recommend human verification"
        )
    
    if not recommendations:
        recommendations.append("✅ AI analysis found no significant concerns - proceed with standard review")
    
    return recommendations