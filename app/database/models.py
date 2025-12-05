# app/models/document.py
from sqlalchemy import Column, String, Integer, DateTime, Text, JSON
from datetime import datetime
from app.database.database import Base

class Document(Base):
    """Enhanced Document model with comprehensive fields for contract analysis"""
    __tablename__ = "documents"
    
    # Primary key - UUID
    id = Column(String(36), primary_key=True, index=True)
    
    # Basic document information
    document_name = Column(String(255), nullable=False, index=True)
    uploaded_on = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Analysis fields
    summary = Column(Text, nullable=True)
    document_type = Column(String(100), nullable=True, index=True)
    document_version = Column(String(50), nullable=True)
    
    # Parties information
    buyer = Column(String(500), nullable=True, index=True)
    seller = Column(String(500), nullable=True, index=True)
    parties_json = Column(JSON, nullable=True)  # All parties involved
    
    # Critical information
    deadlines = Column(JSON, nullable=True)  # List of deadlines
    alerts = Column(JSON, nullable=True)  # Critical alerts
    obligations = Column(JSON, nullable=True)  # Party obligations
    
    # Text data
    cleaned_text = Column(Text, nullable=False)
    text_as_json = Column(JSON, nullable=False)  # Structured JSON with pages, tables, visuals
    
    # Metadata
    page_count = Column(Integer, nullable=True)
    extraction_method = Column(String(50), nullable=True)
    
    def __repr__(self):
        return f"<Document(id={self.id}, name={self.document_name}, type={self.document_type})>"
    
    def to_dict(self):
        """Convert document to dictionary"""
        return {
            "id": self.id,
            "document_name": self.document_name,
            "uploaded_on": self.uploaded_on.isoformat() if self.uploaded_on else None,
            "summary": self.summary,
            "document_type": self.document_type,
            "document_version": self.document_version,
            "buyer": self.buyer,
            "seller": self.seller,
            "parties_json": self.parties_json,
            "deadlines": self.deadlines,
            "alerts": self.alerts,
            "obligations": self.obligations,
            "page_count": self.page_count,
            "extraction_method": self.extraction_method,
            "text_as_json": self.text_as_json
        }