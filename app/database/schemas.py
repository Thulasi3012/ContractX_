from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class SubClause(BaseModel):
    sub_clause: str
    sub_clause_id: str

class Clause(BaseModel):
    clause: str
    clause_id: str
    sub_clauses: List[SubClause] = []

class SubHeading(BaseModel):
    sub_heading: str
    sub_heading_id: str
    clauses: List[Clause] = []

class Section(BaseModel):
    heading: str
    heading_id: str
    sub_headings: List[SubHeading] = []

class TableInfo(BaseModel):
    contains_table: bool
    pages: List[int] = []
    count: int = 0

class ImageInfo(BaseModel):
    contains_image: bool
    pages: List[int] = []
    count: int = 0

class Entities(BaseModel):
    buyer_name: Optional[str] = None
    seller_name: Optional[str] = None
    objection_level: Optional[str] = None
    dates: List[str] = []
    alerts: List[str] = []
    deadlines: List[str] = []
    addresses: List[str] = []

class ChunkMetadata(BaseModel):
    chunk_id: int
    page_start: int
    page_end: int
    total_pages: int

class Chunk(BaseModel):
    metadata: ChunkMetadata
    text: str
    images: List[Dict[str, Any]] = []
    raw_pages: List[Dict[str, Any]] = []

class TextLLMOutput(BaseModel):
    sections: List[Section] = []
    tables: TableInfo
    images: ImageInfo
    entities: Entities
    chunk_metadata: ChunkMetadata

class TableDetectionOutput(BaseModel):
    tables: List[Dict[str, Any]] = []
    chunk_metadata: ChunkMetadata

class TableStructure(BaseModel):
    page: int
    table_id: str
    rows: int
    columns: int
    headers: List[str]
    data: List[List[str]]
    metadata: Dict[str, Any] = {}

class FinalTable(BaseModel):
    table_id: str
    page: int
    structure: TableStructure
    source: str  # 'text_llm', 'hf_detection', or 'both'
    confidence: float

class DocumentAnalysisResponse(BaseModel):
    document_id: str = Field(default_factory=lambda: f"doc_{datetime.now().timestamp()}")
    sections: List[Section]
    entities: Entities
    tables: List[FinalTable]
    images: ImageInfo
    metadata: Dict[str, Any]
    processing_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())