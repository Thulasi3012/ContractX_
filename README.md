# ContractX - Complete Document Analysis with AI Chatbot 🤖

A sophisticated document analysis system that combines AI-powered text extraction, table detection, visual recognition, knowledge graph construction, and an intelligent RAG+KG chatbot for comprehensive contract and business document analysis.

**Version:** 1.0.0  
---

## 📋 Table of Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Database Setup](#database-setup)
- [Project Structure](#project-structure)
- [Project Flow](#project-flow)
- [Configuration](#configuration)
- [API Endpoints](#api-endpoints)
- [Usage Examples](#usage-examples)
- [Response Format](#response-format)
- [Troubleshooting](#troubleshooting)
- [Performance Metrics](#performance-metrics)
- [Advanced Configuration](#advanced-configuration)

---

## Overview

**ContractX** is an enterprise-grade document analysis platform that processes PDF contracts and business documents with multiple AI engines:

### Key Features

 **Multi-Page Processing**: Asynchronous page-by-page analysis  
 **Advanced Text Analysis**: Hierarchical document structure extraction (sections → clauses → sub-clauses)  
 **Smart Table Detection**: Multi-scale detection with merged cell handling  
 **Visual Recognition**: AI-powered chart, graph, and diagram detection with Gemini summaries  
 **Knowledge Graph**: Neo4j-based graph construction for entity relationships  
 **Vector Search (RAG)**: ChromaDB embeddings for semantic search  
 **Intelligent Chatbot**: RAG + Knowledge Graph combined retrieval  
 **Database Storage**: Complete extraction results in PostgreSQL/MySQL  
 **Comprehensive Summaries**: 4-5 paragraph AI-generated document overview  
 **Entity Extraction**: Buyers, sellers, dates, deadlines, obligations, alerts, contact info  

---

## System Requirements

### Hardware
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum (16GB+ recommended)
- **Disk**: 50GB+ for database and vector store
- **GPU**: Optional (CUDA-enabled for faster Gemini API requests)

### Software

**Operating System:**
- Windows 10/11
- macOS 10.14+
- Linux (Ubuntu 20.04+)

**Python:**
- Python 3.10+
- pip package manager

**External Services:**
- **Gemini API Key** (Google Cloud)
- **Neo4j Database** (v4.4+)
- **PostgreSQL** (v12+)

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Thulasi3012/ContractX_.git
cd ContractX
```

### Step 2: Create Virtual Environment

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install System Dependencies

**Windows:**
- Download Poppler from: https://github.com/oschwartz10612/poppler-windows/releases
- Extract and add to PATH

**macOS:**
```bash
brew install poppler
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install poppler-utils
```

### Step 5: Environment Configuration

Create `.env` file in project root:

```env
# API Keys
GEMINI_API_KEY=your_gemini_api_key_here

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/contractx
# OR for MySQL:
# DATABASE_URL=mysql+pymysql://user:password@localhost:3306/contractx

# Server Configuration
HOST=127.0.0.1
PORT=8000
RELOAD=False

# Processing Configuration
GEMINI_MODEL=gemini-2.0-flash
GEMINI_REQUEST_DELAY=3.0
MAX_RETRIES=5
RETRY_DELAY=60

# Upload Configuration
MAX_UPLOAD_SIZE=52428800  # 50MB
UPLOAD_DIR=uploads
LOG_DIR=logs
LOG_LEVEL=INFO

# ChromaDB Configuration
CHROMA_DB_PATH=./chroma_db
```

---

## Database Setup

### PostgreSQL Setup

**1. Install PostgreSQL:**
- Windows: Download from https://www.postgresql.org/download/windows/
- macOS: `brew install postgresql`
- Linux: `sudo apt-get install postgresql postgresql-contrib`

**2. Create Database:**

```sql
CREATE DATABASE contractx;
CREATE USER contractx_user WITH PASSWORD 'your_secure_password';
ALTER ROLE contractx_user SET client_encoding TO 'utf8';
ALTER ROLE contractx_user SET default_transaction_isolation TO 'read committed';
ALTER ROLE contractx_user SET default_transaction_deferrable TO on;
GRANT ALL PRIVILEGES ON DATABASE contractx TO contractx_user;
```

**3. Create Tables:**

```bash
cd Contractx
python -m alembic upgrade head
# OR manually run the schema creation from app/database/models.py
```

### Neo4j Setup

**1. Install Neo4j:**
- Download from: https://neo4j.com/download/
- Or use Docker:

```bash
docker run -d \
  --name neo4j \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

**2. Verify Connection:**

```bash
curl -u neo4j:password http://localhost:7474/db/neo4j/
```

### ChromaDB Setup

ChromaDB is automatically initialized on first run:

```
chroma_db/
├── chroma.sqlite3
└── [collection_folders]/
```

---

## Project Structure

```
ContractX/
├── app/
│   ├── __init__.py
│   ├── main.py                          # FastAPI application
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py                    # Configuration management
│   ├── database/
│   │   ├── __init__.py
│   │   ├── database.py                  # DB connection
│   │   ├── models.py                    # SQLAlchemy models
│   │   └── schemas.py                   # Pydantic schemas
│   ├── services/
│   │   ├── pdf_processor.py             # PDF extraction & table detection
│   │   ├── text_analyzer.py             # Text analysis with Gemini
│   │   ├── image_detector.py            # Visual detection with OpenCV + Gemini
│   │   ├── knowledge_graph_builder.py   # Neo4j graph construction
│   │   ├── database_service.py          # Database operations
│   │   ├── rag_service.py               # ChromaDB vector search
│   │   └── chatbot_service.py           # RAG + KG chatbot
│   └── utils/
│       ├── __init__.py
│       └── file_handler.py              # File upload/cleanup utilities
├── chroma_db/                            # Vector store (auto-created)
├── extracted_tables/                     # Extracted tables (auto-created)
├── logs/                                 # Application logs (auto-created)
├── uploads/                              # Uploaded PDFs (auto-created)
├── .env                                  # Environment variables
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
└── AUDIT_REPORT.md                       # Code audit report
```

---

## Project Flow

### Complete Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DOCUMENT UPLOAD                                  │
│              (POST /api/v1/extract-and-build-kg)                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                 ┌─────────────▼──────────────┐
                 │  Step 1: PDF Conversion    │
                 │  - Extract pages at DPI   │
                 │  - Convert to images      │
                 └─────────────┬──────────────┘
                               │
              ┌────────────────▼────────────────┐
              │  Step 2: Page-by-Page Analysis │
              │  (Async, Parallel Processing)  │
              └────┬─────────────┬─────────────┬┘
                   │             │             │
        ┌──────────▼─┐  ┌────────▼──┐  ┌────────▼──┐
        │TEXT ANALYSIS│  │TABLE      │  │VISUAL     │
        │             │  │DETECTION  │  │DETECTION  │
        │ - Sections  │  │           │  │           │
        │ - Clauses   │  │ - Strict  │  │ - OpenCV  │
        │ - Entities  │  │ - Permissive
        │ - Summary   │  │ - Gemini  │  │ - Gemini  │
        │             │  │ - Merged  │  │ Analysis  │
        │             │  │   Cells   │  │           │
        └──────┬──────┘  └────┬──────┘  └────┬──────┘
               │              │              │
               └──────────────┼──────────────┘
                              │
                  ┌───────────▼───────────┐
                  │ Step 3: Consolidate   │
                  │ - Create Overall      │
                  │   Summary             │
                  │ - Merge Entities      │
                  │ - Generate Document   │
                  │   Type                │
                  └───────┬────────┬──────┘
                          │        │
         ┌────────────────▼─┐   ┌──▼──────────────┐
         │Step 4: Database  │   │Step 5: Knowledge│
         │Storage (SQLAlchemy
         │- Save full        │   │Graph Building   │
         │  extraction       │   │- Create nodes   │
         │- Store UUIDs      │   │- Build edges    │
         │- Index            │   │- Link entities  │
         └────────┬──────────┘   └────────┬────────┘
                  │                       │
                  │    ┌──────────────────┘
                  │    │
                  └────┼───┐
                       │   │
              ┌────────▼───▼─────────┐
              │Step 6: RAG Indexing   │
              │- ChromaDB chunking    │
              │- Vector embeddings    │
              │- Document filtering   │
              └───────────┬───────────┘
                          │
                  ┌───────▼──────────┐
                  │ FINAL RESPONSE   │
                  │ - Document ID    │
                  │ - Full Analysis  │
                  │ - Statistics     │
                  │ - Status         │
                  └──────────────────┘
```

### Chatbot Query Flow

```
┌──────────────────────────────┐
│  User Question               │
│  /api/v1/chatbot/{doc_id}    │
└──────────────┬───────────────┘
               │
        ┌──────▼──────┐
        │Query Routing│
        └──┬─────┬────┘
          │     │
    ┌─────▼─┐  │
    │ RAG   │  │
    │Search │  │
    │(8     │  │
    │chunks)│  │
    └─────┬─┘  │
          │    │ ┌─────────────┐
          │    └─▶│Knowledge   │
          │       │Graph Query │
          │       │(Entity     │
          │       │Relationships
          │       └──────┬──────┘
          │              │
          └──────┬───────┘
                 │
         ┌───────▼────────────┐
         │Context Building    │
         │- Merge RAG results │
         │- Format KG data    │
         │- Create context    │
         └───────┬────────────┘
                 │
         ┌───────▼────────────┐
         │Gemini LLM Response │
         │- Generate answer   │
         │- Cite sources      │
         │- Add confidence    │
         └───────┬────────────┘
                 │
         ┌───────▼────────────┐
         │Chatbot Response    │
         │- Answer text       │
         │- Source references│
         │- Context used     │
         └────────────────────┘
```

---

## Configuration

### Environment Variables Details

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | Required | Google Gemini API key |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | Required | Neo4j password |
| `DATABASE_URL` | Required | Database connection string |
| `GEMINI_MODEL` | `gemini-2.0-flash` | AI model for analysis |
| `GEMINI_REQUEST_DELAY` | `3.0` | Delay between API requests (seconds) |
| `MAX_RETRIES` | `5` | API retry attempts |
| `RETRY_DELAY` | `60` | Delay before retry (seconds) |
| `MAX_UPLOAD_SIZE` | `52428800` | Max file size (50MB) |

---

## API Endpoints

### 1. Document Extraction & Full Pipeline

**Endpoint:** `POST /api/v1/extract-and-build-kg`

**Description:** Complete pipeline - Extract document, build KG, store in DB, index in RAG

**Parameters:**
- `file` (UploadFile, required): PDF file to analyze
- `dpi` (int, optional): Image resolution (100-600, default: 350)

**Request Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/extract-and-build-kg?dpi=350" \
  -H "accept: application/json" \
  -F "file=@contract.pdf"
```

**Response Format:**
```json
{
  "request_id": "req_20251205_121147",
  "filename": "contract.pdf",
  "total_pages": 15,
  "pages": [
    {
      "page_number": 1,
      "text_analysis": {
        "sections": [
          {
            "heading": "1. INTRODUCTION",
            "heading_id": "1",
            "sub_headings": [
              {
                "sub_heading": "1.1 Parties to the Agreement",
                "sub_heading_id": "1.1",
                "clauses": [
                  {
                    "clause": "This agreement is between...",
                    "clause_id": "1.1.1",
                    "sub_clauses": []
                  }
                ]
              }
            ]
          }
        ],
        "entities": {
          "document_type": "Service Agreement",
          "buyer_name": "Acme Corporation",
          "seller_name": "Tech Solutions Inc",
          "dates": ["January 1, 2024", "December 31, 2024"],
          "deadlines": ["March 31, 2024", "June 30, 2024"],
          "alerts": ["Payment due within 30 days", "Confidentiality clause required"],
          "obligations": [
            {
              "party": "Seller",
              "description": "Provide technical support 24/7",
              "page": "1"
            }
          ],
          "addresses": ["123 Business Ave, NYC", "456 Tech Blvd, SF"],
          "contact_info": {
            "email": "contact@techsolutions.com",
            "phone": "+1-234-567-8900"
          }
        },
        "summary": "Page summary text...",
        "sections_count": 12
      },
      "tables": [
        {
          "table_id": "T1",
          "table_title": "Pricing Schedule",
          "table_type": "financial",
          "headers": ["Item", "Unit Price", "Quantity", "Total"],
          "rows": [
            ["Service A", "$100", "10", "$1000"],
            ["Service B", "$200", "5", "$1000"]
          ],
          "total_rows": 2,
          "total_columns": 4,
          "has_merged_cells": false
        }
      ],
      "visuals": [
        {
          "visual_id": "page_1_visual_1",
          "type": "chart",
          "bbox": [120, 350, 800, 920],
          "summary": "Bar chart showing quarterly revenue...",
          "width": 680,
          "height": 570,
          "area": 387600
        }
      ]
    }
  ],
  "summary": {
    "total_sections": 45,
    "total_tables": 8,
    "total_visuals": 3,
    "entities": {
      "buyer_name": "Acme Corporation",
      "seller_name": "Tech Solutions Inc",
      "dates": ["January 1, 2024"],
      "deadlines": ["March 31, 2024"],
      "alerts": ["Payment due within 30 days"],
      "obligations": []
    }
  },
  "overall_summary": {
    "summary": "This is a Service Agreement between Acme Corporation (Buyer/Client) and Tech Solutions Inc (Seller/Provider)...",
    "document_type": "Service Agreement",
    "entities": {
      "buyer_name": "Acme Corporation",
      "seller_name": "Tech Solutions Inc",
      "dates": ["January 1, 2024"],
      "deadlines": ["March 31, 2024"],
      "alerts": ["Payment due within 30 days"],
      "obligations": [...]
    }
  },
  "metadata": {
    "dpi": 350,
    "extraction_method": "gemini-text-extractor",
    "text_model": "gemini-2.0-flash",
    "table_extraction_model": "gemini-2.0-flash",
    "version": "4.0.0"
  },
  "database": {
    "status": "success",
    "document_id": "a1b2c3d4-e5f6-g7h8-i9j0-k1l2m3n4o5p6",
    "message": "Document stored successfully"
  },
  "knowledge_graph": {
    "status": "success",
    "total_nodes": 245,
    "total_relationships": 542
  },
  "rag_indexing": {
    "status": "success",
    "total_chunks": 156
  },
  "status": "success",
  "timestamp": "2025-12-05T12:21:47.123456"
}
```

---

### 2. Chatbot Query (RAG + KG Combined)

**Endpoint:** `POST /api/v1/chatbot/{document_id}`

**Description:** Ask questions about a document using combined RAG + Knowledge Graph

**Parameters:**
- `document_id` (string, path): Document UUID from database
- `question` (string, body): User question
- `include_rag` (bool, optional): Enable RAG retrieval (default: true)
- `include_kg` (bool, optional): Enable Knowledge Graph retrieval (default: true)
- `n_results` (int, optional): Number of RAG chunks to retrieve (default: 8)

**Request Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/chatbot/a1b2c3d4-e5f6-g7h8-i9j0-k1l2m3n4o5p6" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Who is the seller and what are their obligations?",
    "include_rag": true,
    "include_kg": true,
    "n_results": 8
  }'
```

**Response Format:**
```json
{
  "document_id": "a1b2c3d4-e5f6-g7h8-i9j0-k1l2m3n4o5p6",
  "question": "Who is the seller and what are their obligations?",
  "answer": "According to the document, Tech Solutions Inc is the seller. Their primary obligations include: 1) Providing technical support 24/7, 2) Delivering services within the agreed timeline, 3) Maintaining confidentiality of client data, 4) Providing monthly status reports.",
  "sources": {
    "rag_sources": [
      {
        "content": "Tech Solutions Inc shall provide 24/7 technical support...",
        "type": "section",
        "page": 1,
        "section": "1.1 Parties to the Agreement",
        "source": "RAG"
      }
    ],
    "kg_sources": [
      {
        "type": "entity",
        "entity_type": "Organization",
        "name": "Tech Solutions Inc",
        "source": "KG"
      }
    ]
  },
  "context_used": {
    "rag_chunks": 3,
    "kg_nodes": 2
  },
  "confidence": "high"
}
```

---

### 3. Get Document by ID

**Endpoint:** `GET /api/v1/documents/{document_id}`

**Response:**
```json
{
  "id": "a1b2c3d4-e5f6-g7h8-i9j0-k1l2m3n4o5p6",
  "document_name": "contract.pdf",
  "summary": "Overall document summary...",
  "document_type": "Service Agreement",
  "buyer": "Acme Corporation",
  "seller": "Tech Solutions Inc",
  "deadlines": ["March 31, 2024"],
  "alerts": ["Payment due within 30 days"],
  "obligations": [...],
  "page_count": 15,
  "uploaded_on": "2025-12-05T12:21:47.123456"
}
```

---

### 4. Delete Document (All Systems)

**Endpoint:** `DELETE /api/v1/documents/{document_id}`

**Description:** Delete from Database, Knowledge Graph, and RAG

**Response:**
```json
{
  "status": "success",
  "message": "Document deleted from all systems",
  "document_id": "a1b2c3d4-e5f6-g7h8-i9j0-k1l2m3n4o5p6",
  "deleted_from": {
    "database": true,
    "knowledge_graph": true,
    "rag": true
  }
}
```

---

### 5. RAG Search

**Endpoint:** `POST /api/v1/rag/search`

**Description:** Vector search in document chunks

**Request:**
```json
{
  "query": "What are the payment terms?",
  "document_id": "a1b2c3d4-e5f6-g7h8-i9j0-k1l2m3n4o5p6",
  "n_results": 5
}
```

**Response:**
```json
[
  {
    "content": "Payment shall be made within 30 days of invoice...",
    "metadata": {
      "document_id": "a1b2c3d4-e5f6-g7h8-i9j0-k1l2m3n4o5p6",
      "page_number": 5,
      "type": "section",
      "section_heading": "4. PAYMENT TERMS"
    },
    "distance": 0.15
  }
]
```

---

### 6. Health Check

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "models": {
    "text_analysis": "gemini-2.0-flash",
    "table_extraction": "gemini-2.0-flash"
  },
  "version": "4.0.0"
}
```

---

## Usage Examples

### Python Client Example

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# 1. Upload and analyze document
def upload_document(pdf_path):
    with open(pdf_path, 'rb') as f:
        files = {'file': f}
        params = {'dpi': 350}
        response = requests.post(
            f"{BASE_URL}/api/v1/extract-and-build-kg",
            files=files,
            params=params
        )
    return response.json()

# 2. Ask chatbot question
def ask_chatbot(document_id, question):
    response = requests.post(
        f"{BASE_URL}/api/v1/chatbot/{document_id}",
        json={
            "question": question,
            "include_rag": True,
            "include_kg": True,
            "n_results": 8
        }
    )
    return response.json()

# 3. Search in RAG
def search_document(document_id, query):
    response = requests.post(
        f"{BASE_URL}/api/v1/rag/search",
        json={
            "query": query,
            "document_id": document_id,
            "n_results": 5
        }
    )
    return response.json()

# Usage
result = upload_document("contract.pdf")
doc_id = result['database']['document_id']

answer = ask_chatbot(doc_id, "Who is the buyer?")
print(f"Answer: {answer['answer']}")

search_results = search_document(doc_id, "payment terms")
print(f"Found {len(search_results)} chunks")
```


### cURL Examples

```bash
# 1. Upload document
curl -X POST "http://localhost:8000/api/v1/extract-and-build-kg?dpi=350" \
  -F "file=@contract.pdf" | json_pp

# 2. Ask chatbot
curl -X POST "http://localhost:8000/api/v1/chatbot/doc_id_here" \
  -H "Content-Type: application/json" \
  -d '{"question":"Who is the buyer?"}' | json_pp

# 3. Search
curl -X POST "http://localhost:8000/api/v1/rag/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"payment terms","document_id":"doc_id_here"}' | json_pp

# 4. Get document
curl "http://localhost:8000/api/v1/documents/doc_id_here" | json_pp

# 5. Delete document
curl -X DELETE "http://localhost:8000/api/v1/documents/doc_id_here" | json_pp

# 6. Health check
curl "http://localhost:8000/health" | json_pp
```

---

## Response Format Details

### Document Entity Structure

```
Document
├── Text Analysis (per page)
│   ├── Sections
│   │   ├── Sub-headings
│   │   │   └── Clauses
│   │   │       └── Sub-clauses
│   │   └── Entities
│   │       ├── Buyers
│   │       ├── Sellers
│   │       ├── Dates
│   │       ├── Deadlines
│   │       ├── Alerts
│   │       ├── Obligations
│   │       ├── Addresses
│   │       └── Contact Info
│   └── Summary
├── Tables
│   ├── Headers
│   ├── Rows
│   ├── Merged Cells
│   └── Metadata
└── Visuals
    ├── Charts
    ├── Graphs
    ├── Diagrams
    └── AI Summaries
```

---

## Troubleshooting

### Issue: `GEMINI_API_KEY not found`

**Solution:**
```bash
# Verify .env file exists
cat .env | grep GEMINI_API_KEY

# Ensure key is valid
# Get from: https://makersuite.google.com/app/apikey
```

### Issue: Neo4j Connection Failed

**Solution:**
```bash
# Check Neo4j is running
curl -u neo4j:password http://localhost:7474/

# Restart Neo4j
# Windows: net stop/start neo4j
# Linux/Mac: sudo systemctl restart neo4j
```

### Issue: ChromaDB Errors

**Solution:**
```bash
# Delete and recreate
rm -rf chroma_db/

# API will auto-create on next request
```

### Issue: Table Detection Not Working

**Solution:**
- Increase DPI: `dpi=600`
- Check PDF quality
- Enable permissive mode (automatic fallback)
- Verify Gemini API quota

### Issue: High Memory Usage

**Solution:**
- Reduce `MAX_PARALLEL` in image detector
- Process smaller files
- Increase available RAM

### Issue: Asyncio Event Loop Error

**Solution:**
```bash
# Make sure reload=False in production
# Update main.py:
# uvicorn.run(app, reload=False)
```

### Issue: Text Analyzer Format String Error

**Solution:**
- Ensure all braces `{}` in prompts are escaped as `{{}}`
- Check for None values in metadata
- Verify f-string usage is correct

---

## Performance Metrics

### Typical Processing Times (15-page document)

| Operation | Time | Notes |
|-----------|------|-------|
| PDF Conversion | 2-3s | Depends on DPI |
| Text Analysis | 20-30s | 2-3s per page |
| Table Detection | 5-10s | Multi-scale detection |
| Visual Analysis | 10-15s | Gemini API calls |
| KG Building | 5-8s | Neo4j writes |
| RAG Indexing | 3-5s | ChromaDB chunking |
| **Total** | **45-70s** | Full pipeline |

### Resource Usage

- **CPU**: 40-60% during processing
- **RAM**: 1.5-2GB active usage
- **Disk I/O**: 50-100MB during PDF conversion
- **Network**: Minimal (Gemini API calls)

### Throughput

| Document Type | Pages | Tables | Processing Time |
|---|---|---|---|
| Simple contract | 5 | 2 | ~30 seconds |
| Financial report | 10 | 8 | ~2 minutes |
| Legal agreement | 20 | 5 | ~3 minutes |
| Complex pricing | 10 | 15 | ~4 minutes |

---

## Advanced Configuration

### Custom Chunking Strategy

Edit `rag_service.py`:
```python
def clean_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    # Customize chunk sizes and overlap
    # Modify chunking parameters based on document type
```

### Modify Detection Thresholds

Edit `image_detector.py`:
```python
self.MIN_AREA = 20000  # Increase for fewer detections
self.ENTROPY_THRESHOLD = 3.0  # Adjust visual complexity
self.MAX_PARALLEL = 4  # Adjust parallelization
```

### Custom Prompt Engineering

Edit `text_analyzer.py`:
```python
def _create_prompt(self, page_number: int, text: str) -> str:
    # Customize extraction instructions
    # Adjust entity extraction rules
```

### Database Optimization

```sql
-- Create indexes for faster queries
CREATE INDEX idx_document_id ON documents(id);
CREATE INDEX idx_page_number ON documents(page_count);
CREATE INDEX idx_buyer_seller ON documents(buyer, seller);
```

---

## Production Deployment

### Using Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV GEMINI_API_KEY=${GEMINI_API_KEY}
ENV NEO4J_URI=bolt://neo4j:7687
ENV DATABASE_URL=postgresql://user:password@db:5432/contractx

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      GEMINI_API_KEY: ${GEMINI_API_KEY}
      NEO4J_URI: bolt://neo4j:7687
      DATABASE_URL: postgresql://contractx:password@db:5432/contractx
    depends_on:
      - db
      - neo4j

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: contractx
      POSTGRES_USER: contractx
      POSTGRES_PASSWORD: password
    volumes:
      - db_data:/var/lib/postgresql/data

  neo4j:
    image: neo4j:latest
    environment:
      NEO4J_AUTH: neo4j/password
    ports:
      - "7687:7687"
    volumes:
      - neo4j_data:/data

volumes:
  db_data:
  neo4j_data:
```

---

## Support & Contribution

For issues, questions, or contributions:
- **GitHub Issues**: https://github.com/yourusername/ContractX/issues
- **Email**: support@yourcompany.com
- **Documentation**: See AUDIT_REPORT.md

---

## License

MIT License - See LICENSE file for details

---

## Version History

- **v4.0.0** (Current) - Full pipeline with chatbot, RAG, KG, DB, visual summaries
- **v3.5.0** - Image detection with Gemini analysis
- **v3.0.0** - Advanced table detection
- **v2.0.0** - Image detection added
- **v1.0.0** - Initial release

---

## Acknowledgments

- Powered by **Google Gemini** AI
- Graph Database: **Neo4j**
- Vector Store: **ChromaDB**
- PDF Processing: **PyMuPDF**
- Computer Vision: **OpenCV**
- Web Framework: **FastAPI**

---

## Quick Reference

**Start Server:**
```bash
python -m uvicorn app.main:app --reload
```

**Test API:**
```bash
# Navigate to: http://localhost:8000/docs
```

**Process Document:**
```bash
curl -X POST http://localhost:8000/api/v1/extract-and-build-kg \
  -F "file=@contract.pdf"
```

**View Logs:**
```bash
tail -f logs/extraction_*.log
```

**Check Health:**
```bash
curl http://localhost:8000/health
```

---

**ContractX v4.0 - Enterprise Document Intelligence** 🎉
