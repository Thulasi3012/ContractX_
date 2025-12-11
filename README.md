# ContractX - Complete Document Analysis with AI Chatbot 🤖

A sophisticated document analysis system that combines AI-powered text extraction, table detection, visual recognition, knowledge graph construction, and an intelligent RAG+KG chatbot for comprehensive contract and business document analysis.

**Version:** 4.0.0  
**Status:** Production Ready  
---

## 📋 Table of Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Quick Start (Docker)](#quick-start-docker)
- [Installation](#installation)
- [Complete Project Flow](#complete-project-flow)
- [Docker & Service Setup](#docker--service-setup)
- [Configuration](#configuration)
- [Database Setup](#database-setup)
- [RAG & Vector Store Setup](#rag--vector-store-setup)
- [Knowledge Graph Setup](#knowledge-graph-setup)
- [Project Structure](#project-structure)
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

✅ **Multi-Page Processing**: Asynchronous page-by-page analysis  
✅ **Advanced Text Analysis**: Hierarchical document structure extraction (sections → clauses → sub-clauses)  
✅ **Smart Table Detection**: Multi-scale detection with merged cell handling  
✅ **Visual Recognition**: AI-powered chart, graph, and diagram detection with Gemini summaries  
✅ **Knowledge Graph**: Neo4j-based graph construction for entity relationships  
✅ **Vector Search (RAG)**: Qdrant-based semantic search with BGE embeddings  
✅ **Intelligent Chatbot**: RAG + Knowledge Graph combined retrieval  
✅ **Database Storage**: Complete extraction results in PostgreSQL/MySQL  
✅ **LLM Usage Tracking**: Token counting and cost tracking (Gemini API)  
✅ **Comprehensive Summaries**: 4-5 paragraph AI-generated document overview  
✅ **Entity Extraction**: Buyers, sellers, dates, deadlines, obligations, alerts, contact info  

---

## System Requirements

### Hardware
- **CPU**: 4+ cores recommended
- **RAM**: 16GB minimum (32GB+ recommended for production)
- **Disk**: 100GB+ for database, vector store, and file uploads

### Software
- **OS**: Windows, macOS, or Linux
- **Docker**: 20.10+ (for containerized setup)
- **Docker Compose**: 2.0+ (for multi-container orchestration)
- **Python**: 3.9+
- **Git**: For version control

### External Services Required
- **Gemini API Key** (Google Cloud) - for text/table/image analysis
- **PostgreSQL** (v12+) or **MySQL** (v8+) - for document storage
- **Neo4j** (v4.4+) - for knowledge graph
- **Qdrant** (latest) - for vector search & RAG

---

## Quick Start (Docker) ⚡

The fastest way to get ContractX running with all services:

### Prerequisites
- Docker Desktop installed and running
- API keys ready (.env file)

### 1. Clone Repository

```bash
git clone https://github.com/Thulasi3012/ContractX_.git
cd ContractX
```

### 2. Create .env Filec

```bash
cp .env.example .env
# Edit .env with your credentials
```

### 3. Start All Services

```bash
docker-compose up -d
```

### 4. Verify Services

```bash
# Check all containers running
docker-compose ps

# Check logs
docker-compose logs -f app

# Test API
curl http://localhost:8000/health
```

### 5. Stop Services

```bash
docker-compose down
```

---

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/Thulasi3012/ContractX.git
cd ContractX
```

### Step 2: Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS/Linux (Bash):**
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
1. Download Poppler from: https://github.com/oschwartz10612/poppler-windows/releases
2. Extract to `C:\Program Files\poppler`
3. Add to System PATH: `C:\Program Files\poppler\bin`

**macOS:**
```bash
brew install poppler
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install poppler-utils python3-dev build-essential
```

### Step 5: Create Environment Configuration

Create `.env` file in project root:

```env
# ========================================
# GEMINI API CONFIGURATION
# ========================================
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
GEMINI_REQUEST_DELAY=3.0
MAX_RETRIES=5
RETRY_DELAY=60
EXPONENTIAL_BACKOFF=true

# ========================================
# DATABASE CONFIGURATION (PostgreSQL)
# ========================================
DB_HOST=localhost
DB_PORT=5432
DB_NAME=contractx
DB_USER=contractx_user
DB_PASSWORD=your_secure_password

# ========================================
# NEO4J CONFIGURATION (Knowledge Graph)
# ========================================
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# ========================================
# QDRANT CONFIGURATION (Vector Store)
# ========================================
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=contractx_documents
QDRANT_VECTOR_SIZE=1024

# ========================================
# SERVER CONFIGURATION
# ========================================
HOST=0.0.0.0
PORT=8000
DEBUG=false

# ========================================
# FILE UPLOAD CONFIGURATION
# ========================================
MAX_UPLOAD_SIZE=52428800
UPLOAD_DIR=uploads
LOG_DIR=logs
LOG_LEVEL=INFO

# ========================================
# AUTHENTICATION (Optional)
# ========================================
AUTH_REQUIRED=false
```

### Step 6: Initialize Database

```bash
python -c "from app.database.database import init_db; init_db()"
```

---

## Docker & Service Setup

### Create docker-compose.yml

Create `docker-compose.yml` in project root:

```yaml
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    container_name: contractx_postgres
    environment:
      POSTGRES_USER: contractx_user
      POSTGRES_PASSWORD: your_secure_password
      POSTGRES_DB: contractx
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U contractx_user"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - contractx_network

  # Neo4j Graph Database
  neo4j:
    image: neo4j:5.15-enterprise
    container_name: contractx_neo4j
    environment:
      NEO4J_AUTH: neo4j/your_neo4j_password
      NEO4J_ACCEPT_LICENSE_AGREEMENT: "yes"
    ports:
      - "7687:7687"
      - "7474:7474"
    volumes:
      - neo4j_data:/var/lib/neo4j/data
    healthcheck:
      test: ["CMD-SHELL", "wget --quiet --tries=1 --spider http://localhost:7474 || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - contractx_network

  # Qdrant Vector Database
  qdrant:
    image: qdrant/qdrant:latest
    container_name: contractx_qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      QDRANT_HTTP_PORT: 6333
      QDRANT_GRPC_PORT: 6334
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - contractx_network

  # FastAPI Application
  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: contractx_app
    depends_on:
      postgres:
        condition: service_healthy
      neo4j:
        condition: service_healthy
      qdrant:
        condition: service_healthy
    environment:
      GEMINI_API_KEY: ${GEMINI_API_KEY}
      DB_HOST: postgres
      DB_PORT: 5432
      DB_NAME: contractx
      DB_USER: contractx_user
      DB_PASSWORD: your_secure_password
      NEO4J_URI: bolt://neo4j:7687
      NEO4J_USER: neo4j
      NEO4J_PASSWORD: your_neo4j_password
      QDRANT_HOST: qdrant
      QDRANT_PORT: 6333
      PYTHONUNBUFFERED: 1
    ports:
      - "8000:8000"
    volumes:
      - ./uploads:/app/uploads
      - ./logs:/app/logs
      - ./visuals:/app/visuals
    networks:
      - contractx_network
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

volumes:
  postgres_data:
  neo4j_data:
  qdrant_data:

networks:
  contractx_network:
    driver: bridge
```

### Create Dockerfile

Create `Dockerfile` in project root:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create necessary directories
RUN mkdir -p uploads logs visuals chroma_db

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Start Services with Docker Compose

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check service health
docker-compose ps

# Stop services
docker-compose down

# Remove volumes (CAUTION: deletes data)
docker-compose down -v
```

---

## Configuration

### Environment Variables Details

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | Required | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | AI model for text/table/image analysis |
| `GEMINI_REQUEST_DELAY` | `3.0` | Delay between API requests (seconds) |
| `MAX_RETRIES` | `5` | API retry attempts |
| `RETRY_DELAY` | `60` | Delay before retry (seconds) |
| `DB_HOST` | `localhost` | PostgreSQL host |
| `DB_PORT` | `5432` | PostgreSQL port |
| `DB_NAME` | `contractx` | Database name |
| `DB_USER` | `contractx_user` | Database user |
| `DB_PASSWORD` | Required | Database password |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | Required | Neo4j password |
| `QDRANT_HOST` | `localhost` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `QDRANT_COLLECTION` | `contractx_documents` | Qdrant collection name |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `MAX_UPLOAD_SIZE` | `52428800` | Max file size (50MB) |
| `UPLOAD_DIR` | `uploads` | Upload directory path |
| `LOG_DIR` | `logs` | Log directory path |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## Database Setup

### PostgreSQL Setup

**Option 1: Docker (Recommended)**

```bash
docker run -d \
  --name contractx_postgres \
  -e POSTGRES_USER=contractx_user \
  -e POSTGRES_PASSWORD=your_password \
  -e POSTGRES_DB=contractx \
  -p 5432:5432 \
  -v postgres_data:/var/lib/postgresql/data \
  postgres:15-alpine
```

**Option 2: Manual Installation**

1. Download from https://www.postgresql.org/download/
2. Create database and user:

```sql
CREATE DATABASE contractx;
CREATE USER contractx_user WITH PASSWORD 'your_password';
ALTER ROLE contractx_user SET client_encoding TO 'utf8';
GRANT ALL PRIVILEGES ON DATABASE contractx TO contractx_user;
```

### Neo4j Setup

**Option 1: Docker (Recommended)**

```bash
docker run -d \
  --name contractx_neo4j \
  -p 7687:7687 \
  -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/your_password \
  -v neo4j_data:/var/lib/neo4j/data \
  neo4j:5.15-enterprise
```

**Option 2: Manual Installation**

1. Download from https://neo4j.com/download/
2. Start service and access http://localhost:7474
3. Change default password

### Initialize Database Tables

```bash
python
>>> from app.database.database import init_db
>>> init_db()
```

---

## RAG & Vector Store Setup

### Qdrant Installation

**Option 1: Docker (Recommended)**

```bash
docker run -d \
  --name contractx_qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v qdrant_data:/qdrant/storage \
  qdrant/qdrant:latest
```

**Option 2: Docker Compose**

```yaml
qdrant:
  image: qdrant/qdrant:latest
  ports:
    - "6333:6333"
    - "6334:6334"
  volumes:
    - qdrant_data:/qdrant/storage
```

**Option 3: Qdrant Cloud (Production)**

```python
# In config or RAGService initialization
rag_service = RAGService(
    use_cloud=True,
    qdrant_url="https://your-cluster.qdrant.io",
    qdrant_api_key="your-api-key"
)
```

### Verify Qdrant Connection

```bash
# Check health
curl http://localhost:6333/health

# List collections
curl http://localhost:6333/collections
```

---

## Knowledge Graph Setup

### Neo4j Connection Test

```bash
# Docker container
docker exec contractx_neo4j cypher-shell -u neo4j -p your_password "MATCH (n) RETURN count(n) as node_count"

# Or using Python
python -c "
from neo4j import GraphDatabase
driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password'))
with driver.session() as session:
    result = session.run('RETURN 1')
    print('Neo4j Connected!')
"
```

### Create Indexes for Performance

```cypher
# Text search on document names
CREATE INDEX idx_document_name IF NOT EXISTS FOR (d:Document) ON (d.id);

# Section search
CREATE INDEX idx_section_title IF NOT EXISTS FOR (s:Section) ON (s.heading);

# Entity search
CREATE INDEX idx_entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name);
```

# Entity search
CREATE INDEX idx_entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name);
```

---

## Complete Project Flow

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CONTRACTX SYSTEM ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────────────────────┘

                              USER / API CLIENT
                                      │
                    ┌─────────────────▼─────────────────┐
                    │     FastAPI Server                │
                    │   (app/main.py:8000)             │
                    └─────────────────┬─────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
┌──────────────────┐         ┌──────────────────┐        ┌──────────────────┐
│  PDF PROCESSOR   │         │  TEXT ANALYZER   │        │  IMAGE DETECTOR  │
│                  │         │                  │        │                  │
│ - PDF → Images   │         │ - Sections       │        │ - OpenCV Contour │
│ - Extract Pages  │         │ - Clauses        │        │ - Gemini Validate│
│ - DPI Conversion │         │ - Entities       │        │ - Chart Detection│
│                  │         │ - Summary        │        │ - Save Visuals   │
└────────┬─────────┘         └────────┬─────────┘        └────────┬─────────┘
         │                            │                           │
         │                            │                           │
         │  ┌──────────────────────────▼──────────────────────┐  │
         │  │       TABLE EXTRACTION SERVICE                  │  │
         │  │  (Gemini Vision + Merged Cell Handling)         │  │
         │  └──────────────────────┬───────────────────────────┘  │
         │                         │                              │
         └─────────────────────────┼──────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  AGGREGATION & VALIDATION   │
                    │  - Merge results            │
                    │  - Entity deduplication     │
                    │  - Overall summary          │
                    └──────────────┬───────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
        ▼                          ▼                          ▼
┌─────────────────┐       ┌──────────────────┐      ┌──────────────────┐
│  DATABASE STORE │       │  KNOWLEDGE GRAPH │      │  VECTOR INDEXING │
│                 │       │  (Neo4j)         │      │  (Qdrant RAG)    │
│ PostgreSQL      │       │                  │      │                  │
│                 │       │ - Document Node  │      │ - BGE Embeddings │
│ ✓ Full Results  │       │ - Page Nodes     │      │ - Chunks         │
│ ✓ Entities      │       │ - Section Graph  │      │ - Metadata       │
│ ✓ Text JSON     │       │ - Table Nodes    │      │ - Filtering      │
│ ✓ Summary       │       │ - Entity Links   │      │                  │
│ ✓ Metadata      │       │ - Relationships  │      │ Ready for RAG    │
│                 │       │                  │      │                  │
└────────┬────────┘       └────────┬─────────┘      └────────┬─────────┘
         │                         │                         │
         └─────────────────────────┼─────────────────────────┘
                                   │
                         DOCUMENT PROCESSING COMPLETE
                              Document ID
                           Stored & Indexed
                                   │
                    ┌──────────────▼──────────────┐
                    │   CHATBOT INFERENCE         │
                    │   (RAG + Knowledge Graph)   │
                    │                             │
                    │  User Question              │
                    │  ├─ Qdrant Search (RAG)     │
                    │  ├─ Neo4j Queries (KG)      │
                    │  ├─ Combine Results         │
                    │  └─ Gemini Answer           │
                    │                             │
                    └─────────────────────────────┘
```

### Detailed Processing Steps

#### Step 1: PDF Conversion & Page Extraction

```
FILE UPLOAD → PDF Bytes → extract_pages()
                ↓
            pdf2image.convert_from_bytes()
                ↓
        PIL Images (per page)
                ↓
        Pytesseract OCR (Local)
                ↓
     Page Data {pil_image, text}
```

#### Step 2: Page-by-Page Analysis (Parallel)

For each page, execute 3 parallel tasks:

**Task A: Text Analysis**
```
Page Image
    ↓
Gemini Vision (Text Extraction Prompt)
    ↓
Parse Response:
- Sections with IDs
- Clauses with IDs
- Sub-clauses
- Entities (buyer, seller, dates, deadlines, alerts, obligations, addresses)
    ↓
text_analysis_result
```

**Task B: Table Detection**
```
Page Image
    ↓
Gemini Vision (Table Extraction Prompt)
    ↓
Parse JSON:
- table_id, title, type
- headers, rows
- merged_cells handling
- continues_to_next_page flag
    ↓
table_extraction_result
```

**Task C: Visual Detection**
```
Page Image
    ↓
OpenCV Contour Detection
    ↓
Extract Visual Regions
    ↓
For each region:
  - Crop visual
  - Send to Gemini for validation
  - If valid: Save + Summarize
  - If invalid: Reject
    ↓
visual_detection_result
```

#### Step 3: Results Aggregation

```
Collect all page results
    ↓
Merge tables (continued_from_previous_page)
    ↓
Deduplicate entities across pages
    ↓
Build comprehensive summary (4-5 paragraphs)
    ↓
Create text_as_json structure
```

#### Step 4: Storage & Indexing

```
Database Storage (PostgreSQL)
├─ documents table (metadata, summary)
├─ cleaned_text (full text)
└─ text_as_json (structured JSON)

Knowledge Graph (Neo4j)
├─ Document node (root)
├─ Page nodes (HAS_PAGE)
├─ Section nodes (HAS_SECTION)
├─ Clause nodes (HAS_CLAUSE)
├─ Table nodes (HAS_TABLE)
├─ Visual nodes (HAS_VISUAL)
├─ Entity nodes (HAS_ENTITY)
└─ Relationships

Vector Store (Qdrant)
├─ Collection: contractx_documents
├─ Chunks (page, section, table, visual summaries)
├─ Embeddings (BGE-Large)
├─ Metadata (document_id, page, chunk_type)
└─ Filtering (document_id)
```

#### Step 5: Chatbot Query Processing

```
User Question
    ↓
├─ RAG Retrieval (Qdrant)
│  ├─ Embed question
│  ├─ Vector search
│  ├─ Retrieve top N chunks
│  └─ Re-rank by relevance
│
├─ Knowledge Graph Query (Neo4j)
│  ├─ Parse entities
│  ├─ Cypher query generation
│  ├─ Traverse relationships
│  └─ Extract relevant subgraph
│
└─ Answer Generation
   ├─ Combine RAG + KG results
   ├─ Create augmented context
   ├─ Call Gemini with context
   └─ Return answer with sources
```

---

## Project Structure

```
ContractX/
├── app/
│   ├── __init__.py
│   ├── main.py                          # FastAPI application & endpoints
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py                    # Configuration management
│   ├── database/
│   │   ├── __init__.py
│   │   ├── database.py                  # SQLAlchemy DB connection
│   │   ├── models.py                    # Document & LLMUsage models
│   │   └── schemas.py                   # Pydantic request/response schemas
│   ├── services/
│   │   ├── pdf_processor.py             # PDF→Images, Table detection
│   │   ├── text_analyzer.py             # Sections, clauses, entities extraction
│   │   ├── image_detector.py            # OpenCV + Gemini visual detection
│   │   ├── knowledge_graph_builder.py   # Neo4j graph construction
│   │   ├── database_service.py          # PostgreSQL operations
│   │   ├── rag_service.py               # Qdrant vector search & indexing
│   │   ├── chatbot_service.py           # RAG + KG combined chatbot
│   │   └── LLM_tracker.py               # Token/cost tracking
│   └── utils/
│       ├── __init__.py
│       └── file_handler.py              # File upload/cleanup utilities
├── uploads/                              # Uploaded PDFs (auto-created)
├── logs/                                 # Application logs (auto-created)
├── visuals/                              # Extracted visuals (auto-created)
├── .env                                  # Environment variables (CRITICAL)
├── .env.example                          # Example env template
├── requirements.txt                      # Python dependencies
├── docker-compose.yml                    # Multi-container orchestration
├── Dockerfile                            # FastAPI container image
├── README.md                             # This file
└── AUDIT_REPORT.md                       # Code audit & analysis
```

### Key Service Descriptions

| Service | Purpose | External Calls |
|---------|---------|-----------------|
| **pdf_processor.py** | PDF extraction, OCR, table detection | Gemini API |
| **text_analyzer.py** | Document structure, entities, summary | Gemini API |
| **image_detector.py** | Visual recognition, classification | OpenCV, Gemini API |
| **knowledge_graph_builder.py** | Neo4j graph construction | Neo4j Driver |
| **database_service.py** | Document storage & retrieval | PostgreSQL |
| **rag_service.py** | Vector indexing & semantic search | Qdrant, Sentence-Transformers |
| **chatbot_service.py** | Question answering | Neo4j, Qdrant, Gemini API |
| **LLM_tracker.py** | Token counting & cost tracking | None (local) |
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

### Docker Issues

#### Issue: `docker-compose up` fails with connection errors

**Solution:**
```bash
# 1. Check if Docker daemon is running
docker ps

# 2. Stop and remove existing containers
docker-compose down
docker system prune -a

# 3. Rebuild images
docker-compose build --no-cache

# 4. Start fresh
docker-compose up -d
```

#### Issue: Containers keep restarting

**Solution:**
```bash
# View logs to find the issue
docker-compose logs -f [service_name]

# Examples:
docker-compose logs -f app
docker-compose logs -f postgres
docker-compose logs -f neo4j
docker-compose logs -f qdrant
```

#### Issue: Port already in use

**Solution:**
```bash
# Find process using port
# Windows (PowerShell)
netstat -ano | findstr :8000

# Linux/Mac
lsof -i :8000

# Kill process or change port in docker-compose.yml
# Change "8000:8000" to "8001:8000"
```

#### Issue: Volume permissions error

**Solution:**
```bash
# Linux/Mac - fix permissions
docker-compose down
sudo chown -R $USER:$USER uploads/ logs/ visuals/
docker-compose up -d
```

### Qdrant Issues

#### Issue: Qdrant connection refused

**Solution:**
```bash
# Verify Qdrant is running
docker ps | grep qdrant

# Check Qdrant health
curl http://localhost:6333/health

# Restart Qdrant
docker-compose restart qdrant

# Check logs
docker-compose logs qdrant
```

#### Issue: Qdrant Collection not found

**Solution:**
```bash
# Collections are created on first document indexing
# If error persists, manually create:
curl -X PUT http://localhost:6333/collections/contractx_documents \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "size": 1024,
      "distance": "Cosine"
    }
  }'
```

#### Issue: Qdrant storage full or slow

**Solution:**
```bash
# Check disk usage
docker exec contractx_qdrant du -sh /qdrant/storage

# Clear old collections (careful!)
curl -X DELETE http://localhost:6333/collections/contractx_documents

# Increase volume size in docker-compose.yml
# Recreate with larger volume
docker-compose down -v
docker-compose up -d
```

#### Issue: Vector search returning no results

**Solution:**
```bash
# 1. Verify embeddings were indexed
curl http://localhost:6333/collections/contractx_documents/points

# 2. Check vector dimensions match (should be 1024 for BGE)
# 3. Re-index document:
POST /api/v1/extract-and-build-kg?dpi=350

# 4. Search with higher threshold:
curl -X POST http://localhost:8000/api/v1/rag/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "your search",
    "document_id": "doc-id",
    "n_results": 10,
    "score_threshold": 0.3
  }'
```

### Database Issues

#### Issue: PostgreSQL connection timeout

**Solution:**
```bash
# Check if postgres is healthy
docker-compose ps postgres

# View postgres logs
docker-compose logs postgres

# Restart postgres
docker-compose restart postgres

# Test connection
docker exec contractx_postgres psql -U contractx_user -d contractx -c "SELECT 1"
```

#### Issue: Database locked or corrupted

**Solution:**
```bash
# Backup data first!
docker exec contractx_postgres pg_dump -U contractx_user contractx > backup.sql

# Reset database
docker-compose down
docker-compose down -v  # WARNING: deletes data
docker-compose up -d postgres

# Restore from backup
docker exec -i contractx_postgres psql -U contractx_user contractx < backup.sql
```

#### Issue: Table creation failed

**Solution:**
```bash
# Initialize database manually
docker exec contractx_app python -c "from app.database.database import init_db; init_db()"

# Or via Python shell
python
>>> from app.database.database import init_db
>>> init_db()
```

### Neo4j Issues

#### Issue: Neo4j authentication failed

**Solution:**
```bash
# Verify credentials in .env
# Default: neo4j / neo4j (first login)

# Access Neo4j browser
# http://localhost:7474

# Reset password if needed
docker exec contractx_neo4j cypher-shell -u neo4j -p neo4j \
  "ALTER USER neo4j SET PASSWORD 'new_password'"

# Update .env and restart
docker-compose restart app
```

#### Issue: Graph database queries slow

**Solution:**
```bash
# Create indexes (already done in setup)
docker exec contractx_neo4j cypher-shell -u neo4j -p password \
  "CREATE INDEX idx_document IF NOT EXISTS FOR (d:Document) ON (d.id)"

# Monitor performance
# View Query Log: http://localhost:7474 → Profile

# Clear database if needed (CAREFUL!)
docker exec contractx_neo4j cypher-shell -u neo4j -p password \
  "MATCH (n) DETACH DELETE n"
```

### Gemini API Issues

#### Issue: 429 Quota exceeded error

**Solution:**
```bash
# Reduce request rate
# In .env, increase delays:
GEMINI_REQUEST_DELAY=5.0  # Was 3.0
MAX_RETRIES=3
RETRY_DELAY=120

# Or upgrade your plan: https://ai.google.dev/pricing

# Use local OCR for text extraction (free):
# Remove Gemini calls from pdf_processor._extract_page_data()
# Use pytesseract instead
```

#### Issue: Invalid API key error

**Solution:**
```bash
# Get fresh key from: https://aistudio.google.com/app/apikeys
# Update .env
GEMINI_API_KEY=new_key_here

# Verify it works
curl -H "Authorization: Bearer YOUR_KEY" \
  https://generativelanguage.googleapis.com/v1beta/models?key=YOUR_KEY
```

#### Issue: Model not found (gemini-2.0-flash)

**Solution:**
```bash
# Check available models
GEMINI_API_KEY=your_key python -c "
import google.generativeai as genai
genai.configure(api_key='your_key')
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)
"

# Update .env with available model
GEMINI_MODEL=gemini-2.5-flash  # or gemini-1.5-flash
```

### Common General Issues

#### Issue: `GEMINI_API_KEY not found`

**Solution:**
```bash
# Verify .env file exists
cat .env | grep GEMINI_API_KEY

# Check format (no quotes on key)
# WRONG: GEMINI_API_KEY="your_key"
# RIGHT: GEMINI_API_KEY=your_key

# Reload environment
docker-compose restart app
```

#### Issue: Asyncio Event Loop Error

**Solution:**
```bash
# Ensure reload=False in production
# In docker-compose.yml:
command: uvicorn app.main:app --host 0.0.0.0 --port 8000

# NOT with --reload (causes event loop issues)
```

#### Issue: Out of Memory (OOM)

**Solution:**
```bash
# Increase container memory limits in docker-compose.yml:
services:
  app:
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G

# Reduce document page batch size
# Edit main.py: BATCH_SIZE = 3 (default 5)

# Restart
docker-compose up -d
```

#### Issue: High Memory Usage

**Solution:**
- Reduce `MAX_CONCURRENT_GEMINI_CALLS` in image_detector.py (default: 1)
- Process smaller documents first
- Increase available system RAM
- Use pagination for large datasets

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

## Getting Started - Complete Setup Guide

### 1️⃣ Prerequisites Checklist

- [ ] Docker Desktop installed (https://www.docker.com/products/docker-desktop)
- [ ] Gemini API Key obtained (https://aistudio.google.com/app/apikeys)
- [ ] Git installed (https://git-scm.com/)
- [ ] 16GB+ RAM available
- [ ] At least 100GB free disk space

### 2️⃣ Quick Setup (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/yourusername/ContractX.git
cd ContractX

# 2. Copy environment template
cp .env.example .env

# 3. Edit .env with your values
# Minimally required:
# - GEMINI_API_KEY=your_key_here
# - DB_PASSWORD=secure_password
# - NEO4J_PASSWORD=secure_password

# 4. Start all services
docker-compose up -d

# 5. Wait for services to be healthy (30-60 seconds)
docker-compose ps

# 6. Test the API
curl http://localhost:8000/health
```

### 3️⃣ Test with Sample Document

```bash
# Upload a test PDF
curl -X POST "http://localhost:8000/api/v1/extract-and-build-kg?dpi=350" \
  -F "file=@sample_contract.pdf"

# Response will include:
# - document_id
# - Extracted text, tables, visuals
# - Database & graph status
# - RAG indexing status
```

### 4️⃣ Verify All Services

**PostgreSQL:**
```bash
docker exec contractx_postgres psql -U contractx_user -d contractx -c "SELECT 1"
# Output: 1 ✓
```

**Neo4j:**
```bash
curl -u neo4j:your_password http://localhost:7474/db/neo4j/
# Status: 200 OK ✓
```

**Qdrant:**
```bash
curl http://localhost:6333/health
# Response: {"status":"ok"} ✓
```

**FastAPI:**
```bash
curl http://localhost:8000/health
# Response: {"status":"healthy"} ✓
```

### 5️⃣ Access Web Interfaces

| Service | URL | Default Login |
|---------|-----|---|
| **API Docs** | http://localhost:8000/docs | N/A |
| **API RedDoc** | http://localhost:8000/redoc | N/A |
| **Neo4j Browser** | http://localhost:7474 | neo4j / password |
| **Qdrant Dashboard** | http://localhost:6333/dashboard | N/A |

### 6️⃣ Query Uploaded Document

```bash
# Get document ID from upload response
DOCUMENT_ID="a1b2c3d4-e5f6-g7h8-i9j0-k1l2m3n4o5p6"

# Ask chatbot
curl -X POST "http://localhost:8000/api/v1/chatbot/$DOCUMENT_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Who are the parties involved?",
    "include_rag": true,
    "include_kg": true
  }'
```

### 7️⃣ Monitor Processing

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f app       # FastAPI
docker-compose logs -f postgres  # Database
docker-compose logs -f neo4j     # Knowledge Graph
docker-compose logs -f qdrant    # Vector Store
```

### 8️⃣ Stop Services

```bash
# Stop all containers (keeps data)
docker-compose down

# Stop and remove all data (CAUTION!)
docker-compose down -v
```

---

## Detailed Workflow Example

### Complete End-to-End Example

```bash
#!/bin/bash

# 1. Start services
echo "🚀 Starting ContractX services..."
docker-compose up -d
sleep 30  # Wait for services to be healthy

# 2. Create test data
cat > test_config.json << 'EOF'
{
  "pdf_file": "contract.pdf",
  "dpi": 350,
  "enable_ocr": true
}
EOF

# 3. Upload document
echo "📄 Uploading document..."
RESPONSE=$(curl -s -X POST "http://localhost:8000/api/v1/extract-and-build-kg?dpi=350" \
  -F "file=@contract.pdf")

# Extract document ID
DOCUMENT_ID=$(echo $RESPONSE | grep -o '"document_id":"[^"]*' | cut -d'"' -f4)
echo "✅ Document uploaded: $DOCUMENT_ID"

# 4. Check processing status
echo "⏳ Processing document..."
sleep 10

# 5. Get document info
echo "📊 Document info:"
curl -s "http://localhost:8000/api/v1/documents/$DOCUMENT_ID" | jq .

# 6. Search in RAG
echo "🔍 Searching document..."
curl -s -X POST "http://localhost:8000/api/v1/rag/search" \
  -H "Content-Type: application/json" \
  -d "{
    \"query\": \"What are the payment terms?\",
    \"document_id\": \"$DOCUMENT_ID\",
    \"n_results\": 5
  }" | jq .

# 7. Ask chatbot
echo "💬 Asking chatbot..."
curl -s -X POST "http://localhost:8000/api/v1/chatbot/$DOCUMENT_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Who is the buyer and seller?",
    "include_rag": true,
    "include_kg": true
  }' | jq .

# 8. View logs
echo "📋 Recent logs:"
docker-compose logs --tail=20 app

echo "✨ Complete! Document processing finished."
```

---

## Production Deployment

### Docker Swarm / Kubernetes

For production, use container orchestration:

```bash
# Kubernetes deployment (example)
kubectl apply -f k8s/contractx-deployment.yaml
kubectl apply -f k8s/postgres-statefulset.yaml
kubectl apply -f k8s/neo4j-statefulset.yaml
kubectl apply -f k8s/qdrant-statefulset.yaml
```

### Environment Variables for Production

```env
# Database
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40

# API Rate Limiting
GEMINI_REQUEST_DELAY=2.0
MAX_RETRIES=3

# Security
DEBUG=false
AUTH_REQUIRED=true
CORS_ORIGINS=["https://yourdomain.com"]

# Logging
LOG_LEVEL=WARNING
SENTRY_DSN=your_sentry_dsn

# Monitoring
PROMETHEUS_ENABLED=true
```

### Database Backups

```bash
# PostgreSQL backup
docker exec contractx_postgres pg_dump -U contractx_user contractx > backup.sql

# Neo4j backup
docker exec contractx_neo4j neo4j-admin dump --to-path=/var/lib/neo4j/backups

# Qdrant backup (copy volume)
docker volume inspect contractx_qdrant_data
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
