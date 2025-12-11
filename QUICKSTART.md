# ContractX Quick Start Guide

**Get up and running in 5 minutes with Docker!**

---

## ⚡ Ultra-Quick Start

```bash
# 1. Clone & setup
git clone https://github.com/yourusername/ContractX.git && cd ContractX
cp .env.example .env

# 2. Edit .env (at minimum)
nano .env
# Add: GEMINI_API_KEY=your_key_here

# 3. Start everything
docker-compose up -d

# 4. Wait 30 seconds, then test
sleep 30
curl http://localhost:8000/health

# 5. Upload your first PDF
curl -X POST "http://localhost:8000/api/v1/extract-and-build-kg?dpi=350" \
  -F "file=@your_contract.pdf"
```

Done! You now have:
- ✅ FastAPI server (port 8000)
- ✅ PostgreSQL database (port 5432)
- ✅ Neo4j graph DB (port 7687)
- ✅ Qdrant vector store (port 6333)

---

## 🌐 Access Interfaces

| Service | URL | Purpose |
|---------|-----|---------|
| **Interactive API** | http://localhost:8000/docs | Try all endpoints |
| **Neo4j Browser** | http://localhost:7474 | View knowledge graph |
| **Qdrant Dashboard** | http://localhost:6333/dashboard | View vector index |

---

## 📄 Common Tasks

### Upload & Analyze PDF

```bash
curl -X POST "http://localhost:8000/api/v1/extract-and-build-kg?dpi=350" \
  -F "file=@contract.pdf"
```

**Response includes:**
- `document_id` - Save this for querying
- `text_analysis` - Sections, entities, summary
- `tables` - Extracted tables with merged cells
- `visuals` - Detected charts, diagrams, images

### Ask Questions About Document

```bash
curl -X POST "http://localhost:8000/api/v1/chatbot/DOCUMENT_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Who is the seller?",
    "include_rag": true,
    "include_kg": true
  }'
```

### Search Document

```bash
curl -X POST "http://localhost:8000/api/v1/rag/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "payment terms",
    "document_id": "DOCUMENT_ID",
    "n_results": 5
  }'
```

### List All Documents

```bash
curl http://localhost:8000/api/v1/documents
```

### Delete Document

```bash
curl -X DELETE "http://localhost:8000/api/v1/documents/DOCUMENT_ID"
```

---

## 🔧 Troubleshooting

### Services not starting?

```bash
# Check logs
docker-compose logs

# Rebuild and restart
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Quota exceeded error?

Increase delays in `.env`:
```env
GEMINI_REQUEST_DELAY=5.0    # Was 3.0
MAX_RETRIES=3
```

### Port already in use?

Change port in `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Use 8001 instead
```

### Out of memory?

Increase Docker memory:
- Docker Desktop → Settings → Resources → Memory: 8GB+

---

## 📊 Example Python Client

```python
import requests
import json

BASE = "http://localhost:8000"

# 1. Upload PDF
with open('contract.pdf', 'rb') as f:
    response = requests.post(
        f"{BASE}/api/v1/extract-and-build-kg?dpi=350",
        files={'file': f}
    )
    doc_id = response.json()['database']['document_id']
    print(f"Document ID: {doc_id}")

# 2. Ask question
answer = requests.post(
    f"{BASE}/api/v1/chatbot/{doc_id}",
    json={"question": "What are the key dates?"}
).json()
print(f"Answer: {answer['answer']}")

# 3. Search
results = requests.post(
    f"{BASE}/api/v1/rag/search",
    json={
        "query": "payment",
        "document_id": doc_id
    }
).json()
print(f"Found {len(results)} results")
```

---

## 📦 What Gets Extracted

| Component | Details |
|-----------|---------|
| **Text** | Sections, clauses, entities |
| **Tables** | Headers, rows, merged cells |
| **Visuals** | Charts, diagrams, logos |
| **Entities** | Buyer, seller, dates, deadlines |
| **Summary** | 4-5 paragraph AI overview |

---

## 🛑 Stop Services

```bash
# Keep data
docker-compose down

# Delete everything (CAREFUL!)
docker-compose down -v
```

---

## 📚 Next Steps

1. **Explore API Docs**: http://localhost:8000/docs
2. **Read Full README**: See main README.md
3. **Check Logs**: `docker-compose logs -f`
4. **Deploy to Production**: See README.md → Production Deployment

---

## ❓ Need Help?

- **API Issues**: Check `docker-compose logs app`
- **Database Issues**: Check `docker-compose logs postgres`
- **Graph Issues**: Check `docker-compose logs neo4j`
- **Vector Issues**: Check `docker-compose logs qdrant`

---

**ContractX v4.0** - Enterprise Document Intelligence System 🎉
