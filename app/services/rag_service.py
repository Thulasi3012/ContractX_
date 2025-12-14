"""
Advanced RAG Service using Qdrant for vector storage
- Document-level isolation with document_id filtering
- BGE-Large embeddings for superior accuracy
- Production-ready with error handling
- Supports re-ranking and hybrid search
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict, Any, Optional
from uuid import uuid4
import json
from app.config.config import Config


class RAGService:
    """Advanced RAG service with Qdrant and document isolation"""
    
    def __init__(
    self,
    qdrant_host: Optional[str] = None,
    qdrant_port: Optional[int] = None,
    embedding_model: str = "BAAI/bge-large-en-v1.5",
    use_cloud: bool = False,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None
):
        """
        Initialize RAG service with Qdrant
        """
        # Use passed arguments, fallback to Config, then defaults
        self.qdrant_host = qdrant_host or Config.QDRANT_HOST or "localhost"
        self.qdrant_port = qdrant_port or Config.QDRANT_PORT or 6333

        # Initialize Qdrant client
        if use_cloud:
            if not qdrant_url or not qdrant_api_key:
                raise ValueError("Qdrant Cloud requires url and api_key!")
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            print(f"[OK] Connected to Qdrant Cloud: {qdrant_url}")
        else:
            self.client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
            print(f"[OK] Connected to Qdrant: {self.qdrant_host}:{self.qdrant_port}")

        # Initialize embedding model
        print(f"[LOADING] Embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        print(f"[OK] Embedding model loaded (dim={self.embedding_dim})")

        # Collection name
        self.collection_name = "contractx_documents"

        # Create collection if not exists
        self._ensure_collection()

        print(f"[OK] RAG Service initialized")
        print(f"  - Collection: {self.collection_name}")
        print(f"  - Embedding: {embedding_model}")
        print(f"  - Vector dim: {self.embedding_dim}")

    
    def _embed_text(self, text: str) -> List[float]:
        """Generate embedding for text"""
        return self.embedding_model.encode(text, normalize_embeddings=True).tolist()
    
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False
        )
        return embeddings.tolist()
    
    def index_document(
        self, 
        document_id: str, 
        document_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Index document in Qdrant with proper chunking
        
        Args:
            document_id: UUID from database
            document_data: Complete extraction results
            
        Returns:
            Indexing statistics
        """
        print(f"\n[RAG] Indexing document: {document_id}")
        
        chunks = []
        metadatas = []
        chunk_counter = 0
        
        def clean_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
            """Remove None values and ensure serializable types"""
            cleaned = {}
            for key, value in meta.items():
                if value is None:
                    continue
                # Keep native types for Qdrant
                if isinstance(value, (str, int, float, bool)):
                    cleaned[key] = value
                elif isinstance(value, (list, dict)):
                    cleaned[key] = json.dumps(value)
                else:
                    cleaned[key] = str(value)
            return cleaned
        
        # Process each page
        for page in document_data.get('pages', []):
            page_num = page['page_number']
            
            # 1. Text sections
            text_analysis = page.get('text_analysis', {})
            for section in text_analysis.get('sections', []):
                chunk_text = f"Section {section.get('heading_id', 'unknown')}: {section.get('heading', '')}\n"
                
                for sub_heading in section.get('sub_headings', []):
                    sub_heading_text = sub_heading.get('sub_heading', '')
                    if sub_heading_text:
                        chunk_text += f"  {sub_heading_text}\n"
                    
                    for clause in sub_heading.get('clauses', []):
                        clause_text = clause.get('clause', '')
                        if clause_text:
                            chunk_text += f"    Clause {clause.get('clause_id', 'unknown')}: {clause_text}\n"
                        
                        for sub_clause in clause.get('sub_clauses', []):
                            sub_clause_text = sub_clause.get('sub_clause', '')
                            if sub_clause_text:
                                chunk_text += f"      {sub_clause_text}\n"
                
                if chunk_text.strip():
                    chunks.append(chunk_text)
                    meta = {
                        "document_id": document_id,
                        "page_number": page_num,
                        "chunk_type": "section",
                        "section_heading": section.get('heading', ''),
                        "section_id": section.get('heading_id', 'unknown'),
                        "content": chunk_text
                    }
                    metadatas.append(clean_metadata(meta))
                    chunk_counter += 1
            
            # 2. Tables (convert to text)
            for table_idx, table in enumerate(page.get('tables', []), 1):
                table_text = f"Table {table_idx}: {table.get('table_title', 'Untitled')} ({table.get('table_type', 'unknown')})\n"
                table_text += f"Headers: {', '.join(str(h) for h in table.get('headers', []))}\n"
                
                for row_idx, row in enumerate(table.get('rows', []), 1):
                    row_str = ', '.join(str(cell) for cell in row if cell is not None)
                    table_text += f"Row {row_idx}: {row_str}\n"
                
                if table.get('has_merged_cells'):
                    merged_info = table.get('merged_cells', 'yes')
                    table_text += f"Note: Table has merged cells - {merged_info}\n"
                
                if table_text.strip():
                    chunks.append(table_text)
                    meta = {
                        "document_id": document_id,
                        "page_number": page_num,
                        "chunk_type": "table",
                        "table_id": table.get('table_id', f'table_{table_idx}'),
                        "table_title": table.get('table_title', ''),
                        "total_rows": table.get('total_rows', 0),
                        "total_columns": table.get('total_columns', 0),
                        "content": table_text
                    }
                    metadatas.append(clean_metadata(meta))
                    chunk_counter += 1
            
            # 3. Visuals (as descriptions)
            # 3. Visuals (as descriptions + OCR)
            for visual_idx, visual in enumerate(page.get('visuals', []), 1):
                if visual.get('not_visual'):
                    continue
                
                visual_text = f"Visual: {visual.get('type', 'unknown')} on page {page_num}\n"

                # High-level summary (LLM / vision summary)
                if visual.get('summary'):
                    visual_text += f"Summary: {visual['summary']}\n"

                # 🔥 RAW OCR / detected text (VERY IMPORTANT)
                if visual.get('extracted_text'):
                    visual_text += "Extracted Text:\n"
                    visual_text += f"{visual['extracted_text']}\n"

                # 🔥 Labels / objects / chart tags
                if visual.get('labels'):
                    visual_text += f"Labels: {', '.join(visual['labels'])}\n"

                # Image size
                if visual.get('width') and visual.get('height'):
                    visual_text += f"Size: {visual['width']}x{visual['height']}px\n"

                # Structured data (charts / tables detected inside image)
                if visual.get('data'):
                    visual_text += f"Structured Data: {str(visual['data'])}\n"

                
                if visual_text.strip():
                    chunks.append(visual_text)
                    meta = {
                        "document_id": document_id,
                        "page_number": page_num,
                        "chunk_type": "visual",
                        "visual_id": visual.get('visual_id', f'visual_{visual_idx}'),
                        "visual_type": visual.get('type', ''),
                        "has_summary": visual.get('summary') is not None,
                        "content": visual_text
                    }
                    metadatas.append(clean_metadata(meta))
                    chunk_counter += 1
            
            # 4. Page-level summary
            page_summary = text_analysis.get('summary', '')
            if page_summary:
                summary_text = f"Page {page_num} Summary: {page_summary}"
                chunks.append(summary_text)
                meta = {
                    "document_id": document_id,
                    "page_number": page_num,
                    "chunk_type": "summary",
                    "is_derived": True,
                    "content": summary_text
                }
                metadatas.append(clean_metadata(meta))
                chunk_counter += 1
        
        # 5. Global document summary
        overall_summary = document_data.get('overall_summary', {})
        if overall_summary:
            doc_summary_text = f"Document Summary: {overall_summary.get('summary', '')}\n"
            
            entities = overall_summary.get('entities', {})
            if entities.get('buyer_name'):
                doc_summary_text += f"Buyer: {entities['buyer_name']}\n"
            if entities.get('seller_name'):
                doc_summary_text += f"Seller: {entities['seller_name']}\n"
            if entities.get('dates'):
                doc_summary_text += f"Dates: {', '.join(str(d) for d in entities['dates'] if d)}\n"
            if entities.get('deadlines'):
                doc_summary_text += f"Deadlines: {', '.join(str(d) for d in entities['deadlines'] if d)}\n"
            if entities.get('alerts'):
                doc_summary_text += f"Alerts: {', '.join(str(a) for a in entities['alerts'] if a)}\n"
            
            chunks.append(doc_summary_text)
            meta = {
                "document_id": document_id,
                "page_number": 0,
                "chunk_type": "document_summary",
                "is_derived": True,
                "document_type": overall_summary.get('document_type', ''),
                "buyer": entities.get('buyer_name', ''),
                "seller": entities.get('seller_name', ''),
                "content": doc_summary_text
            }
            metadatas.append(clean_metadata(meta))
            chunk_counter += 1
        
        # Generate embeddings for all chunks
        if chunks:
            try:
                print(f"[EMBED] Generating embeddings for {len(chunks)} chunks...")
                embeddings = self._embed_batch(chunks)
                
                # Create points for Qdrant
                points = []
                for i, (embedding, metadata) in enumerate(zip(embeddings, metadatas)):
                    point = PointStruct(
                        id=uuid4().hex,
                        vector=embedding,
                        payload=metadata
                    )
                    points.append(point)
                
                # Upsert to Qdrant
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                
                print(f"[OK] Indexed {len(chunks)} chunks for document {document_id}")
                
                return {
                    "status": "success",
                    "document_id": document_id,
                    "total_chunks": len(chunks),
                    "chunk_types": {
                        "sections": sum(1 for m in metadatas if m.get('chunk_type') == 'section'),
                        "tables": sum(1 for m in metadatas if m.get('chunk_type') == 'table'),
                        "visuals": sum(1 for m in metadatas if m.get('chunk_type') == 'visual'),
                        "summaries": sum(1 for m in metadatas if m.get('chunk_type') in ['summary', 'document_summary'])
                    }
                }
            except Exception as e:
                print(f"[ERROR] Qdrant indexing error: {str(e)}")
                raise
        else:
            print(f"[WARNING] No chunks to index for document {document_id}")
            return {
                "status": "warning",
                "document_id": document_id,
                "total_chunks": 0,
                "message": "No content to index"
            }

    def search(
        self,
        query: str,
        document_id: Optional[str] = None,
        n_results: int = 8,
        score_threshold: float = 0.5,
        chunk_types: Optional[List[str]] = None,
        include_derived: bool = False
    ) -> List[Dict[str, Any]]:
        print(f"\n[SEARCH] Query: {query[:100]}...")
        if document_id:
            print(f"  - Document filter: {document_id}")

        query_vector = self._embed_text(query)

        must_conditions = []
        should_conditions = []
        must_not_conditions = []

        # Document-level isolation
        if document_id:
            must_conditions.append(
                FieldCondition(key="document_id", match=MatchValue(value=document_id))
            )

        # Chunk-type filtering (OR logic)
        if chunk_types:
            for chunk_type in chunk_types:
                should_conditions.append(
                    FieldCondition(key="chunk_type", match=MatchValue(value=chunk_type))
                )

        # Exclude derived content by default
        if not include_derived:
            must_not_conditions.append(
                FieldCondition(key="is_derived", match=MatchValue(value=True))
            )

        # Build final filter safely
        filter_kwargs = {}
        if must_conditions:
            filter_kwargs["must"] = must_conditions
        if should_conditions:
            filter_kwargs["should"] = should_conditions
            filter_kwargs["minimum_should_match"] = 1
        if must_not_conditions:
            filter_kwargs["must_not"] = must_not_conditions

        query_filter = Filter(**filter_kwargs) if filter_kwargs else None

        # Execute search
        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=n_results,
                query_filter=query_filter,
                with_payload=True,
                score_threshold=score_threshold
            )
            results = response.points if hasattr(response, 'points') else response
        except AttributeError:
            # Fallback for older API
            print("[INFO] Falling back to legacy search method")
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=n_results,
                query_filter=query_filter,
                score_threshold=score_threshold
            )

        print(f"[OK] Found {len(results)} results")

        # Format results
        retrieved = []
        for hit in results:
            retrieved.append({
                "content": hit.payload.get("content", ""),
                "metadata": {k: v for k, v in hit.payload.items() if k != "content"},
                "score": hit.score,
                "id": hit.id
            })

        return retrieved

    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete all chunks for a document
        
        Args:
            document_id: UUID of document to delete
            
        Returns:
            Success status
        """
        try:
            # Delete using filter
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
            )
            print(f"[OK] Deleted all chunks for document {document_id}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to delete document from RAG: {e}")
            return False
    
    def get_document_stats(self, document_id: str) -> Dict[str, Any]:
        """
        Get statistics for a specific document
        
        Args:
            document_id: UUID of document
            
        Returns:
            Statistics dictionary
        """
        try:
            # Scroll through points with filter
            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                ),
                limit=1000
            )
            
            if results:
                chunk_types = {}
                for point in results:
                    chunk_type = point.payload.get('chunk_type', 'unknown')
                    chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                
                return {
                    "document_id": document_id,
                    "total_chunks": len(results),
                    "chunk_types": chunk_types,
                    "indexed": True
                }
            else:
                return {
                    "document_id": document_id,
                    "total_chunks": 0,
                    "indexed": False
                }
        except Exception as e:
            print(f"[ERROR] Failed to get stats: {e}")
            return {
                "document_id": document_id,
                "error": str(e),
                "indexed": False
            }
    
    def clear_all(self):
        """Clear entire RAG vector store (use with caution!)"""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            self._ensure_collection()
            print("[OK] RAG store cleared")
        except Exception as e:
            print(f"[ERROR] Failed to clear RAG store: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall RAG statistics"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "total_chunks": collection_info.points_count,
                "collection_name": self.collection_name,
                "vector_dim": self.embedding_dim
            }
        except Exception as e:
            print(f"[ERROR] Failed to get stats: {e}")
            return {"error": str(e)}


# Usage Example
if __name__ == "__main__":
    # Initialize RAG service (local Qdrant)
    rag = RAGService(
        qdrant_host=Config.QDRANT_HOST,
        qdrant_port=Config.QDRANT_PORT,
        embedding_model="BAAI/bge-large-en-v1.5"
    )
    

    # Get stats
    stats = rag.get_stats()
    print(f"\nRAG Stats: {stats}")
    
    # Example: Search within a specific document
    results = rag.search(
        query="What are the payment terms?",
        document_id="your-document-uuid",
        n_results=5
    )
    
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Score: {result['score']:.3f}")
        print(f"Content: {result['content'][:200]}...")
        print(f"Metadata: {result['metadata']}")