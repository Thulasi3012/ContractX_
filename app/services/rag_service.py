"""
RAG Service using ChromaDB for vector storage
Supports document-level isolation with document_id filtering
"""

import chromadb
from chromadb.config import Settings
import google.generativeai as genai
import os
from typing import List, Dict, Any
import json

class RAGService:
    """RAG service with document isolation"""
    
    def __init__(self):
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="contractx_documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize Gemini for embeddings
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set!")
        
        genai.configure(api_key=api_key)
        
        print("[OK] RAG Service initialized")
        print(f"  - ChromaDB path: ./chroma_db")
        print(f"  - Collection: contractx_documents")
    
    def index_document(self, document_id: str, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Index document in RAG store with proper chunking
        
        Args:
            document_id: UUID from database
            document_data: Complete extraction results
            
        Returns:
            Indexing statistics
        """
        print(f"\n[RAG] Indexing document: {document_id}")
        
        chunks = []
        metadatas = []
        ids = []
        
        chunk_counter = 0
        
        def clean_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
            """Remove None values and convert to strings"""
            cleaned = {}
            for key, value in meta.items():
                if value is None:
                    continue
                # Convert to string if not already
                if isinstance(value, (list, dict)):
                    cleaned[key] = str(value)
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
                        "type": "section",
                        "section_heading": section.get('heading', ''),
                        "section_id": section.get('heading_id', 'unknown')
                    }
                    metadatas.append(clean_metadata(meta))
                    ids.append(f"{document_id}_page_{page_num}_section_{chunk_counter}")
                    chunk_counter += 1
            
            # 2. Tables (convert to text)
            for table_idx, table in enumerate(page.get('tables', []), 1):
                table_text = f"Table {table_idx}: {table.get('table_title', 'Untitled')} ({table.get('table_type', 'unknown')})\n"
                table_text += f"Headers: {', '.join(str(h) for h in table.get('headers', []))}\n"
                
                for row_idx, row in enumerate(table.get('rows', [])[:10], 1):  # First 10 rows
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
                        "type": "table",
                        "table_id": table.get('table_id', f'table_{table_idx}'),
                        "table_title": table.get('table_title', ''),
                        "rows": table.get('total_rows', 0),
                        "columns": table.get('total_columns', 0)
                    }
                    metadatas.append(clean_metadata(meta))
                    ids.append(f"{document_id}_page_{page_num}_table_{chunk_counter}")
                    chunk_counter += 1
            
            # 3. Visuals (as descriptions)
            for visual_idx, visual in enumerate(page.get('visuals', []), 1):
                if visual.get('not_visual'):
                    continue
                
                visual_text = f"Visual: {visual.get('type', 'unknown')} on page {page_num}\n"
                visual_text += f"Summary: {visual.get('summary', 'No summary available')}\n"
                visual_text += f"Location: {visual.get('bbox', [])}\n"
                
                if visual.get('width') and visual.get('height'):
                    visual_text += f"Size: {visual.get('width')}x{visual.get('height')}px\n"
                
                if visual.get('data'):
                    visual_text += f"Data: {str(visual.get('data'))}\n"
                
                if visual_text.strip():
                    chunks.append(visual_text)
                    meta = {
                        "document_id": document_id,
                        "page_number": page_num,
                        "type": "visual",
                        "visual_id": visual.get('visual_id', f'visual_{visual_idx}'),
                        "visual_type": visual.get('type', ''),
                        "has_summary": bool(visual.get('summary'))
                    }
                    metadatas.append(clean_metadata(meta))
                    ids.append(f"{document_id}_page_{page_num}_visual_{chunk_counter}")
                    chunk_counter += 1
            
            # 4. Page-level summary and entities
            page_summary = text_analysis.get('summary', '')
            if page_summary:
                chunks.append(f"Page {page_num} Summary: {page_summary}")
                meta = {
                    "document_id": document_id,
                    "page_number": page_num,
                    "type": "summary"
                }
                metadatas.append(clean_metadata(meta))
                ids.append(f"{document_id}_page_{page_num}_summary")
                chunk_counter += 1
        
        # 5. Global document entities and summary
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
            if entities.get('obligations'):
                doc_summary_text += f"Total Obligations: {len(entities['obligations'])}\n"
            
            chunks.append(doc_summary_text)
            meta = {
                "document_id": document_id,
                "page_number": 0,
                "type": "document_summary",
                "document_type": overall_summary.get('document_type', ''),
                "buyer": entities.get('buyer_name', ''),
                "seller": entities.get('seller_name', '')
            }
            metadatas.append(clean_metadata(meta))
            ids.append(f"{document_id}_document_summary")
            chunk_counter += 1
        
        # Store in ChromaDB
        if chunks:
            try:
                self.collection.add(
                    documents=chunks,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"[OK] Indexed {len(chunks)} chunks for document {document_id}")
                
                return {
                    "status": "success",
                    "document_id": document_id,
                    "total_chunks": len(chunks),
                    "chunk_types": {
                        "sections": sum(1 for m in metadatas if m['type'] == 'section'),
                        "tables": sum(1 for m in metadatas if m['type'] == 'table'),
                        "visuals": sum(1 for m in metadatas if m['type'] == 'visual'),
                        "summaries": sum(1 for m in metadatas if m['type'] in ['summary', 'document_summary'])
                    }
                }
            except Exception as e:
                print(f"[ERROR] ChromaDB indexing error: {str(e)}")
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
        document_id: str = None,
        n_results: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Search RAG index with optional document_id filter
        
        Args:
            query: Search query
            document_id: Optional UUID to filter by document
            n_results: Number of results to return
            
        Returns:
            List of relevant chunks with metadata
        """
        where_filter = {"document_id": document_id} if document_id else None
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )
        
        if not results['documents'] or not results['documents'][0]:
            return []
        
        retrieved = []
        for i, doc in enumerate(results['documents'][0]):
            retrieved.append({
                "content": doc,
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i] if results['distances'] else 0
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
            # Get all IDs for this document
            results = self.collection.get(
                where={"document_id": document_id}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                print(f"[OK] Deleted {len(results['ids'])} chunks for document {document_id}")
                return True
            else:
                print(f"[INFO] No chunks found for document {document_id}")
                return False
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
            results = self.collection.get(
                where={"document_id": document_id}
            )
            
            if results['ids']:
                chunk_types = {}
                for metadata in results['metadatas']:
                    chunk_type = metadata.get('type', 'unknown')
                    chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                
                return {
                    "document_id": document_id,
                    "total_chunks": len(results['ids']),
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
            # Delete collection and recreate
            self.client.delete_collection(name="contractx_documents")
            self.collection = self.client.get_or_create_collection(
                name="contractx_documents",
                metadata={"hnsw:space": "cosine"}
            )
            print("[OK] RAG store cleared")
        except Exception as e:
            print(f"[ERROR] Failed to clear RAG store: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall RAG statistics"""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection_name": self.collection.name
        }