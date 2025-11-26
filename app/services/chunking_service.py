from typing import List, Dict, Any
from app.database.schemas import Chunk, ChunkMetadata

class ChunkingService:
    """Splits document pages into overlapping chunks"""
    
    def create_chunks(
        self, 
        pages: List[Dict[str, Any]], 
        chunk_size: int = 5, 
        overlap: int = 1
    ) -> List[Chunk]:
        """
        Create overlapping chunks from pages
        
        Args:
            pages: List of page dictionaries
            chunk_size: Number of pages per chunk
            overlap: Number of overlapping pages between chunks
            
        Returns:
            List of Chunk objects with metadata
            
        Example:
            20 pages, chunk_size=5, overlap=1:
            - Chunk 1: pages 1-5
            - Chunk 2: pages 5-9 (page 5 overlaps)
            - Chunk 3: pages 9-13
            - Chunk 4: pages 13-17
            - Chunk 5: pages 17-20
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        
        if overlap < 0:
            raise ValueError("overlap cannot be negative")
        
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        
        total_pages = len(pages)
        chunks = []
        chunk_id = 1
        
        # Calculate step size (how many pages to move forward each time)
        step = chunk_size - overlap
        
        # Create chunks with sliding window
        start_idx = 0
        
        while start_idx < total_pages:
            end_idx = min(start_idx + chunk_size, total_pages)
            
            # Get pages for this chunk
            chunk_pages = pages[start_idx:end_idx]
            
            # Combine text from all pages in chunk
            combined_text = '\n\n'.join([
                f"=== Page {page['page_number']} ===\n{page['text']}" 
                for page in chunk_pages
            ])
            
            # Collect all images from chunk pages
            chunk_images = []
            for page in chunk_pages:
                chunk_images.extend(page.get('images', []))
            
            # Create chunk metadata
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                page_start=chunk_pages[0]['page_number'],
                page_end=chunk_pages[-1]['page_number'],
                total_pages=len(chunk_pages)
            )
            
            # Create chunk object
            chunk = Chunk(
                metadata=metadata,
                text=combined_text,
                images=chunk_images,
                raw_pages=chunk_pages
            )
            
            chunks.append(chunk)
            
            # Move to next chunk
            chunk_id += 1
            start_idx += step
            
            # Handle last chunk case
            if start_idx >= total_pages:
                break
        
        return chunks
    
    def get_chunk_statistics(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """
        Get statistics about chunks
        
        Args:
            chunks: List of chunks
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_pages": 0,
                "avg_pages_per_chunk": 0,
                "total_images": 0
            }
        
        total_images = sum(len(chunk.images) for chunk in chunks)
        last_chunk = chunks[-1]
        total_pages = last_chunk.metadata.page_end
        
        return {
            "total_chunks": len(chunks),
            "total_pages": total_pages,
            "avg_pages_per_chunk": total_pages / len(chunks),
            "total_images": total_images,
            "chunks_info": [
                {
                    "chunk_id": chunk.metadata.chunk_id,
                    "page_range": f"{chunk.metadata.page_start}-{chunk.metadata.page_end}",
                    "pages": chunk.metadata.total_pages,
                    "images": len(chunk.images),
                    "text_length": len(chunk.text)
                }
                for chunk in chunks
            ]
        }