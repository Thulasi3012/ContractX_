from typing import List, Dict, Any
from app.database.schemas import (
    TextLLMOutput, TableDetectionOutput, FinalTable, 
    Chunk, Section, Entities, ImageInfo
)
from collections import defaultdict
import logging
from datetime import datetime

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/merger_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MergerService:
    """Merges and resolves results from different processing stages"""
    
    def resolve_table_conflicts(
        self,
        text_results: List[TextLLMOutput],
        table_detections: List[TableDetectionOutput]
    ) -> List[Dict[str, Any]]:
        """
        Compare Text LLM and HuggingFace table detections and resolve conflicts
        
        Strategy:
        1. If both detect table on same page → confirmed (source: 'both')
        2. If only HF detects → use HF detection (source: 'hf_detection')
        3. If only Text LLM detects → use Text LLM (source: 'text_llm')
        
        Args:
            text_results: Results from Text LLM processing
            table_detections: Results from HuggingFace table detection
            
        Returns:
            List of resolved table metadata for Table LLM processing
        """
        logger.info("="*80)
        logger.info("Resolving Table Detection Conflicts")
        logger.info("="*80)
        
        resolved_tables = []
        
        # Build page-level mappings
        text_tables_by_page = defaultdict(list)
        hf_tables_by_page = defaultdict(list)
        
        # Map Text LLM tables by page
        for text_result in text_results:
            if text_result.tables.contains_table:
                for page in text_result.tables.pages:
                    text_tables_by_page[page].append({
                        'chunk_metadata': text_result.chunk_metadata,
                        'source': 'text_llm',
                        'count': text_result.tables.count
                    })
                    logger.info(f"Text LLM detected table on page {page}")
        
        # Map HF detections by page
        for detection in table_detections:
            for table in detection.tables:
                page = table['page']
                hf_tables_by_page[page].append({
                    'chunk_metadata': detection.chunk_metadata,
                    'table_metadata': table,
                    'source': 'hf_detection'
                })
                logger.info(f"HF detected table on page {page}: {table['table_id']}")
        
        # Get all unique pages with table detections
        all_pages = set(text_tables_by_page.keys()) | set(hf_tables_by_page.keys())
        
        logger.info(f"Total unique pages with table detections: {len(all_pages)}")
        logger.info(f"Pages: {sorted(all_pages)}")
        
        for page in sorted(all_pages):
            text_detected = page in text_tables_by_page
            hf_detected = page in hf_tables_by_page
            
            logger.info(f"\n--- Page {page} Analysis ---")
            logger.info(f"Text LLM detected: {text_detected}")
            logger.info(f"HF detected: {hf_detected}")
            
            if text_detected and hf_detected:
                # Both detected - high confidence
                logger.info("✓ BOTH DETECTED - High confidence")
                for hf_table in hf_tables_by_page[page]:
                    resolved_tables.append({
                        'page': page,
                        'source': 'both',
                        'confidence': 0.95,
                        'table_id': hf_table['table_metadata']['table_id'],
                        'metadata': hf_table['table_metadata'],
                        'chunk': self._get_chunk_for_page(text_results, page)
                    })
                    logger.info(f"  Added: {hf_table['table_metadata']['table_id']} (confidence: 0.95)")
            
            elif hf_detected and not text_detected:
                # Only HF detected - use HF
                logger.info("⚠ ONLY HF DETECTED - Using HF detection")
                for hf_table in hf_tables_by_page[page]:
                    conf = hf_table['table_metadata'].get('confidence', 0.7)
                    resolved_tables.append({
                        'page': page,
                        'source': 'hf_detection',
                        'confidence': conf,
                        'table_id': hf_table['table_metadata']['table_id'],
                        'metadata': hf_table['table_metadata'],
                        'chunk': self._get_chunk_for_page(table_detections, page)
                    })
                    logger.info(f"  Added: {hf_table['table_metadata']['table_id']} (confidence: {conf})")
            
            elif text_detected and not hf_detected:
                # Only Text LLM detected - send to Table LLM for validation
                logger.info("⚠ ONLY TEXT LLM DETECTED - Needs validation")
                for text_table in text_tables_by_page[page]:
                    resolved_tables.append({
                        'page': page,
                        'source': 'text_llm',
                        'confidence': 0.6,
                        'table_id': f"text_page_{page}_table",
                        'metadata': {
                            'page': page,
                            'table_id': f"text_page_{page}_table",
                            'detected_by': 'text_llm'
                        },
                        'chunk': self._get_chunk_for_page(text_results, page)
                    })
                    logger.info(f"  Added: text_page_{page}_table (confidence: 0.6)")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Total resolved tables: {len(resolved_tables)}")
        logger.info(f"Breakdown by source:")
        source_counts = defaultdict(int)
        for table in resolved_tables:
            source_counts[table['source']] += 1
        for source, count in source_counts.items():
            logger.info(f"  - {source}: {count}")
        logger.info("="*80)
        
        return resolved_tables
    
    def _get_chunk_for_page(self, results: List[Any], page: int) -> Chunk:
        """Find the chunk containing a specific page"""
        for result in results:
            chunk_meta = result.chunk_metadata
            if chunk_meta.page_start <= page <= chunk_meta.page_end:
                # Return the chunk from the result
                # Note: We need to reconstruct chunk from metadata
                return Chunk(
                    metadata=chunk_meta,
                    text="",  # Will be filled by caller if needed
                    images=[],
                    raw_pages=[]
                )
        
        # Return empty chunk if not found
        return None
    
    def merge_results(
        self,
        text_results: List[TextLLMOutput],
        final_tables: List[FinalTable],
        chunks: List[Chunk]
    ) -> Dict[str, Any]:
        """
        Merge all processing results into final document structure
        
        Args:
            text_results: Structured text from Text LLM
            final_tables: Validated tables from Table LLM
            chunks: Original chunks with metadata
            
        Returns:
            Complete document analysis result
        """
        
        # Merge sections from all chunks
        all_sections = []
        for result in text_results:
            all_sections.extend(result.sections)
        
        # Deduplicate and organize sections
        merged_sections = self._merge_sections(all_sections)
        
        # Merge entities from all chunks
        merged_entities = self._merge_entities([r.entities for r in text_results])
        
        # Merge image information
        merged_images = self._merge_images([r.images for r in text_results])
        
        # Sort tables by page number
        sorted_tables = sorted(final_tables, key=lambda t: t.page)
        
        # Calculate statistics
        total_pages = chunks[-1].metadata.page_end if chunks else 0
        
        final_result = {
            "document_id": f"doc_{chunks[0].metadata.chunk_id}" if chunks else "doc_unknown",
            "sections": [section.dict() for section in merged_sections],
            "entities": merged_entities.dict(),
            "tables": [table.dict() for table in sorted_tables],
            "images": merged_images.dict(),
            "metadata": {
                "total_pages": total_pages,
                "total_chunks": len(chunks),
                "total_sections": len(merged_sections),
                "total_tables": len(sorted_tables),
                "total_images": merged_images.count,
                "processing_summary": {
                    "text_llm_chunks_processed": len(text_results),
                    "tables_detected": len(sorted_tables),
                    "entities_extracted": bool(merged_entities.buyer_name or merged_entities.seller_name)
                }
            },
            "status": "success"
        }
        
        return final_result
    
    def _merge_sections(self, sections: List[Section]) -> List[Section]:
        """
        Merge and deduplicate sections from multiple chunks
        Handle overlapping content from chunk overlaps
        """
        # Simple deduplication by heading_id
        seen_ids = set()
        merged = []
        
        for section in sections:
            if section.heading_id not in seen_ids:
                merged.append(section)
                seen_ids.add(section.heading_id)
        
        return sorted(merged, key=lambda s: s.heading_id)
    
    def _merge_entities(self, entities_list: List[Entities]) -> Entities:
        """Merge entities from multiple chunks, taking first non-null values"""
        
        merged = Entities()
        all_dates = []
        all_alerts = []
        all_deadlines = []
        all_addresses = []
        
        for entities in entities_list:
            if not merged.buyer_name and entities.buyer_name:
                merged.buyer_name = entities.buyer_name
            
            if not merged.seller_name and entities.seller_name:
                merged.seller_name = entities.seller_name
            
            if not merged.objection_level and entities.objection_level:
                merged.objection_level = entities.objection_level
            
            all_dates.extend(entities.dates)
            all_alerts.extend(entities.alerts)
            all_deadlines.extend(entities.deadlines)
            all_addresses.extend(entities.addresses)
        
        # Deduplicate lists
        merged.dates = list(set(all_dates))
        merged.alerts = list(set(all_alerts))
        merged.deadlines = list(set(all_deadlines))
        merged.addresses = list(set(all_addresses))
        
        return merged
    
    def _merge_images(self, images_list: List[ImageInfo]) -> ImageInfo:
        """Merge image information from multiple chunks"""
        
        all_pages = []
        total_count = 0
        contains_image = False
        
        for img_info in images_list:
            if img_info.contains_image:
                contains_image = True
                all_pages.extend(img_info.pages)
                total_count += img_info.count
        
        return ImageInfo(
            contains_image=contains_image,
            pages=sorted(list(set(all_pages))),
            count=total_count
        )