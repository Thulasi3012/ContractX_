from transformers import AutoImageProcessor, TableTransformerForObjectDetection
import torch
from PIL import Image
import fitz  # PyMuPDF
from typing import List, Dict, Any
import io
import time
from app.database.schemas import Chunk, TableDetectionOutput
import logging
from datetime import datetime

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/table_detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TableDetectionService:
    """Detects tables using Microsoft's table-transformer-detection model"""
    
    def __init__(self):
        """Initialize HuggingFace table detection model"""
        logger.info("="*80)
        logger.info("Initializing TableDetectionService")
        logger.info("="*80)
        
        try:
            model_name = "microsoft/table-transformer-detection"
            logger.info(f"Loading model: {model_name}")
            
            self.image_processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = TableTransformerForObjectDetection.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.confidence_threshold = 0.7
            
            logger.info(f"✓ Model loaded successfully")
            logger.info(f"Device: {self.device}")
            logger.info(f"Confidence Threshold: {self.confidence_threshold}")
            logger.info("="*80)
            
        except Exception as e:
            logger.error(f"✗ Error loading table detection model: {e}")
            raise
    
    async def detect_tables(self, chunk: Chunk) -> TableDetectionOutput:
        """
        Detect tables in document chunk using HuggingFace model
        
        Args:
            chunk: Document chunk with raw page data
            
        Returns:
            Table detection results with locations and metadata
        """
        chunk_id = chunk.metadata.chunk_id
        page_range = f"{chunk.metadata.page_start}-{chunk.metadata.page_end}"
        
        logger.info("="*80)
        logger.info(f"Detecting Tables in Chunk {chunk_id} (Pages {page_range})")
        logger.info("="*80)
        
        detected_tables = []
        
        try:
            for idx, page_data in enumerate(chunk.raw_pages):
                page_num = page_data['page_number']
                raw_page = page_data.get('raw_page')
                
                logger.info(f"Processing page {page_num} ({idx+1}/{len(chunk.raw_pages)})")
                
                if raw_page is None:
                    logger.warning(f"⚠ No raw page data for page {page_num} (likely DOCX)")
                    continue
                
                start_time = time.time()
                
                # Convert page to image
                page_image = self._page_to_image(raw_page)
                
                if page_image is None:
                    logger.warning(f"⚠ Failed to convert page {page_num} to image")
                    continue
                
                logger.info(f"  Image size: {page_image.size[0]}x{page_image.size[1]}")
                
                # Detect tables in the page image
                tables_on_page = self._detect_tables_in_image(page_image, page_num)
                
                elapsed = time.time() - start_time
                logger.info(f"  ✓ Detected {len(tables_on_page)} tables in {elapsed:.2f}s")
                
                if tables_on_page:
                    for table in tables_on_page:
                        logger.info(f"    - {table['table_id']}: confidence={table['confidence']}, bbox={table['bounding_box']}")
                
                detected_tables.extend(tables_on_page)
            
            logger.info(f"✓ Total tables detected in chunk {chunk_id}: {len(detected_tables)}")
            
            result = TableDetectionOutput(
                tables=detected_tables,
                chunk_metadata=chunk.metadata
            )
            
            logger.info("="*80)
            return result
            
        except Exception as e:
            logger.error(f"✗ Error in table detection for chunk {chunk_id}: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            return TableDetectionOutput(
                tables=[],
                chunk_metadata=chunk.metadata
            )
    
    def _page_to_image(self, page) -> Image.Image:
        """
        Convert PDF page to PIL Image
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            PIL Image
        """
        try:
            # Render page at high resolution for better detection
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            
            return image
            
        except Exception as e:
            logger.error(f"Error converting page to image: {e}")
            return None
    
    def _detect_tables_in_image(self, image: Image.Image, page_num: int) -> List[Dict[str, Any]]:
        """
        Detect tables in a single page image
        
        Args:
            image: PIL Image of the page
            page_num: Page number
            
        Returns:
            List of detected tables with bounding boxes
        """
        try:
            # Prepare image for model
            inputs = self.image_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run detection
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Process results
            target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
            results = self.image_processor.post_process_object_detection(
                outputs, 
                threshold=self.confidence_threshold, 
                target_sizes=target_sizes
            )[0]
            
            detected_tables = []
            
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                # Only keep table detections (label id for table is typically 0)
                if self.model.config.id2label[label.item()] == "table":
                    box_coords = [round(i, 2) for i in box.tolist()]
                    
                    detected_tables.append({
                        "page": page_num,
                        "table_id": f"hf_page_{page_num}_table_{len(detected_tables) + 1}",
                        "confidence": round(score.item(), 3),
                        "bounding_box": {
                            "x_min": box_coords[0],
                            "y_min": box_coords[1],
                            "x_max": box_coords[2],
                            "y_max": box_coords[3]
                        },
                        "dimensions": {
                            "width": box_coords[2] - box_coords[0],
                            "height": box_coords[3] - box_coords[1]
                        }
                    })
            
            return detected_tables
            
        except Exception as e:
            logger.error(f"Error detecting tables in image for page {page_num}: {e}")
            return []
    
    def get_table_regions(self, chunk: Chunk, detected_tables: List[Dict[str, Any]]) -> List[Image.Image]:
        """
        Extract table regions as separate images for further processing
        
        Args:
            chunk: Document chunk
            detected_tables: List of detected tables with bounding boxes
            
        Returns:
            List of cropped table images
        """
        logger.info(f"Extracting {len(detected_tables)} table regions as images")
        table_images = []
        
        try:
            for table in detected_tables:
                page_num = table['page']
                bbox = table['bounding_box']
                
                logger.info(f"Extracting table {table['table_id']} from page {page_num}")
                
                # Find the corresponding raw page
                raw_page = None
                for page_data in chunk.raw_pages:
                    if page_data['page_number'] == page_num:
                        raw_page = page_data.get('raw_page')
                        break
                
                if raw_page is None:
                    logger.warning(f"⚠ No raw page found for table {table['table_id']}")
                    continue
                
                # Convert page to image
                page_image = self._page_to_image(raw_page)
                
                if page_image:
                    # Crop table region
                    table_region = page_image.crop((
                        bbox['x_min'],
                        bbox['y_min'],
                        bbox['x_max'],
                        bbox['y_max']
                    ))
                    
                    logger.info(f"  ✓ Extracted region: {table_region.size[0]}x{table_region.size[1]}")
                    
                    table_images.append({
                        "image": table_region,
                        "metadata": table
                    })
            
            logger.info(f"✓ Successfully extracted {len(table_images)} table regions")
            return table_images
            
        except Exception as e:
            logger.error(f"✗ Error extracting table regions: {e}")
            return []