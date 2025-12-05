"""
Advanced Image/Visual Detection using OpenCV + Gemini Analysis
Detects charts, graphs, diagrams, Gantt charts with AI-powered summaries
"""

import cv2
import numpy as np
import fitz
import asyncio
import json
from typing import List, Dict, Any, Tuple
from PIL import Image
import google.generativeai as genai
from app.config.config import Config

class ImageDetector:
    """Detect and analyze non-table visuals with Gemini"""
    
    def __init__(self):
        # Detection parameters
        self.DPI = 250
        self.MIN_AREA = 20000
        self.MIN_W = 120
        self.MIN_H = 120
        self.EDGE_DENSITY_THRESHOLD = 0.003
        self.ENTROPY_THRESHOLD = 3.0
        
        # Permissive parameters
        self.MIN_AREA_PERMISSIVE = 8000
        self.EDGE_DENSITY_THRESHOLD_PERMISSIVE = 0.0015
        self.ENTROPY_THRESHOLD_PERMISSIVE = 2.5
        self.MIN_W_PERMISSIVE = 80
        self.MIN_H_PERMISSIVE = 60
        
        # Header/footer detection
        self.HEADER_FOOTER_MARGIN = 100
        self.MAX_PAGE_COVERAGE = 0.85
        
        # Gemini config
        self.LLM_MODEL = "gemini-2.5-flash"
        self.GEMINI_RETRY_SLEEP = 25
        self.MAX_PARALLEL = 4
        
        # Initialize Gemini
        api_key = Config.GEMINI_API_KEY
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY")
        
        genai.configure(api_key=api_key)
        self.llm = genai.GenerativeModel(self.LLM_MODEL)
        
        print("[OK] ImageDetector initialized with Gemini analysis")
        print(f"  - DPI: {self.DPI}")
        print(f"  - Min area (strict): {self.MIN_AREA}px")
        print(f"  - Min area (permissive): {self.MIN_AREA_PERMISSIVE}px")
        print(f"  - Gemini Model: {self.LLM_MODEL}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get detector status"""
        return {
            "model": "opencv-morphological-detection",
            "analyzer": "gemini-visual-analysis",
            "dpi": self.DPI,
            "min_area_strict": self.MIN_AREA,
            "min_area_permissive": self.MIN_AREA_PERMISSIVE
        }
    
    async def detect_images(
        self, 
        page_image: Image.Image,
        page_number: int,
        pdf_path: str = None
    ) -> List[Dict[str, Any]]:
        """
        Detect visual elements and send to Gemini for analysis (ASYNC)
        
        Returns:
            List of detected visuals with bbox, type, and AI-generated summary
        """
        print(f"\n[IMAGE DETECTOR] Processing page {page_number}...")
        
        # Convert PIL to OpenCV
        page_bgr = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)
        h, w = page_bgr.shape[:2]
        
        # Get text boxes to mask them out
        text_boxes = []
        if pdf_path:
            text_boxes = self._get_text_boxes(pdf_path, page_number - 1)
        
        # Apply text mask
        masked_gray, mask = self._apply_text_mask(page_bgr, text_boxes)
        
        # Detect visuals
        if self._page_is_text_heavy(page_bgr, text_boxes, threshold=0.90):
            print(f"  [i] Page {page_number} is >90% text, using permissive mode only")
            boxes = self._detect_visuals_permissive(page_bgr, mask)
        else:
            # Try strict detection first
            boxes = self._detect_visuals_strict(page_bgr, mask)
            
            # Fallback to permissive if nothing found
            if not boxes:
                print(f"  [i] No strict detections, trying permissive mode...")
                boxes = self._detect_visuals_permissive(page_bgr, mask)
        
        if not boxes:
            print(f"  [OK] No visuals detected on page {page_number}")
            return []
        
        print(f"  [OK] Detected {len(boxes)} candidate visual(s), sending to Gemini...")
        
        # AWAIT the async processing
        results = await self._process_visuals_async(
            page_number=page_number,
            page_bgr=page_bgr,
            boxes=boxes,
            page_height=h,
            page_width=w
        )
        
        return results
    
    async def _process_visuals_async(
        self,
        page_number: int,
        page_bgr: np.ndarray,
        boxes: List[List[int]],
        page_height: int,
        page_width: int
    ) -> List[Dict[str, Any]]:
        """Process visuals asynchronously with Gemini"""
        sem = asyncio.Semaphore(self.MAX_PARALLEL)
        results = []
        
        async def process_single_visual(bbox_idx: int, bbox: List[int]):
            async with sem:
                try:
                    visual_id = f"page_{page_number}_visual_{bbox_idx + 1}"
                    
                    # Crop and convert to bytes
                    x1, y1, x2, y2 = bbox
                    crop = page_bgr[y1:y2, x1:x2]
                    _, buf = cv2.imencode(".png", crop)
                    crop_bytes = buf.tobytes()
                    
                    # Pre-filter: Check if it's likely a table
                    if self._is_likely_table(crop):
                        print(f"    [i] {visual_id} detected as table, skipping")
                        return None
                    
                    # Send to Gemini
                    analysis = await self._analyze_visual_with_gemini(crop_bytes, visual_id)
                    
                    if analysis.get("not_visual"):
                        print(f"    [i] {visual_id} marked as not_visual by Gemini")
                        return None
                    
                    # Add metadata
                    analysis["visual_id"] = visual_id
                    analysis["page_number"] = page_number
                    analysis["bbox"] = bbox
                    analysis["width"] = x2 - x1
                    analysis["height"] = y2 - y1
                    analysis["area"] = (x2 - x1) * (y2 - y1)
                    
                    print(f"    [OK] {visual_id}: type={analysis.get('type')}, summary_len={len(analysis.get('summary', ''))}")
                    return analysis
                    
                except Exception as e:
                    print(f"    [ERROR] Failed to process bbox {bbox_idx}: {str(e)}")
                    return None
        
        # Create tasks for all boxes
        tasks = [process_single_visual(idx, bbox) for idx, bbox in enumerate(boxes)]
        
        # Gather results
        gathered = await asyncio.gather(*tasks, return_exceptions=False)
        
        # Filter out None results
        results = [r for r in gathered if r is not None]
        
        return results
    
    async def _analyze_visual_with_gemini(self, image_bytes: bytes, visual_id: str) -> Dict[str, Any]:
        """Send visual to Gemini for analysis"""
        prompt = f"""
Analyze this visual from a PDF document page.

CRITICAL RULES:
- Extract ONLY information clearly visible in the image
- Never infer, estimate, approximate, assume, or fabricate values
- Any partially unreadable element must be null, not guessed
- Do NOT reinterpret scales, ranges, dates, or numeric axes
- Ignore headers, footers, logos, watermarks, decorative elements
- If visual is small or dont understandable, respond with "unreadable": true
- If this is a TABLE (data in rows/columns), respond with "not_visual": true
- Allowed visuals: chart, graph, diagram, map, flowchart, gantt, image, signature
- For timelines/axes/bars/nodes: record only fully readable text and values
- Keep original label formatting
- If scale boundaries unclear: set to null
- If positional relationships exist: describe ONLY if explicitly shown

OUTPUT FORMAT - STRICT JSON ONLY (no markdown, no explanations):

{{
  "visual_id": "{visual_id}",
  "type": "<chart|graph|diagram|map|flowchart|gantt|image|signature|null>",
  "data": {{}},
  "summary": "<detailed explanation of what this visual shows, max 500 chars>",
  "unreadable": false,
  "not_visual": false
}}
"""
        
        try:
            contents = [
                {"mime_type": "image/png", "data": image_bytes},
                {"text": prompt}
            ]
            
            response = await self.llm.generate_content_async(contents=contents)
            raw_text = response.text or ""
            
            # Extract JSON
            start = raw_text.find("{")
            end = raw_text.rfind("}") + 1
            
            if start >= 0 and end > start:
                json_str = raw_text[start:end]
                analysis = json.loads(json_str)
            else:
                analysis = {
                    "type": None,
                    "data": {},
                    "summary": "Failed to parse Gemini response",
                    "unreadable": True,
                    "not_visual": True
                }
            
            # Ensure required fields
            analysis.setdefault("visual_id", visual_id)
            analysis.setdefault("type", None)
            analysis.setdefault("data", {})
            analysis.setdefault("summary", "")
            analysis.setdefault("unreadable", False)
            analysis.setdefault("not_visual", False)
            
            return self._convert_to_native(analysis)
            
        except Exception as e:
            print(f"    [ERROR] Gemini analysis failed: {str(e)}")
            return {
                "visual_id": visual_id,
                "type": None,
                "data": {},
                "summary": f"Analysis error: {str(e)}",
                "unreadable": True,
                "not_visual": False
            }
    
    def _is_likely_table(self, crop: np.ndarray) -> bool:
        """Pre-filter: Check if crop is likely a table"""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 120)
        
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            return False
        
        # Check for table-like patterns (grids)
        text_ratio = np.sum(gray < 200) / (h * w + 1e-12)
        edge_density = np.sum(edges > 0) / (h * w + 1e-12)
        
        # Tables typically have high text ratio and regular edge patterns
        return text_ratio > 0.5 and edge_density < 0.01
    
    def _get_text_boxes(self, pdf_path: str, page_index: int) -> List[List[int]]:
        """Get text box coordinates from PDF"""
        try:
            doc = fitz.open(pdf_path)
            page = doc.load_page(page_index)
            raw_blocks = page.get_text("blocks")
            
            boxes = []
            for b in raw_blocks:
                if len(b) >= 5:
                    x1, y1, x2, y2, text = b[0], b[1], b[2], b[3], b[4]
                    if text and str(text).strip():
                        boxes.append([int(x1), int(y1), int(x2), int(y2)])
            
            doc.close()
            return boxes
        except Exception as e:
            print(f"  [!] Error getting text boxes: {e}")
            return []
    
    def _apply_text_mask(
        self, 
        page_bgr: np.ndarray, 
        text_boxes: List[List[int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create mask to exclude text regions"""
        mask = np.ones(page_bgr.shape[:2], dtype=np.uint8) * 255
        
        for (x1, y1, x2, y2) in text_boxes:
            # Add padding
            pad_x = int((x2 - x1) * 0.02) + 2
            pad_y = int((y2 - y1) * 0.02) + 2
            
            xa = max(0, x1 - pad_x)
            ya = max(0, y1 - pad_y)
            xb = min(page_bgr.shape[1], x2 + pad_x)
            yb = min(page_bgr.shape[0], y2 + pad_y)
            
            cv2.rectangle(mask, (xa, ya), (xb, yb), 0, -1)
        
        gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
        masked = cv2.bitwise_and(gray, gray, mask=mask)
        
        return masked, mask
    
    def _page_is_text_heavy(
        self, 
        page_bgr: np.ndarray, 
        text_boxes: List[List[int]], 
        threshold: float = 0.75
    ) -> bool:
        """Check if page is mostly text"""
        h, w = page_bgr.shape[:2]
        page_area = h * w
        text_area = 0
        
        for (x1, y1, x2, y2) in text_boxes:
            text_area += max(0, (x2 - x1) * (y2 - y1))
        
        return (text_area / (page_area + 1e-12)) > threshold
    
    def _detect_visuals_strict(
        self, 
        page_bgr: np.ndarray, 
        mask: np.ndarray
    ) -> List[List[int]]:
        """Strict detection"""
        gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
        if mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=mask)
        
        th = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 51, 9
        )
        
        merge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
        merged = cv2.dilate(th, merge_kernel, iterations=2)
        
        contours, _ = cv2.findContours(
            merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        boxes = []
        h, w = page_bgr.shape[:2]
        
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area = bw * bh
            
            if area < self.MIN_AREA or bw < self.MIN_W or bh < self.MIN_H:
                continue
            
            bbox = [x, y, x + bw, y + bh]
            
            if self._is_in_header_footer(bbox, h) or self._is_full_page_detection(bbox, w, h):
                continue
            
            crop = page_bgr[y:y+bh, x:x+bw]
            
            if self._is_logo(crop) or self._is_watermark(crop) or not self._is_real_visual(crop):
                continue
            
            boxes.append(bbox)
        
        return self._merge_boxes(boxes)
    
    def _detect_visuals_permissive(
        self, 
        page_bgr: np.ndarray, 
        mask: np.ndarray
    ) -> List[List[int]]:
        """Permissive detection for difficult visuals"""
        gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
        
        if mask is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            relaxed_mask = cv2.erode(mask, kernel, iterations=1)
            gray = cv2.bitwise_and(gray, gray, mask=relaxed_mask)
        
        th = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 41, 7
        )
        
        merge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        merged = cv2.dilate(th, merge_kernel, iterations=1)
        
        contours, _ = cv2.findContours(
            merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        boxes = []
        h, w = page_bgr.shape[:2]
        
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area = bw * bh
            
            if area < self.MIN_AREA_PERMISSIVE or bw < self.MIN_W_PERMISSIVE or bh < self.MIN_H_PERMISSIVE:
                continue
            
            bbox = [x, y, x + bw, y + bh]
            
            if self._is_in_header_footer(bbox, h) or self._is_full_page_detection(bbox, w, h):
                continue
            
            crop = page_bgr[y:y+bh, x:x+bw]
            
            if self._is_logo(crop) and (bw < 220 and bh < 220):
                continue
            
            if self._is_watermark(crop) and area > 200000:
                continue
            
            if self._is_real_visual(
                crop,
                self.EDGE_DENSITY_THRESHOLD_PERMISSIVE,
                self.ENTROPY_THRESHOLD_PERMISSIVE,
                100000
            ):
                boxes.append(bbox)
        
        return self._merge_boxes(boxes, iou_thresh=0.1)
    
    def _is_in_header_footer(self, bbox: List[int], page_height: int) -> bool:
        """Check if bbox is in header/footer"""
        x1, y1, x2, y2 = bbox
        center_y = (y1 + y2) / 2
        return center_y < self.HEADER_FOOTER_MARGIN or center_y > (page_height - self.HEADER_FOOTER_MARGIN)
    
    def _is_full_page_detection(self, bbox: List[int], page_width: int, page_height: int) -> bool:
        """Check if bbox covers too much of page"""
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        page_area = page_width * page_height
        coverage = bbox_area / (page_area + 1e-12)
        return coverage > self.MAX_PAGE_COVERAGE
    
    def _is_logo(self, crop: np.ndarray) -> bool:
        """Detect if region is a logo"""
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            return False
        
        ar = w / h
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        entropy = self._shannon_entropy(gray)
        
        return entropy < 3.0 and (0.5 < ar < 1.8) and (w < 360 and h < 360)
    
    def _is_watermark(self, crop: np.ndarray) -> bool:
        """Detect if region is a watermark"""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        area = crop.shape[0] * crop.shape[1]
        
        if area == 0:
            return False
        
        edge_density = np.sum(edges > 0) / (area + 1e-12)
        return edge_density < 0.002 and area > 50000
    
    def _is_real_visual(
        self, 
        crop: np.ndarray,
        edge_thresh: float = None,
        entropy_thresh: float = None,
        area_thresh: int = 150000
    ) -> bool:
        """Check if region is a real visual element"""
        if edge_thresh is None:
            edge_thresh = self.EDGE_DENSITY_THRESHOLD
        if entropy_thresh is None:
            entropy_thresh = self.ENTROPY_THRESHOLD
        
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        area = crop.shape[0] * crop.shape[1]
        
        if area == 0:
            return False
        
        edge_density = np.sum(edges > 0) / (area + 1e-12)
        entropy = self._shannon_entropy(gray)
        
        return (edge_density > edge_thresh) or (entropy > entropy_thresh) or (area > area_thresh)
    
    def _shannon_entropy(self, gray: np.ndarray) -> float:
        """Calculate Shannon entropy"""
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        probs = hist / (hist.sum() + 1e-12)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0
    
    def _merge_boxes(
        self, 
        boxes: List[List[int]], 
        iou_thresh: float = 0.15
    ) -> List[List[int]]:
        """Merge overlapping boxes"""
        if not boxes:
            return []
        
        boxes_np = np.array(boxes)
        x1 = boxes_np[:, 0]
        y1 = boxes_np[:, 1]
        x2 = boxes_np[:, 2]
        y2 = boxes_np[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        idxs = list(range(len(boxes)))
        keep = []
        
        while idxs:
            i = idxs.pop(0)
            bx = [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])]
            keep.append(bx)
            
            remove = []
            for j in idxs:
                xx1 = max(x1[i], x1[j])
                yy1 = max(y1[i], y1[j])
                xx2 = min(x2[i], x2[j])
                yy2 = min(y2[i], y2[j])
                
                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)
                inter = w * h
                union = areas[i] + areas[j] - inter
                iou = inter / (union + 1e-12)
                
                if iou > iou_thresh:
                    remove.append(j)
                    keep[-1][0] = min(keep[-1][0], int(x1[j]))
                    keep[-1][1] = min(keep[-1][1], int(y1[j]))
                    keep[-1][2] = max(keep[-1][2], int(x2[j]))
                    keep[-1][3] = max(keep[-1][3], int(y2[j]))
            
            idxs = [k for k in idxs if k not in remove]
        
        return keep
    
    def _convert_to_native(self, obj: Any) -> Any:
        """Convert numpy types to native Python types"""
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, list):
            return [self._convert_to_native(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): self._convert_to_native(v) for k, v in obj.items()}
        return obj