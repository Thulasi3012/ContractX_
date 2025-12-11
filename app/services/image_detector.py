import os
import io
import json
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image
import cv2
import numpy as np
import google.generativeai as genai
from app.config.config import Config
from app.services.LLM_tracker import LLMUsageTracker


class ImageDetector:
    """
    Detects and analyzes visuals in document pages

    Detection: OpenCV-based contour detection (reliable bboxes)
    Validation: Gemini vision model (verify it's actually a visual)

    Detects: charts, flow diagrams, gantt charts, logos, signatures, maps, etc.
    Excludes: plain text, tables (unless visuals inside tables), headers/footers

    Flow:
    1. Detect visual regions using OpenCV contour detection
    2. Extract and crop visuals
    3. Send to Gemini in batches for validation & analysis
    4. Save validated visuals locally
    5. Return consolidated JSON
    """

    # CONFIGURATION: Maximum concurrent Gemini API calls
    MAX_CONCURRENT_GEMINI_CALLS = 1  # Process 1 visual at a time

    VISUAL_TYPES = [
        "chart", "bar_chart", "line_chart", "pie_chart", "scatter_chart",
        "flow_diagram", "flowchart", "process_diagram", "workflow_diagram",
        "gantt_chart", "timeline", "schedule",
        "logo", "brand_mark", "company_logo",
        "signature", "handwritten_signature",
        "map", "geographical_map", "location_map",
        "infographic", "illustration", "diagram",
        "graph", "network_diagram", "tree_diagram",
        "architectural_diagram", "technical_drawing",
        "organizational_chart", "hierarchy_chart",
        "venn_diagram", "bubble_chart", "heatmap",
        "screenshot", "image", "photo"
    ]

    def __init__(self):
        """Initialize image detector with Gemini"""
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
        self.output_dir = Path("visuals")
        self.output_dir.mkdir(exist_ok=True)
        self.request_count = 0
        self.last_request_time = 0
        self.processed_logos = set()  # Track logos to extract only once

        # Add semaphore for concurrent limiting
        self.semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_GEMINI_CALLS)

        print(f"[ImageDetector] Initialized with model: {Config.GEMINI_MODEL}")
        print(f"[ImageDetector] Detection method: OpenCV Contours + Gemini Validation")
        print(f"[ImageDetector] Output directory: {self.output_dir.absolute()}")
        print(f"[ImageDetector] Gemini processing: {self.MAX_CONCURRENT_GEMINI_CALLS} concurrent calls")

    async def _rate_limit_async(self):
        """
        Async rate limiting for better concurrency
        """
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < Config.GEMINI_REQUEST_DELAY:
            sleep_time = Config.GEMINI_REQUEST_DELAY - time_since_last
            await asyncio.sleep(sleep_time)

        self.last_request_time = time.time()
        self.request_count += 1

    async def detect_images(
        self,
        page_image: Image.Image,
        page_number: int,
        pdf_path: str = None,
        usage_tracker: LLMUsageTracker = None  # ADDED: Usage tracker parameter
    ) -> List[Dict]:
        """
        Main detection method - detects and analyzes visuals in a page

        Args:
            page_image: PIL Image of the page
            page_number: Page number in document
            pdf_path: Path to original PDF (for folder naming)
            usage_tracker: LLMUsageTracker instance for tracking API usage

        Returns:
            List of visual dictionaries with analysis
        """
        print(f"\n[ImageDetector] Processing page {page_number}...")

        # Step 1: Detect visual regions using OpenCV
        visual_regions = self._detect_visual_regions_cv(page_image, page_number)

        if not visual_regions:
            print(f"[ImageDetector] No visuals detected on page {page_number}")
            return []

        print(f"[ImageDetector] Found {len(visual_regions)} potential visual(s) using CV")

        # Step 2: Create output folder for this document
        doc_name = Path(pdf_path).stem if pdf_path else f"document_{int(time.time())}"
        doc_folder = self.output_dir / doc_name / "visuals"
        doc_folder.mkdir(parents=True, exist_ok=True)

        # Step 3: Process all visuals concurrently with semaphore limiting
        print(f"[ImageDetector] Processing {len(visual_regions)} visuals (max {self.MAX_CONCURRENT_GEMINI_CALLS} concurrent)...")

        tasks = []
        for idx, region in enumerate(visual_regions, start=1):
            task = self._process_single_visual(
                region, idx, page_number, page_image, doc_folder, usage_tracker  # ADDED: Pass tracker
            )
            tasks.append(task)

        # Use gather with semaphore
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful results
        validated_visuals = []
        for result in results:
            if isinstance(result, dict) and result:
                validated_visuals.append(result)
            elif isinstance(result, Exception):
                print(f"[ImageDetector] Error processing visual: {str(result)}")

        print(f"\n[ImageDetector] Page {page_number} complete: {len(validated_visuals)} visual(s) validated")
        return validated_visuals

    async def _process_single_visual(
        self,
        region: Dict,
        idx: int,
        page_number: int,
        page_image: Image.Image,
        doc_folder: Path,
        usage_tracker: LLMUsageTracker = None  # ADDED: Usage tracker parameter
    ) -> Optional[Dict]:
        """
        Process a single visual region with semaphore limiting

        Returns validated visual dict or None if rejected
        """
        visual_id = f"visual_{page_number}.{idx}"
        print(f"\n[ImageDetector] Processing {visual_id}...")

        # Crop visual from page
        cropped_visual = self._crop_region_pixels(page_image, region['bbox'])

        # More robust size validation
        min_width, min_height = 50, 50
        if cropped_visual.width < min_width or cropped_visual.height < min_height:
            print(f"[ImageDetector] ✗ {visual_id} rejected (too small: {cropped_visual.width}x{cropped_visual.height})")
            return None

        # Use semaphore to limit concurrent Gemini calls
        async with self.semaphore:
            # Validate and analyze with Gemini
            analysis = await self._analyze_visual_with_gemini(
                cropped_visual,
                visual_id,
                page_number,
                usage_tracker  # ADDED: Pass tracker
            )

        if not analysis or not analysis.get('is_valid_visual'):
            reason = analysis.get('reason', 'Unknown') if analysis else 'Analysis failed'
            print(f"[ImageDetector] ✗ {visual_id} rejected: {reason}")
            return None

        # Better logo detection logic
        visual_type = analysis.get('type', '').lower()
        if any(logo_type in visual_type for logo_type in ['logo', 'brand_mark', 'company_logo']):
            if self.processed_logos:
                print(f"[ImageDetector] Skipping duplicate logo on page {page_number}")
                return None
            else:
                self.processed_logos.add(visual_id)
                print(f"[ImageDetector] First logo detected: {visual_id}")

        # Save cropped visual
        visual_path = doc_folder / f"{visual_id}.png"
        try:
            cropped_visual.save(visual_path, "PNG")
        except Exception as e:
            print(f"[ImageDetector] Error saving visual: {str(e)}")
            return None

        # Convert bbox to percentage for consistency
        bbox_percent = self._pixels_to_percent(region['bbox'], page_image.size)

        # Build final visual record
        visual_record = {
            "visual_id": visual_id,
            "page_number": page_number,
            "bbox": bbox_percent,
            "type": analysis['type'],
            "summary": analysis['summary'],
            "data": analysis.get('data', {}),
            "tokens": {
                "input": analysis.get('input_tokens', 0),
                "output": analysis.get('output_tokens', 0)
            },
            "file_path": str(visual_path)
        }

        print(f"[ImageDetector] ✓ {visual_id} validated and saved")
        print(f"  Type: {analysis['type']}")
        print(f"  Size: {cropped_visual.width}x{cropped_visual.height}")
        print(f"  Summary length: {len(analysis['summary'])} chars")

        return visual_record

    def _detect_visual_regions_cv(
        self,
        page_image: Image.Image,
        page_number: int
    ) -> List[Dict]:
        """
        Detect visual regions using OpenCV contour detection

        Uses two methods:
        1. Edge-based detection (for line drawings, charts, diagrams)
        2. Color-based detection (for colored graphics, logos)

        Returns list of regions with pixel-based bounding boxes
        """
        # Convert PIL to OpenCV
        img_cv = cv2.cvtColor(np.array(page_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        height, width = gray.shape

        regions = []

        # Method 1: Edge-based detection
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # Configurable thresholds
            min_area = (width * height) * 0.005  # 0.5% of page
            max_area = (width * height) * 0.80   # 80% of page
            min_dimension = 80
            aspect_ratio = w / h if h > 0 else 0

            if (min_area < area < max_area and
                w > min_dimension and h > min_dimension and
                    0.1 < aspect_ratio < 10):

                regions.append({
                    'bbox': [x, y, x + w, y + h],
                    'area': area,
                    'confidence': 1.0,
                    'method': 'edge'
                })

        # Method 2: Color-based detection
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)

        # Detect non-white regions
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        mask_nonwhite = cv2.bitwise_not(mask_white)

        # Clean up with morphological operations
        kernel = np.ones((10, 10), np.uint8)
        mask_nonwhite = cv2.morphologyEx(mask_nonwhite, cv2.MORPH_CLOSE, kernel)

        contours_color, _ = cv2.findContours(mask_nonwhite, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours_color:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / h if h > 0 else 0

            min_area = (width * height) * 0.005
            max_area = (width * height) * 0.80
            min_dimension = 80

            if (min_area < area < max_area and
                w > min_dimension and h > min_dimension and
                    0.1 < aspect_ratio < 10):

                bbox_new = [x, y, x + w, y + h]

                # Check for duplicates with existing regions
                is_duplicate = any(
                    self._bbox_overlap(bbox_new, r['bbox']) > 0.5
                    for r in regions
                )

                if not is_duplicate:
                    regions.append({
                        'bbox': bbox_new,
                        'area': area,
                        'confidence': 1.0,
                        'method': 'color'
                    })

        # Sort by area (largest first)
        regions.sort(key=lambda x: x['area'], reverse=True)

        print(f"[ImageDetector] CV detected {len(regions)} potential visual regions")

        return regions

    def _bbox_overlap(self, bbox1: List[int], bbox2: List[int]) -> float:
        """
        Calculate IoU (Intersection over Union) between two bboxes
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _crop_region_pixels(self, page_image: Image.Image, bbox: List[int]) -> Image.Image:
        """
        Crop a region from the page image using pixel coordinates

        Args:
            page_image: Full page PIL Image
            bbox: [x1, y1, x2, y2] in pixels

        Returns:
            Cropped PIL Image
        """
        x1, y1, x2, y2 = bbox

        # Add small padding
        padding = 10
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(page_image.width, x2 + padding)
        y2 = min(page_image.height, y2 + padding)

        return page_image.crop((x1, y1, x2, y2))

    def _pixels_to_percent(self, bbox: List[int], page_size: Tuple[int, int]) -> List[float]:
        """
        Convert pixel bbox to percentage bbox
        """
        width, height = page_size
        x1, y1, x2, y2 = bbox

        # Ensure coordinates are within bounds
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        return [
            round((x1 / width) * 100, 2),
            round((y1 / height) * 100, 2),
            round((x2 / width) * 100, 2),
            round((y2 / height) * 100, 2)
        ]

    async def _analyze_visual_with_gemini(
        self,
        visual_image: Image.Image,
        visual_id: str,
        page_number: int,
        usage_tracker: LLMUsageTracker = None  # ADDED: Usage tracker parameter
    ) -> Optional[Dict]:
        """
        Send cropped visual to Gemini for validation and analysis
        
        UPDATED: Tracks LLM usage similar to TextAnalyzer
        """
        await self._rate_limit_async()

        visual_types_str = ", ".join(self.VISUAL_TYPES)

        prompt = f"""Analyze this cropped image and determine if it's a valid visual element.

Visual ID: {visual_id}
Page: {page_number}

VALID visual types include:
{visual_types_str}

INVALID (reject these):
- Plain text paragraphs
- Pure data tables without embedded visuals or charts
- Headers/footers with only text (unless they contain logos)
- Page numbers
- Empty/blank regions
- Borders or lines without content
- Text-only content

Task:
1. Determine if this is a VALID visual (true/false)
2. If valid, classify the exact type from the list above (be specific)
3. If valid, provide a comprehensive summary (2-3 sentences describing what you see)
4. If valid, extract any data/information visible in the visual

Return ONLY valid JSON (no markdown, no backticks):
{{
  "is_valid_visual": true/false,
  "type": "specific_type_from_list",
  "summary": "Detailed description of what the visual shows, including any labels, values, or key information",
  "data": {{
    "title": "Chart/diagram title if present",
    "key_points": ["point 1", "point 2"],
    "values": {{}},
    "text_content": "Any text visible in the visual",
    "labels": ["label1", "label2"]
  }}
}}

If NOT a valid visual, return:
{{
  "is_valid_visual": false,
  "reason": "Why it was rejected (e.g., 'only contains text', 'empty region', 'table without visuals')"
}}"""

        try:
            # ADDED: Track usage start
            if usage_tracker:
                usage_tracker.start_request(prompt)
            
            response = self.model.generate_content([prompt, visual_image])
            
            # ADDED: Track usage end
            if usage_tracker:
                usage_tracker.end_request(response)

            # Parse response
            result_text = response.text.strip()

            # More robust JSON extraction
            if "```" in result_text:
                parts = result_text.split("```")
                for part in parts:
                    if part.strip().startswith("json"):
                        result_text = part[4:].strip()
                        break
                    elif part.strip() and not part.strip().startswith("```"):
                        result_text = part.strip()
                        break

            result = json.loads(result_text)

            # Add token usage
            if hasattr(response, 'usage_metadata'):
                result['input_tokens'] = response.usage_metadata.prompt_token_count
                result['output_tokens'] = response.usage_metadata.candidates_token_count
            else:
                result['input_tokens'] = 0
                result['output_tokens'] = 0

            return result

        except json.JSONDecodeError as e:
            print(f"[ImageDetector] JSON parsing error for {visual_id}: {str(e)}")
            print(f"[ImageDetector] Raw response: {result_text[:200]}...")
            return None
        except Exception as e:
            print(f"[ImageDetector] Error analyzing {visual_id}: {str(e)}")
            return None

    def reset_logo_tracking(self):
        """Reset logo tracking for new document"""
        self.processed_logos.clear()

    def get_statistics(self) -> Dict:
        """Get detection statistics"""
        return {
            "total_requests": self.request_count,
            "output_directory": str(self.output_dir.absolute()),
            "logos_extracted": len(self.processed_logos),
            "supported_visual_types": len(self.VISUAL_TYPES),
            "detection_method": "OpenCV Contours + Edge Detection",
            "concurrent_limit": self.MAX_CONCURRENT_GEMINI_CALLS
        }