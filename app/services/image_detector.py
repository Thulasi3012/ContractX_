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
        Enhanced detector with multi-chart splitting, UI filtering, schematic handling, and Gantt chart fixes.

        Key improvements:
        - Splits dashboard-like pages into individual charts
        - Filters out UI elements (headers, search bars, nav)
        - Better Gantt chart extraction with axis alignment
        - Handles full-page schematics as single visuals
        - Filters component labels and sub-blocks
        - Saves rejected visuals for debugging
        """

        MAX_CONCURRENT_GEMINI_CALLS = 1

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
            "screenshot", "image", "photo", "network diagram",
            "circuit_diagram", "wiring_diagram", "electrical_schematic",
            "power_distribution_diagram", "system_diagram", "avionics_schematic"
        ]

        UI_REJECT_KEYWORDS = [
            "search bar", "navigation", "header", "footer", "menu", "toolbar",
            "button", "input field", "text box", "dropdown", "filter", "search box",
            "nav bar", "top bar", "title bar", "sidebar", "control panel"
        ]

        def __init__(self):
            genai.configure(api_key=Config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
            self.output_dir = Path("visuals")
            self.output_dir.mkdir(exist_ok=True)
            self.rejected_dir = self.output_dir / "rejected"
            self.rejected_dir.mkdir(exist_ok=True)
            self.request_count = 0
            self.last_request_time = 0
            self.processed_logos = set()
            self.semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_GEMINI_CALLS)

            # Counters for diagnostics
            self.counters = {
                "cv_candidates": 0,
                "rejected_too_small": 0,
                "rejected_gemini": 0,
                "rejected_table_cv": 0,
                "rejected_table_text": 0,
                "rejected_ui_element": 0,
                "validated_visuals": 0,
                "skipped_duplicate_logo": 0,
                "noise_rejected": 0,
                "total_requests": 0,
                "dashboard_splits": 0,
                "schematic_detected": 0,
                "component_blocks_filtered": 0
            }

            print(f"[ImageDetector] Initialized with model: {Config.GEMINI_MODEL}")
            print(
                f"[ImageDetector] Output directory: {self.output_dir.absolute()}")
            print(
                f"[ImageDetector] Rejected directory: {self.rejected_dir.absolute()}")
            print(
                f"[ImageDetector] Max concurrent Gemini calls: {self.MAX_CONCURRENT_GEMINI_CALLS}")

        # -----------------------
        # Public methods
        # -----------------------
        async def detect_images(
            self,
            page_image: Image.Image,
            page_number: int,
            pdf_path: str = None,
            usage_tracker: LLMUsageTracker = None
        ) -> List[Dict]:
            """
            Detect and analyze visuals on a single page image.
            Returns validated visuals (list of dicts).
            """
            print(f"\n[ImageDetector] Processing page {page_number}...")

            # First check if this is a full-page schematic
            schematic_check = await self._check_full_page_schematic(
                page_image, page_number, usage_tracker
            )

            if schematic_check["is_schematic"]:
                print(
                    f"[ImageDetector] Full-page schematic detected on page {page_number}")
                self.counters["schematic_detected"] += 1

                doc_name = Path(
                    pdf_path).stem if pdf_path else f"document_{int(time.time())}"
                doc_folder = self.output_dir / doc_name / "visuals"
                doc_folder.mkdir(parents=True, exist_ok=True)

                # Process entire page as single visual
                visual_record = await self._process_full_page_schematic(
                    page_image, page_number, doc_folder, schematic_check, usage_tracker
                )

                if visual_record:
                    return [visual_record]
                else:
                    return []

            # Not a schematic - proceed with normal detection
            regions = self._detect_visual_regions_cv(page_image, page_number)
            self.counters["cv_candidates"] += len(regions)

            if not regions:
                print(f"[ImageDetector] No CV candidates on page {page_number}")
                return []

            doc_name = Path(
                pdf_path).stem if pdf_path else f"document_{int(time.time())}"
            doc_folder = self.output_dir / doc_name / "visuals"
            doc_folder.mkdir(parents=True, exist_ok=True)

            rejected_folder = self.rejected_dir / doc_name
            rejected_folder.mkdir(parents=True, exist_ok=True)

            # Check if this is a dashboard page (multiple large regions)
            regions = await self._handle_dashboard_page(
                regions, page_image, page_number, doc_folder, rejected_folder, usage_tracker
            )

            # Create tasks for processing each region
            tasks = [
                self._process_single_visual(
                    region, idx, page_number, page_image, doc_folder, rejected_folder, usage_tracker)
                for idx, region in enumerate(regions, start=1)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            validated_visuals = []
            for res in results:
                if isinstance(res, dict) and res:
                    validated_visuals.append(res)
                elif isinstance(res, Exception):
                    print(f"[ImageDetector] Error in processing task: {res}")

            print(
                f"[ImageDetector] Page {page_number} done. Validated visuals: {len(validated_visuals)}")
            return validated_visuals

        # -----------------------
        # Schematic detection and handling
        # -----------------------
        async def _check_full_page_schematic(
            self,
            page_image: Image.Image,
            page_number: int,
            usage_tracker: LLMUsageTracker = None
        ) -> Dict:
            """
            Check if the entire page is a technical schematic/circuit diagram.
            Returns dict with is_schematic, type, title, figure_number.
            """
            await self._rate_limit_async()

            prompt = """Analyze this entire page and determine if it's a TECHNICAL SCHEMATIC/DIAGRAM.

    A TECHNICAL SCHEMATIC is:
    - Circuit diagram, wiring diagram, electrical schematic
    - Power distribution diagram, system diagram
    - Avionics schematic, control system diagram
    - Network topology diagram with interconnected components
    - Shows connections, wiring, signal flow between components
    - Usually spans the entire page as one unified technical drawing
    - Contains component labels, connection lines, symbols

    NOT a schematic:
    - Dashboard with multiple separate charts
    - Flowchart or process diagram (use standard diagram type)
    - Organization chart
    - Simple block diagram without technical connections

    If this IS a technical schematic:
    1. Extract the FIGURE NUMBER if visible (e.g., "Figure 002", "Fig. 2", "FIG-001")
    2. Extract the TITLE/DESCRIPTION (e.g., "ADIRS - Power Supply Distribution", "Hydraulic System Schematic")
    3. Identify the schematic type

    Return ONLY JSON:
    {
    "is_schematic": true/false,
    "schematic_type": "circuit_diagram" or "wiring_diagram" or "power_distribution_diagram" etc.,
    "figure_number": "Figure 002" or null,
    "title": "Full title text" or null,
    "reason": "brief explanation"
    }
    """

            try:
                if usage_tracker:
                    usage_tracker.start_request(prompt)

                response = self.model.generate_content([prompt, page_image])

                if usage_tracker:
                    usage_tracker.end_request(response)

                result_text = response.text.strip()

                # Extract JSON
                if "```" in result_text:
                    parts = result_text.split("```")
                    for part in parts:
                        if part.strip().startswith("json"):
                            result_text = part[4:].strip()
                            break

                result = json.loads(result_text)

                print(
                    f"[ImageDetector] Schematic check result: {result.get('is_schematic')} - {result.get('reason')}")

                return result

            except Exception as e:
                print(f"[ImageDetector] Error in schematic detection: {e}")
                return {"is_schematic": False, "reason": "error"}

        async def _process_full_page_schematic(
            self,
            page_image: Image.Image,
            page_number: int,
            doc_folder: Path,
            schematic_info: Dict,
            usage_tracker: LLMUsageTracker = None
        ) -> Optional[Dict]:
            """
            Process a full-page schematic as a single visual.
            """
            visual_id = f"visual_{page_number}.1"
            print(
                f"\n[ImageDetector] Processing full-page schematic {visual_id}...")

            # Get detailed analysis
            analysis = await self._analyze_schematic_with_gemini(
                page_image, visual_id, page_number, schematic_info, usage_tracker
            )

            if not analysis:
                print(f"[ImageDetector] ✗ {visual_id} schematic analysis failed")
                return None

            # Save the full page image
            visual_path = doc_folder / f"{visual_id}.png"
            try:
                page_image.save(visual_path, "PNG")
            except Exception as e:
                print(f"[ImageDetector] Error saving {visual_id}: {e}")
                return None

            # Full page bbox in percent
            bbox_percent = [0.0, 0.0, 100.0, 100.0]

            # Build visual record
            visual_record = {
                "visual_id": visual_id,
                "page_number": page_number,
                "bbox": bbox_percent,
                "type": schematic_info.get("schematic_type", "circuit_diagram"),
                "figure_number": schematic_info.get("figure_number"),
                "title": schematic_info.get("title"),
                "summary": analysis.get("summary"),
                "data": analysis.get("data", {}),
                "tokens": {
                    "input": analysis.get("input_tokens", 0),
                    "output": analysis.get("output_tokens", 0)
                },
                "file_path": str(visual_path),
                "is_full_page_schematic": True
            }

            self.counters["validated_visuals"] += 1
            print(
                f"[ImageDetector] ✓ {visual_id} full-page schematic validated and saved -> {visual_path.name}")
            if schematic_info.get("figure_number"):
                print(f"  Figure: {schematic_info.get('figure_number')}")
            if schematic_info.get("title"):
                print(f"  Title: {schematic_info.get('title')}")

            return visual_record

        async def _analyze_schematic_with_gemini(
            self,
            schematic_image: Image.Image,
            visual_id: str,
            page_number: int,
            schematic_info: Dict,
            usage_tracker: LLMUsageTracker = None
        ) -> Optional[Dict]:
            """
            Analyze a technical schematic in detail.
            """
            await self._rate_limit_async()

            schematic_type = schematic_info.get(
                "schematic_type", "circuit_diagram")
            figure_num = schematic_info.get("figure_number", "Unknown")
            title = schematic_info.get("title", "Unknown")

            prompt = f"""Analyze this technical schematic/diagram in detail.

    Schematic Type: {schematic_type}
    Figure Number: {figure_num}
    Title: {title}
    Visual ID: {visual_id}
    Page: {page_number}

    Provide a detailed analysis:

    1. **Main Components**: List the major components, modules, or systems shown (e.g., ADIRU 1, ADIRU 2, ADIRU 3, relays, power supplies)

    2. **Connections**: Describe the main connections, signal flows, or power distribution paths

    3. **Labels and Identifiers**: Extract key labels, component IDs, pin numbers, wire labels (e.g., 4DA3, 3DA3, 5FP3, 8FP, NORM, CAPT/3, ATT HDG)

    4. **Functional Description**: Briefly describe what this schematic represents and its purpose

    5. **Key Details**: Any important notes, test points, relay positions, or operational modes shown

    Return ONLY JSON:
    {{
    "summary": "2-3 sentence overview of the schematic",
    "data": {{
        "main_components": ["component1", "component2", ...],
        "connections": ["connection description 1", "connection description 2", ...],
        "labels": ["label1", "label2", ...],
        "functional_description": "What this system does",
        "key_details": ["detail1", "detail2", ...]
    }}
    }}
    """

            try:
                if usage_tracker:
                    usage_tracker.start_request(prompt)

                response = self.model.generate_content([prompt, schematic_image])

                if usage_tracker:
                    usage_tracker.end_request(response)

                result_text = response.text.strip()

                # Extract JSON
                if "```" in result_text:
                    parts = result_text.split("```")
                    for part in parts:
                        if part.strip().startswith("json"):
                            result_text = part[4:].strip()
                            break

                result = json.loads(result_text)

                # Attach token info
                if hasattr(response, "usage_metadata"):
                    result["input_tokens"] = getattr(
                        response.usage_metadata, "prompt_token_count", 0)
                    result["output_tokens"] = getattr(
                        response.usage_metadata, "candidates_token_count", 0)
                else:
                    result["input_tokens"] = 0
                    result["output_tokens"] = 0

                return result

            except Exception as e:
                print(
                    f"[ImageDetector] Error analyzing schematic {visual_id}: {e}")
                return None

        # -----------------------
        # Dashboard detection and splitting
        # -----------------------
        async def _handle_dashboard_page(
            self,
            regions: List[Dict],
            page_image: Image.Image,
            page_number: int,
            doc_folder: Path,
            rejected_folder: Path,
            usage_tracker: LLMUsageTracker = None
        ) -> List[Dict]:
            """
            Detect if page is a dashboard with multiple charts and split if needed.
            Returns updated regions list.
            """
            if not regions:
                return regions

            # Check if we have a very large region (>70% of page) with multiple visual clusters
            page_area = page_image.width * page_image.height
            large_regions = [r for r in regions if r['area'] > page_area * 0.70]

            if not large_regions:
                return regions

            large_region = large_regions[0]

            # First check: is this truly a dashboard or a single full-page visual?
            cropped = self._crop_region_pixels(page_image, large_region['bbox'])
            is_dashboard = await self._is_dashboard_layout(cropped, page_number, usage_tracker)

            if not is_dashboard:
                print(
                    f"[ImageDetector] Large region is a single full-page visual, not splitting")
                return regions

            print(
                f"[ImageDetector] Dashboard detected, attempting to split into individual charts...")
            self.counters["dashboard_splits"] += 1

            # Split the large region into sub-regions
            sub_regions = self._split_dashboard_region(
                page_image, large_region['bbox'])

            if len(sub_regions) > 1:
                print(
                    f"[ImageDetector] Split dashboard into {len(sub_regions)} sub-regions")
                # Remove the original large region and add sub-regions
                regions = [r for r in regions if r != large_region]
                regions.extend(sub_regions)

            return regions

        async def _is_dashboard_layout(
            self,
            image: Image.Image,
            page_number: int,
            usage_tracker: LLMUsageTracker = None
        ) -> bool:
            """
            Use Gemini to determine if an image is a dashboard with multiple charts
            or a single full-page visual.
            """
            await self._rate_limit_async()

            prompt = """Analyze this image and determine if it's a DASHBOARD or a SINGLE VISUAL.

    A DASHBOARD contains:
    - Multiple separate charts/graphs side by side or stacked
    - Different data visualizations in distinct sections
    - Clear visual separation between components

    A SINGLE VISUAL is:
    - One unified chart/diagram/infographic
    - One Gantt chart (even if large)
    - One flowchart or process diagram
    - One map or illustration
    - One circuit/wiring diagram or schematic (even if complex)

    Return ONLY JSON:
    {"is_dashboard": true/false, "reason": "brief explanation"}
    """

            try:
                if usage_tracker:
                    usage_tracker.start_request(prompt)

                response = self.model.generate_content([prompt, image])

                if usage_tracker:
                    usage_tracker.end_request(response)

                result_text = response.text.strip()

                # Extract JSON
                if "```" in result_text:
                    parts = result_text.split("```")
                    for part in parts:
                        if part.strip().startswith("json"):
                            result_text = part[4:].strip()
                            break

                result = json.loads(result_text)
                is_dash = result.get("is_dashboard", False)
                reason = result.get("reason", "")

                print(f"[ImageDetector] Dashboard check: {is_dash} - {reason}")
                return is_dash

            except Exception as e:
                print(f"[ImageDetector] Error in dashboard detection: {e}")
                return False

        def _split_dashboard_region(self, page_image: Image.Image, bbox: List[int]) -> List[Dict]:
            """
            Split a dashboard region into individual chart regions using visual clustering.
            """
            x1, y1, x2, y2 = bbox
            crop = page_image.crop((x1, y1, x2, y2))
            img_cv = cv2.cvtColor(np.array(crop.convert("RGB")), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

            # Use adaptive thresholding to find content regions
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 21, 10)

            # Find connected components
            kernel = np.ones((15, 15), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=3)

            contours, _ = cv2.findContours(
                dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            sub_regions = []
            crop_h, crop_w = gray.shape
            # At least 5% of the cropped region
            min_area = (crop_w * crop_h) * 0.05

            for contour in contours:
                cx, cy, cw, ch = cv2.boundingRect(contour)
                area = cw * ch

                if area > min_area and cw > 100 and ch > 100:
                    # Convert back to page coordinates
                    abs_x1 = x1 + cx
                    abs_y1 = y1 + cy
                    abs_x2 = x1 + cx + cw
                    abs_y2 = y1 + cy + ch

                    sub_regions.append({
                        'bbox': [abs_x1, abs_y1, abs_x2, abs_y2],
                        'area': area,
                        'confidence': 1.0,
                        'method': 'dashboard_split'
                    })

            # Sort by position (top to bottom, left to right)
            sub_regions.sort(key=lambda r: (r['bbox'][1], r['bbox'][0]))

            return sub_regions if len(sub_regions) > 1 else []

        # -----------------------
        # Single visual processing
        # -----------------------
        async def _process_single_visual(
            self,
            region: Dict,
            idx: int,
            page_number: int,
            page_image: Image.Image,
            doc_folder: Path,
            rejected_folder: Path,
            usage_tracker: LLMUsageTracker = None
        ) -> Optional[Dict]:
            visual_id = f"visual_{page_number}.{idx}"
            print(f"\n[ImageDetector] Processing {visual_id}...")

            cropped_visual = self._crop_region_pixels(page_image, region['bbox'])

            # Reject tiny regions early
            min_width, min_height = 50, 50
            if cropped_visual.width < min_width or cropped_visual.height < min_height:
                self.counters["rejected_too_small"] += 1
                self._save_rejected(cropped_visual, visual_id,
                                    rejected_folder, "too_small")
                print(
                    f"[ImageDetector] ✗ {visual_id} rejected (too small: {cropped_visual.width}x{cropped_visual.height})")
                return None

            # Check if this is a component block (small labeled box inside schematic)
            if self._is_component_block(cropped_visual):
                self.counters["component_blocks_filtered"] += 1
                self._save_rejected(cropped_visual, visual_id,
                                    rejected_folder, "component_block")
                print(
                    f"[ImageDetector] ✗ {visual_id} rejected (component block/label)")
                return None

            # Send to Gemini with concurrency limit
            async with self.semaphore:
                analysis = await self._analyze_visual_with_gemini(
                    cropped_visual, visual_id, page_number, usage_tracker
                )

            if not analysis:
                self.counters["rejected_gemini"] += 1
                self._save_rejected(cropped_visual, visual_id,
                                    rejected_folder, "no_analysis")
                print(f"[ImageDetector] ✗ {visual_id} rejected (no analysis)")
                return None

            if not analysis.get("is_valid_visual", False):
                self.counters["rejected_gemini"] += 1
                reason = analysis.get("reason", "Unknown")
                self._save_rejected(cropped_visual, visual_id,
                                    rejected_folder, f"gemini_{reason}")
                print(
                    f"[ImageDetector] ✗ {visual_id} rejected by Gemini: {reason}")
                return None

            # Check for UI elements
            visual_type = (analysis.get("type") or "").lower()
            summary = (analysis.get("summary") or "").lower()

            if self._is_ui_element(visual_type, summary):
                self.counters["rejected_ui_element"] += 1
                self._save_rejected(cropped_visual, visual_id,
                                    rejected_folder, "ui_element")
                print(
                    f"[ImageDetector] ✗ {visual_id} rejected (UI element: {visual_type})")
                return None

            # Logo duplicate handling
            if any(logo_key in visual_type for logo_key in ["logo", "brand_mark", "company_logo"]):
                if self.processed_logos:
                    self.counters["skipped_duplicate_logo"] += 1
                    self._save_rejected(cropped_visual, visual_id,
                                        rejected_folder, "duplicate_logo")
                    print(f"[ImageDetector] Skipping duplicate logo {visual_id}")
                    return None
                else:
                    self.processed_logos.add(visual_id)
                    print(f"[ImageDetector] First logo kept: {visual_id}")

            # Table filtering (post-Gemini)
            summary_text = summary + " " + json.dumps(analysis.get("data", {}))
            if self._is_pure_table_from_text(summary_text):
                self.counters["rejected_table_text"] += 1
                self._save_rejected(cropped_visual, visual_id,
                                    rejected_folder, "pure_table_text")
                print(
                    f"[ImageDetector] ✗ {visual_id} rejected (pure table by text summary)")
                return None

            # CV gridline check
            page_arr = np.array(page_image.convert("RGB"))
            if self._is_grid_table(page_arr, region['bbox']):
                self.counters["rejected_table_cv"] += 1
                self._save_rejected(cropped_visual, visual_id,
                                    rejected_folder, "grid_table_cv")
                print(
                    f"[ImageDetector] ✗ {visual_id} rejected (pure grid-like table detected by CV)")
                return None

            # Save visual file
            visual_path = doc_folder / f"{visual_id}.png"
            try:
                cropped_visual.save(visual_path, "PNG")
            except Exception as e:
                print(f"[ImageDetector] Error saving {visual_id}: {e}")
                return None

            bbox_percent = self._pixels_to_percent(region['bbox'], page_image.size)

            # Build final record
            visual_record = {
                "visual_id": visual_id,
                "page_number": page_number,
                "bbox": bbox_percent,
                "type": analysis.get("type"),
                "summary": analysis.get("summary"),
                "data": analysis.get("data", {}),
                "tokens": {
                    "input": analysis.get("input_tokens", 0),
                    "output": analysis.get("output_tokens", 0)
                },
                "file_path": str(visual_path)
            }

            self.counters["validated_visuals"] += 1
            print(
                f"[ImageDetector] ✓ {visual_id} validated and saved -> {visual_path.name}")
            print(
                f"  Type: {analysis.get('type')}, Size: {cropped_visual.width}x{cropped_visual.height}")
            return visual_record

        # -----------------------
        # Component block detection
        # -----------------------
        def _is_component_block(self, image: Image.Image) -> bool:
            """
            Detect if image is a component block/label box (small labeled rectangle).
            These are typically:
            - Small boxes with 1-3 words (PITOT, STATIC, TAT, ADIRU, NORM, CAPT/3, etc.)
            - High text-to-image ratio
            - Simple rectangular shape with text
            - No complex visual content
            """
            # Size check - component blocks are usually small
            if image.width > 300 or image.height > 200:
                return False

            # Very small images are likely labels
            if image.width < 150 and image.height < 80:
                # Check if mostly text
                img_cv = cv2.cvtColor(
                    np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

                # Simple threshold
                _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

                # Calculate text area (dark pixels)
                text_pixels = cv2.countNonZero(binary)
                total_pixels = image.width * image.height
                text_ratio = text_pixels / total_pixels

                # If >15% is text and image is small, likely a label
                if text_ratio > 0.15:
                    return True

            # Check for simple rectangular structure
            img_cv = cv2.cvtColor(
                np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # Count edge pixels
            edge_pixels = cv2.countNonZero(edges)
            total_pixels = image.width * image.height

            # Simple boxes have few edges relative to size
            # Complex diagrams have many edges
            edge_ratio = edge_pixels / total_pixels

            if edge_ratio < 0.15:  # Very simple structure
                return True

            return False

        # -----------------------
        # UI element detection
        # -----------------------
        def _is_ui_element(self, visual_type: str, summary: str) -> bool:
            """
            Detect if the visual is a UI element (header, search bar, nav, etc.)
            that should not be classified as a visual.
            """
            combined_text = f"{visual_type} {summary}".lower()

            return any(keyword in combined_text for keyword in self.UI_REJECT_KEYWORDS)

        def _save_rejected(self, image: Image.Image, visual_id: str, rejected_folder: Path, reason: str):
            """
            Save rejected visuals for debugging purposes.
            """
            try:
                # Sanitize reason for filename
                reason_safe = reason.replace(" ", "_").replace("/", "_")[:50]
                filename = f"{visual_id}_{reason_safe}.png"
                save_path = rejected_folder / filename
                image.save(save_path, "PNG")
                print(f"[ImageDetector] Saved rejected visual: {filename}")
            except Exception as e:
                print(
                    f"[ImageDetector] Error saving rejected visual {visual_id}: {e}")

        # -----------------------
        # OpenCV candidate detection
        # -----------------------
        def _detect_visual_regions_cv(self, page_image: Image.Image, page_number: int) -> List[Dict]:
            """
            Detect candidate regions using edge-based & color-based methods.
            """
            img_cv = cv2.cvtColor(
                np.array(page_image.convert("RGB")), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape

            regions: List[Dict] = []

            # Edge-based detection
            edges = cv2.Canny(gray, 50, 150)
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=2)
            contours, _ = cv2.findContours(
                dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                min_area = (width * height) * 0.005  # 0.5%
                max_area = (width * height) * 0.98
                min_dim = 80
                aspect_ratio = w / h if h > 0 else 0

                if (min_area < area < max_area and w > min_dim and h > min_dim and 0.1 < aspect_ratio < 10):
                    regions.append({
                        'bbox': [x, y, x + w, y + h],
                        'area': area,
                        'confidence': 1.0,
                        'method': 'edge'
                    })

            # Color-based detection (non-white regions)
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            mask_white = cv2.inRange(hsv, lower_white, upper_white)
            mask_nonwhite = cv2.bitwise_not(mask_white)
            kernel = np.ones((10, 10), np.uint8)
            mask_nonwhite = cv2.morphologyEx(
                mask_nonwhite, cv2.MORPH_CLOSE, kernel)
            contours_color, _ = cv2.findContours(
                mask_nonwhite, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours_color:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                min_area = (width * height) * 0.005
                max_area = (width * height) * 0.80
                min_dim = 80
                aspect_ratio = w / h if h > 0 else 0

                if (min_area < area < max_area and w > min_dim and h > min_dim and 0.1 < aspect_ratio < 10):
                    bbox_new = [x, y, x + w, y + h]
                    is_duplicate = any(self._bbox_overlap(
                        bbox_new, r['bbox']) > 0.5 for r in regions)
                    if not is_duplicate:
                        regions.append({
                            'bbox': bbox_new,
                            'area': area,
                            'confidence': 1.0,
                            'method': 'color'
                        })

            # Sort by area (largest first)
            regions.sort(key=lambda r: r['area'], reverse=True)
            print(
                f"[ImageDetector] CV detected {len(regions)} potential visual regions")
            return regions

        # -----------------------
        # Utility helpers
        # -----------------------
        def _bbox_overlap(self, bbox1: List[int], bbox2: List[int]) -> float:
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2
            x1_i = max(x1_1, x1_2)
            y1_i = max(y1_1, y1_2)
            x2_i = min(x2_1, x2_2)
            y2_i = min(y2_1, y2_2)
            if x2_i <= x1_i or y2_i <= y1_i:
                return 0.0
            intersection = (x2_i - x1_i) * (y2_i - y1_i)
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union = area1 + area2 - intersection
            return intersection / union if union > 0 else 0.0

        def _crop_region_pixels(self, page_image: Image.Image, bbox: List[int]) -> Image.Image:
            x1, y1, x2, y2 = bbox
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(page_image.width, x2 + padding)
            y2 = min(page_image.height, y2 + padding)
            return page_image.crop((x1, y1, x2, y2))

        def _pixels_to_percent(self, bbox: List[int], page_size: Tuple[int, int]) -> List[float]:
            width, height = page_size
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            return [round((x1 / width) * 100, 2), round((y1 / height) * 100, 2),
                    round((x2 / width) * 100, 2), round((y2 / height) * 100, 2)]

        # -----------------------
        # Gemini integration
        # -----------------------
        async def _rate_limit_async(self):
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < Config.GEMINI_REQUEST_DELAY:
                await asyncio.sleep(Config.GEMINI_REQUEST_DELAY - time_since_last)
            self.last_request_time = time.time()
            self.request_count += 1
            self.counters["total_requests"] += 1

        async def _analyze_visual_with_gemini(
            self,
            visual_image: Image.Image,
            visual_id: str,
            page_number: int,
            usage_tracker: LLMUsageTracker = None
        ) -> Optional[Dict]:
            await self._rate_limit_async()

            visual_types_str = ", ".join(self.VISUAL_TYPES)
            prompt = f"""Analyze this cropped image and determine if it's a valid visual element.

    Visual ID: {visual_id}
    Page: {page_number}

    VALID visual types include:
    {visual_types_str}

    STRICT RULES (must follow):
    1. UI ELEMENTS: If this is a search bar, navigation header, menu, toolbar, or any UI chrome → respond is_valid_visual: false, reason: "UI element"

    2. COMPONENT LABELS: If this is just a small labeled box containing 1-3 words (like "PITOT", "STATIC", "TAT", "ADIRU", "NORM", "CAPT/3") with no other visual content → respond is_valid_visual: false, reason: "component label"

    3. TABLES: If the image contains rows/columns, grid lines, tabular structure, OR text arranged in a matrix → this is a TABLE.
    - If the region is a pure table (no embedded charts/figures/images) → respond is_valid_visual: false and reason: "pure table".
    - Do not classify tables as charts/diagrams/infographics if they are purely tabular.
    - If visual elements exist inside a table (chart, icon, image, bar inside a cell), classify as a valid visual.

    4. GANTT CHARTS: For Gantt charts specifically:
    - Extract the EXACT timeline shown on the X-axis (months/dates visible)
    - For each activity/task bar:
        * Find where the bar STARTS on the timeline (align with X-axis)
        * Find where the bar ENDS on the timeline (align with X-axis)
        * Record milestones if marked with diamonds/circles
    - Return data with precise start/end aligned to the visible scale
    - Example format: {{"tasks": [{{"name": "Activity A", "start": "Jan 2024", "end": "Mar 2024", "milestones": ["Feb 2024"]}}]}}

    Task:
    1) Return valid JSON only (no markdown).
    2) Provide is_valid_visual (true/false). If true, fill 'type' from the list above.
    3) If true, provide a 2-3 sentence summary describing the visual and a 'data' object with any extracted text/values/labels.
    - For Gantt charts: extract timeline scale and task start/end dates aligned to that scale
    - For bar charts: extract categories and values
    - For all charts: extract axis labels, legends, and data points
    4) If false, return a 'reason' field explaining the rejection.

    Return ONLY JSON, e.g.:
    {{"is_valid_visual": true/false, "type": "...", "summary": "...", "data": {{...}}}}
    """

            try:
                if usage_tracker:
                    usage_tracker.start_request(prompt)

                response = self.model.generate_content([prompt, visual_image])

                if usage_tracker:
                    usage_tracker.end_request(response)

                result_text = response.text.strip()

                # Robust JSON extraction
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

                # Attach token info
                if hasattr(response, "usage_metadata"):
                    result["input_tokens"] = getattr(
                        response.usage_metadata, "prompt_token_count", 0)
                    result["output_tokens"] = getattr(
                        response.usage_metadata, "candidates_token_count", 0)
                else:
                    result["input_tokens"] = 0
                    result["output_tokens"] = 0

                return result

            except json.JSONDecodeError as e:
                print(f"[ImageDetector] JSON parsing error for {visual_id}: {e}")
                print(
                    f"[ImageDetector] Raw response (truncated): {result_text[:300]}")
                return None
            except Exception as e:
                print(f"[ImageDetector] Error analyzing {visual_id}: {e}")
                return None

        # -----------------------
        # Table detection helpers
        # -----------------------
        def _is_pure_table_from_text(self, summary_text: str) -> bool:
            """
            Conservative check: if summary mentions table/grid/rows/columns WITHOUT visual keywords.
            """
            text = (summary_text or "").lower()

            table_indicators = ["table", "grid", "rows",
                                "columns", "cells", "tabular", "header row"]
            visual_indicators = [
                "chart", "graph", "diagram", "flow", "bar", "plot", "line",
                "pie", "icon", "image", "illustration", "gantt", "timeline", "figure"
            ]

            # If any explicit visual keyword found → not a pure table
            if any(v in text for v in visual_indicators):
                return False

            # If table-related words found and no visual indicators → pure table
            if any(t in text for t in table_indicators):
                return True

            return False

        def _is_grid_table(self, page_arr: np.ndarray, bbox: List[int]) -> bool:
            """
            Detect if a bbox is a PURE table via dense, regularly spaced gridlines.
            """
            x1, y1, x2, y2 = bbox
            h, w = page_arr.shape[:2]

            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                return False

            crop = page_arr[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # Hough line detection
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                    threshold=120, minLineLength=30, maxLineGap=8)
            if lines is None or len(lines) < 6:
                return False

            horizontal_positions = []
            vertical_positions = []

            for l in lines[:, 0]:
                x1_l, y1_l, x2_l, y2_l = l
                if abs(y1_l - y2_l) <= 3 and abs(x2_l - x1_l) >= 20:
                    horizontal_positions.append((y1_l + y2_l) // 2)
                elif abs(x1_l - x2_l) <= 3 and abs(y2_l - y1_l) >= 20:
                    vertical_positions.append((x1_l + x2_l) // 2)

            horizontal_positions = sorted(set(horizontal_positions))
            vertical_positions = sorted(set(vertical_positions))

            # Conservative thresholds: many horizontals AND many verticals
            if len(horizontal_positions) < 4 or len(vertical_positions) < 4:
                return False

            # Check spacing regularity
            horizontal_diffs = np.diff(horizontal_positions) if len(
                horizontal_positions) > 1 else np.array([])
            if horizontal_diffs.size == 0:
                return False

            mean_diff = float(np.mean(horizontal_diffs))
            std_diff = float(np.std(horizontal_diffs))

            if mean_diff > 5 and (std_diff / mean_diff) < 0.6:
                vertical_diffs = np.diff(vertical_positions) if len(
                    vertical_positions) > 1 else np.array([])
                if vertical_diffs.size > 0:
                    v_mean = float(np.mean(vertical_diffs))
                    v_std = float(np.std(vertical_diffs))
                    if v_mean > 5 and (v_std / v_mean) < 0.8:
                        return True
                else:
                    return True

            return False

        # -----------------------
        # Utilities for external use
        # -----------------------
        def reset_logo_tracking(self):
            self.processed_logos.clear()

        def get_statistics(self) -> Dict:
            stats = {
                "output_directory": str(self.output_dir.absolute()),
                "rejected_directory": str(self.rejected_dir.absolute()),
                "logos_extracted": len(self.processed_logos),
                "supported_visual_types": len(self.VISUAL_TYPES),
                "detection_method": "OpenCV + Gemini + Dashboard Splitting + Schematic Detection",
                "concurrent_limit": self.MAX_CONCURRENT_GEMINI_CALLS,
            }
            stats.update(self.counters)
            return stats
            