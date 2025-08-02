"""
PDF Processing Module for extracting layout elements from PDF files
"""

import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import io
import json
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Process PDF files and extract layout elements"""
    
    def __init__(self):
        self.dpi = 150  # DPI for PDF to image conversion
    
    def process_pdf(self, pdf_path: str, page_num: int = 0) -> Tuple[List[Dict], np.ndarray]:
        """Process a PDF page and extract layout elements"""
        try:
            # Open PDF
            doc = fitz.open(pdf_path)
            
            if page_num >= len(doc):
                raise ValueError(f"Page {page_num} not found in PDF")
            
            page = doc[page_num]
            
            # Convert page to image
            mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to OpenCV format
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Extract text blocks from PDF
            text_blocks = self._extract_text_blocks(page, mat)
            
            # Extract images from PDF
            image_blocks = self._extract_image_blocks(page, mat)
            
            # Detect additional elements using computer vision
            cv_elements = self._detect_cv_elements(image)
            
            # Combine all detections
            all_elements = text_blocks + image_blocks + cv_elements
            
            # Remove duplicates and merge overlapping elements
            merged_elements = self._merge_elements(all_elements)
            
            doc.close()
            
            return merged_elements, image
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise
    
    def _extract_text_blocks(self, page, matrix) -> List[Dict]:
        """Extract text blocks from PDF page"""
        text_blocks = []
        blocks = page.get_text("dict")
        
        for block in blocks["blocks"]:
            if "lines" in block:  # Text block
                bbox = block["bbox"]
                
                # Transform coordinates using matrix
                x0, y0, x1, y1 = bbox
                p0 = fitz.Point(x0, y0) * matrix
                p1 = fitz.Point(x1, y1) * matrix
                
                # Extract text content
                text_content = ""
                for line in block["lines"]:
                    for span in line["spans"]:
                        text_content += span["text"] + " "
                
                text_content = text_content.strip()
                
                if text_content:  # Only add if there's actual text
                    element = {
                        'type': self._classify_text_element(text_content, bbox),
                        'x': int(p0.x),
                        'y': int(p0.y),
                        'width': int(p1.x - p0.x),
                        'height': int(p1.y - p0.y),
                        'text': text_content,
                        'confidence': 0.95,
                        'source': 'pdf_text'
                    }
                    text_blocks.append(element)
        
        return text_blocks
    
    def _extract_image_blocks(self, page, matrix) -> List[Dict]:
        """Extract image blocks from PDF page"""
        image_blocks = []
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            try:
                # Get image bbox
                img_rect = page.get_image_bbox(img)
                if img_rect:
                    x0, y0, x1, y1 = img_rect
                    p0 = fitz.Point(x0, y0) * matrix
                    p1 = fitz.Point(x1, y1) * matrix
                    
                    element = {
                        'type': 'image',
                        'x': int(p0.x),
                        'y': int(p0.y),
                        'width': int(p1.x - p0.x),
                        'height': int(p1.y - p0.y),
                        'text': f'Image {img_index + 1}',
                        'confidence': 0.90,
                        'source': 'pdf_image'
                    }
                    image_blocks.append(element)
            except Exception as e:
                logger.warning(f"Error extracting image {img_index}: {e}")
        
        return image_blocks
    
    def _detect_cv_elements(self, image: np.ndarray) -> List[Dict]:
        """Detect additional elements using computer vision"""
        elements = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect lines (potential separators)
        lines = self._detect_lines(gray)
        for line in lines:
            x1, y1, x2, y2 = line
            elements.append({
                'type': 'separator',
                'x': min(x1, x2),
                'y': min(y1, y2),
                'width': abs(x2 - x1),
                'height': abs(y2 - y1),
                'text': '',
                'confidence': 0.75,
                'source': 'cv_detection'
            })
        
        # Detect potential form fields
        form_fields = self._detect_form_fields(gray)
        elements.extend(form_fields)
        
        return elements
    
    def _detect_lines(self, gray_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect lines in the image"""
        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        
        if lines is not None:
            return [tuple(line[0]) for line in lines]
        return []
    
    def _detect_form_fields(self, gray_image: np.ndarray) -> List[Dict]:
        """Detect potential form fields"""
        form_fields = []
        
        # Use morphological operations to detect rectangular regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size and aspect ratio (typical form field characteristics)
            if 50 < w < 300 and 15 < h < 50:
                aspect_ratio = w / h
                if 2 < aspect_ratio < 15:  # Typical input field ratio
                    form_fields.append({
                        'type': 'input',
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'text': '',
                        'confidence': 0.70,
                        'source': 'cv_form_detection'
                    })
        
        return form_fields
    
    def _classify_text_element(self, text: str, bbox: Tuple[float, float, float, float]) -> str:
        """Classify text element based on content and size"""
        x0, y0, x1, y1 = bbox
        width = x1 - x0
        height = y1 - y0
        
        # Check if it's a title (large text, short content)
        if height > 20 and len(text) < 50:
            return 'title'
        
        # Check if it's a heading
        if height > 15 and len(text) < 100:
            return 'heading'
        
        # Check if it's a label (short text)
        if len(text) < 30:
            return 'label'
        
        # Default to paragraph text
        return 'text'
    
    def _merge_elements(self, elements: List[Dict]) -> List[Dict]:
        """Merge overlapping elements"""
        if not elements:
            return []
        
        # Sort by area (larger elements first)
        elements.sort(key=lambda x: x['width'] * x['height'], reverse=True)
        
        merged = []
        for element in elements:
            overlap_found = False
            
            for merged_element in merged:
                if self._calculate_overlap(element, merged_element) > 0.7:
                    # Merge elements - keep the one with higher confidence
                    if element['confidence'] > merged_element['confidence']:
                        merged_element.update(element)
                    overlap_found = True
                    break
            
            if not overlap_found:
                merged.append(element)
        
        return merged
    
    def _calculate_overlap(self, elem1: Dict, elem2: Dict) -> float:
        """Calculate overlap ratio between two elements"""
        x1_1, y1_1 = elem1['x'], elem1['y']
        x2_1, y2_1 = x1_1 + elem1['width'], y1_1 + elem1['height']
        
        x1_2, y1_2 = elem2['x'], elem2['y']
        x2_2, y2_2 = x1_2 + elem2['width'], y1_2 + elem2['height']
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = elem1['width'] * elem1['height']
        area2 = elem2['width'] * elem2['height']
        
        # Return overlap as ratio of smaller element
        smaller_area = min(area1, area2)
        return intersection / smaller_area if smaller_area > 0 else 0.0

# Example usage
if __name__ == "__main__":
    processor = PDFProcessor()
    
    try:
        elements, image = processor.process_pdf("sample.pdf", 0)
        print(f"Extracted {len(elements)} elements from PDF")
        
        for i, element in enumerate(elements):
            print(f"Element {i+1}: {element['type']} - '{element['text'][:50]}...'")
            
    except Exception as e:
        print(f"Error: {e}")
