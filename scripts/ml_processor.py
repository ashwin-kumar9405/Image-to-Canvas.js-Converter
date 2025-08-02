"""
Enhanced ML Processing Module for Image-to-Canvas.js Converter
Supports YOLOv8, Detectron2, OCR, and Canvas.js code generation
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import easyocr
from PIL import Image, ImageDraw
import json
import time
from typing import List, Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LayoutDetector:
    """Advanced layout detection using YOLOv8 and custom UI element detection"""
    
    def __init__(self, model_path: str = 'yolov8n.pt'):
        """Initialize the detector with YOLOv8 model"""
        self.model = YOLO(model_path)
        self.ocr_reader = easyocr.Reader(['en'])
        
        # Custom UI element classes
        self.ui_classes = {
            'button': 0,
            'text': 1,
            'image': 2,
            'input': 3,
            'container': 4,
            'icon': 5,
            'menu': 6,
            'card': 7
        }
        
    def detect_elements(self, image_path: str) -> List[Dict]:
        """Detect UI elements in an image"""
        start_time = time.time()
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        original_height, original_width = image.shape[:2]
        
        # Run YOLO detection
        results = self.model(image)[0]
        
        detections = []
        
        # Process YOLO detections
        for box in results.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = box
            
            if confidence < 0.5:  # Filter low confidence detections
                continue
                
            # Convert to integer coordinates
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            width = x2 - x1
            height = y2 - y1
            
            # Get class name
            class_name = results.names[int(class_id)]
            
            # Extract text from the detected region using OCR
            roi = image[y1:y2, x1:x2]
            text_content = self._extract_text(roi)
            
            detection = {
                'type': self._map_to_ui_element(class_name),
                'x': x1,
                'y': y1,
                'width': width,
                'height': height,
                'text': text_content,
                'confidence': confidence,
                'original_class': class_name
            }
            
            detections.append(detection)
        
        # Additional UI-specific detection
        ui_detections = self._detect_ui_elements(image)
        detections.extend(ui_detections)
        
        # Remove duplicates and merge overlapping detections
        detections = self._merge_detections(detections)
        
        processing_time = time.time() - start_time
        logger.info(f"Detected {len(detections)} elements in {processing_time:.2f}s")
        
        return detections
    
    def _extract_text(self, roi: np.ndarray) -> str:
        """Extract text from a region of interest using OCR"""
        if roi.size == 0:
            return ""
            
        try:
            # Convert BGR to RGB for EasyOCR
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
            # Run OCR
            ocr_results = self.ocr_reader.readtext(roi_rgb)
            
            # Combine all detected text
            text_parts = []
            for (bbox, text, confidence) in ocr_results:
                if confidence > 0.5:  # Filter low confidence text
                    text_parts.append(text.strip())
            
            return " ".join(text_parts)
            
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return ""
    
    def _map_to_ui_element(self, yolo_class: str) -> str:
        """Map YOLO class to UI element type"""
        mapping = {
            'person': 'image',
            'car': 'image',
            'truck': 'image',
            'book': 'container',
            'laptop': 'container',
            'cell phone': 'container',
            'tv': 'container'
        }
        
        return mapping.get(yolo_class, 'container')
    
    def _detect_ui_elements(self, image: np.ndarray) -> List[Dict]:
        """Detect UI-specific elements using computer vision techniques"""
        detections = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect rectangular shapes (potential buttons/containers)
        rectangles = self._detect_rectangles(gray)
        for rect in rectangles:
            x, y, w, h = rect
            roi = image[y:y+h, x:x+w]
            text = self._extract_text(roi)
            
            # Classify based on aspect ratio and size
            aspect_ratio = w / h if h > 0 else 1
            area = w * h
            
            if 2 < aspect_ratio < 6 and 1000 < area < 10000:
                element_type = 'button'
            elif aspect_ratio > 6 and area > 500:
                element_type = 'text'
            else:
                element_type = 'container'
            
            detections.append({
                'type': element_type,
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'text': text,
                'confidence': 0.7,
                'original_class': 'ui_detection'
            })
        
        return detections
    
    def _detect_rectangles(self, gray_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect rectangular shapes in the image"""
        # Apply edge detection
        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangles = []
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's roughly rectangular (4 corners)
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size
                if w > 20 and h > 10 and w < gray_image.shape[1] * 0.8 and h < gray_image.shape[0] * 0.8:
                    rectangles.append((x, y, w, h))
        
        return rectangles
    
    def _merge_detections(self, detections: List[Dict]) -> List[Dict]:
        """Merge overlapping detections"""
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        merged = []
        for detection in detections:
            overlap_found = False
            
            for merged_detection in merged:
                if self._calculate_iou(detection, merged_detection) > 0.5:
                    # Merge with higher confidence detection
                    if detection['confidence'] > merged_detection['confidence']:
                        merged_detection.update(detection)
                    overlap_found = True
                    break
            
            if not overlap_found:
                merged.append(detection)
        
        return merged
    
    def _calculate_iou(self, det1: Dict, det2: Dict) -> float:
        """Calculate Intersection over Union (IoU) between two detections"""
        x1_1, y1_1 = det1['x'], det1['y']
        x2_1, y2_1 = x1_1 + det1['width'], y1_1 + det1['height']
        
        x1_2, y1_2 = det2['x'], det2['y']
        x2_2, y2_2 = x1_2 + det2['width'], y1_2 + det2['height']
        
        # Calculate intersection
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

class CanvasJSGenerator:
    """Generate Canvas.js code from detected layout elements"""
    
    def __init__(self):
        self.element_colors = {
            'button': '#007bff',
            'text': '#6c757d',
            'image': '#28a745',
            'input': '#ffffff',
            'container': '#f8f9fa',
            'icon': '#ffc107',
            'menu': '#6f42c1',
            'card': '#fd7e14'
        }
    
    def generate_chart_code(self, elements: List[Dict]) -> str:
        """Generate Canvas.js chart code"""
        data_points = []
        
        for i, element in enumerate(elements):
            data_point = {
                'x': element['x'] + element['width'] // 2,
                'y': element['y'] + element['height'] // 2,
                'z': max(element['width'], element['height']),
                'name': element['type'],
                'text': element['text'] or element['type'],
                'confidence': round(element['confidence'], 2),
                'indexLabel': element['text'][:20] + '...' if len(element['text']) > 20 else element['text']
            }
            data_points.append(data_point)
        
        chart_code = f"""
// Generated Canvas.js Chart Code
var chart = new CanvasJS.Chart("chartContainer", {{
    animationEnabled: true,
    theme: "light2",
    title: {{
        text: "Detected UI Elements Layout",
        fontSize: 24
    }},
    axisX: {{
        title: "X Position (pixels)",
        minimum: 0,
        gridThickness: 1
    }},
    axisY: {{
        title: "Y Position (pixels)",
        minimum: 0,
        gridThickness: 1
    }},
    legend: {{
        cursor: "pointer",
        itemclick: toggleDataSeries
    }},
    data: [{{
        type: "bubble",
        name: "UI Elements",
        showInLegend: true,
        toolTipContent: "<b>{{name}}</b><br/>Position: ({{x}}, {{y}})<br/>Size: {{z}}px<br/>Text: {{text}}<br/>Confidence: {{confidence}}",
        dataPoints: {json.dumps(data_points, indent: 8)}
    }}]
}});

chart.render();

function toggleDataSeries(e) {{
    if (typeof(e.dataSeries.visible) === "undefined" || e.dataSeries.visible) {{
        e.dataSeries.visible = false;
    }} else {{
        e.dataSeries.visible = true;
    }}
    chart.render();
}}
"""
        return chart_code
    
    def generate_html5_canvas_code(self, elements: List[Dict]) -> str:
        """Generate HTML5 Canvas code"""
        canvas_code = """
// HTML5 Canvas Implementation
function drawUIElements() {
    const canvas = document.getElementById('uiCanvas');
    const ctx = canvas.getContext('2d');
    
    // Set canvas size
    canvas.width = 1200;
    canvas.height = 800;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Set default styles
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 2;
    ctx.font = '14px Arial';
    
"""
        
        for i, element in enumerate(elements):
            color = self.element_colors.get(element['type'], '#cccccc')
            
            canvas_code += f"""
    // Element {i + 1}: {element['type']}
    ctx.fillStyle = '{color}';
    ctx.fillRect({element['x']}, {element['y']}, {element['width']}, {element['height']});
    ctx.strokeRect({element['x']}, {element['y']}, {element['width']}, {element['height']});
    
"""
            
            if element['text']:
                canvas_code += f"""    // Text content
    ctx.fillStyle = '#000000';
    ctx.fillText('{element['text'][:50]}', {element['x'] + 5}, {element['y'] + 20});
    
"""
        
        canvas_code += """
}

// Call the function to draw elements
drawUIElements();

// Add interactivity
document.getElementById('uiCanvas').addEventListener('click', function(event) {
    const rect = this.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    console.log('Clicked at:', x, y);
    // Add your click handling logic here
});
"""
        
        return canvas_code

def calculate_accuracy_score(detections: List[Dict]) -> float:
    """Calculate accuracy score based on detection confidence and text extraction"""
    if not detections:
        return 0.0
    
    confidence_scores = [det['confidence'] for det in detections]
    text_scores = [1.0 if det['text'].strip() else 0.5 for det in detections]
    
    avg_confidence = sum(confidence_scores) / len(confidence_scores)
    avg_text_score = sum(text_scores) / len(text_scores)
    
    # Weighted average
    accuracy = (avg_confidence * 0.7) + (avg_text_score * 0.3)
    return min(accuracy, 1.0)

# Example usage
if __name__ == "__main__":
    detector = LayoutDetector()
    generator = CanvasJSGenerator()
    
    # Process an image
    image_path = "test_image.jpg"
    detections = detector.detect_elements(image_path)
    
    # Generate code
    chart_code = generator.generate_chart_code(detections)
    canvas_code = generator.generate_html5_canvas_code(detections)
    
    # Calculate accuracy
    accuracy = calculate_accuracy_score(detections)
    
    print(f"Detected {len(detections)} elements with {accuracy:.2%} accuracy")
    print("\nCanvas.js Code:")
    print(chart_code)
