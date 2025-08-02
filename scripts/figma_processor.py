"""
Figma File Processing Module for extracting layout elements from Figma JSON exports
"""

import json
import math
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class FigmaProcessor:
    """Process Figma JSON files and extract layout elements"""
    
    def __init__(self):
        self.element_type_mapping = {
            'RECTANGLE': 'container',
            'TEXT': 'text',
            'FRAME': 'container',
            'GROUP': 'container',
            'VECTOR': 'icon',
            'ELLIPSE': 'icon',
            'LINE': 'separator',
            'COMPONENT': 'component',
            'INSTANCE': 'component'
        }
    
    def process_figma_file(self, figma_json: Dict[str, Any]) -> List[Dict]:
        """Process Figma JSON and extract layout elements"""
        try:
            elements = []
            
            # Process document structure
            if 'document' in figma_json:
                document = figma_json['document']
                elements.extend(self._process_node(document, 0, 0))
            
            # Filter and enhance elements
            elements = self._filter_elements(elements)
            elements = self._enhance_elements(elements)
            
            return elements
            
        except Exception as e:
            logger.error(f"Error processing Figma file: {e}")
            raise
    
    def _process_node(self, node: Dict[str, Any], parent_x: float = 0, parent_y: float = 0) -> List[Dict]:
        """Recursively process Figma nodes"""
        elements = []
        
        # Get node properties
        node_type = node.get('type', 'UNKNOWN')
        name = node.get('name', '')
        
        # Calculate absolute position
        x = parent_x + node.get('x', 0)
        y = parent_y + node.get('y', 0)
        width = node.get('width', 0)
        height = node.get('height', 0)
        
        # Skip very small elements
        if width < 5 or height < 5:
            if 'children' in node:
                for child in node['children']:
                    elements.extend(self._process_node(child, x, y))
            return elements
        
        # Extract text content
        text_content = self._extract_text_content(node)
        
        # Determine element type
        element_type = self._determine_element_type(node, text_content)
        
        # Calculate confidence based on node properties
        confidence = self._calculate_confidence(node, element_type)
        
        # Create element
        element = {
            'type': element_type,
            'x': int(x),
            'y': int(y),
            'width': int(width),
            'height': int(height),
            'text': text_content,
            'confidence': confidence,
            'source': 'figma',
            'figma_type': node_type,
            'name': name,
            'properties': self._extract_properties(node)
        }
        
        elements.append(element)
        
        # Process children
        if 'children' in node:
            for child in node['children']:
                elements.extend(self._process_node(child, x, y))
        
        return elements
    
    def _extract_text_content(self, node: Dict[str, Any]) -> str:
        """Extract text content from a Figma node"""
        text_content = ""
        
        # Direct text content
        if node.get('type') == 'TEXT':
            text_content = node.get('characters', '')
        
        # Text in name (for components/instances)
        elif node.get('name'):
            name = node.get('name', '')
            # Check if name looks like text content
            if not name.startswith(('Rectangle', 'Frame', 'Group', 'Vector', 'Ellipse')):
                text_content = name
        
        # Search in children for text
        if not text_content and 'children' in node:
            text_parts = []
            for child in node['children']:
                child_text = self._extract_text_content(child)
                if child_text:
                    text_parts.append(child_text)
            text_content = ' '.join(text_parts)
        
        return text_content.strip()
    
    def _determine_element_type(self, node: Dict[str, Any], text_content: str) -> str:
        """Determine the UI element type based on Figma node properties"""
        node_type = node.get('type', 'UNKNOWN')
        name = node.get('name', '').lower()
        width = node.get('width', 0)
        height = node.get('height', 0)
        
        # Check for button characteristics
        if self._is_button(node, text_content, name):
            return 'button'
        
        # Check for input field characteristics
        if self._is_input_field(node, name):
            return 'input'
        
        # Check for text elements
        if node_type == 'TEXT' or text_content:
            if height > 30 and len(text_content) < 50:
                return 'title'
            elif height > 20 and len(text_content) < 100:
                return 'heading'
            else:
                return 'text'
        
        # Check for image placeholders
        if self._is_image_placeholder(node, name):
            return 'image'
        
        # Check for icons
        if node_type in ['VECTOR', 'ELLIPSE'] or 'icon' in name:
            return 'icon'
        
        # Check for separators
        if node_type == 'LINE' or (width > height * 10 and height < 5):
            return 'separator'
        
        # Default mapping
        return self.element_type_mapping.get(node_type, 'container')
    
    def _is_button(self, node: Dict[str, Any], text_content: str, name: str) -> bool:
        """Check if node represents a button"""
        # Check name
        button_keywords = ['button', 'btn', 'cta', 'submit', 'click']
        if any(keyword in name for keyword in button_keywords):
            return True
        
        # Check text content
        if text_content:
            button_texts = ['click', 'submit', 'send', 'save', 'cancel', 'ok', 'yes', 'no', 'continue', 'next', 'back']
            if any(text.lower() in text_content.lower() for text in button_texts):
                return True
        
        # Check dimensions (typical button size)
        width = node.get('width', 0)
        height = node.get('height', 0)
        if 60 < width < 300 and 30 < height < 60:
            aspect_ratio = width / height if height > 0 else 1
            if 1.5 < aspect_ratio < 8:
                return True
        
        # Check for background fill (buttons usually have background)
        fills = node.get('fills', [])
        if fills and any(fill.get('visible', True) for fill in fills):
            return True
        
        return False
    
    def _is_input_field(self, node: Dict[str, Any], name: str) -> bool:
        """Check if node represents an input field"""
        input_keywords = ['input', 'field', 'textbox', 'textarea', 'search', 'email', 'password']
        if any(keyword in name for keyword in input_keywords):
            return True
        
        # Check dimensions (typical input field size)
        width = node.get('width', 0)
        height = node.get('height', 0)
        if width > 100 and 25 < height < 60:
            aspect_ratio = width / height if height > 0 else 1
            if aspect_ratio > 3:
                return True
        
        return False
    
    def _is_image_placeholder(self, node: Dict[str, Any], name: str) -> bool:
        """Check if node represents an image placeholder"""
        image_keywords = ['image', 'img', 'photo', 'picture', 'avatar', 'logo', 'placeholder']
        if any(keyword in name for keyword in image_keywords):
            return True
        
        # Check for image fills
        fills = node.get('fills', [])
        for fill in fills:
            if fill.get('type') == 'IMAGE':
                return True
        
        return False
    
    def _calculate_confidence(self, node: Dict[str, Any], element_type: str) -> float:
        """Calculate confidence score for element detection"""
        confidence = 0.8  # Base confidence
        
        # Increase confidence for explicit types
        if node.get('type') == 'TEXT' and element_type in ['text', 'title', 'heading']:
            confidence += 0.15
        
        # Increase confidence for named elements
        name = node.get('name', '').lower()
        if element_type in name:
            confidence += 0.1
        
        # Decrease confidence for generic containers
        if element_type == 'container' and node.get('type') in ['FRAME', 'GROUP']:
            confidence -= 0.1
        
        return min(confidence, 1.0)
    
    def _extract_properties(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant properties from Figma node"""
        properties = {}
        
        # Colors
        if 'fills' in node:
            fills = node['fills']
            if fills:
                fill = fills[0]  # Take first fill
                if fill.get('type') == 'SOLID':
                    color = fill.get('color', {})
                    properties['background_color'] = self._rgba_to_hex(color)
        
        # Border
        if 'strokes' in node and node['strokes']:
            stroke = node['strokes'][0]
            if stroke.get('type') == 'SOLID':
                color = stroke.get('color', {})
                properties['border_color'] = self._rgba_to_hex(color)
                properties['border_width'] = node.get('strokeWeight', 1)
        
        # Corner radius
        if 'cornerRadius' in node:
            properties['border_radius'] = node['cornerRadius']
        
        # Opacity
        if 'opacity' in node:
            properties['opacity'] = node['opacity']
        
        # Typography (for text elements)
        if node.get('type') == 'TEXT' and 'style' in node:
            style = node['style']
            properties.update({
                'font_family': style.get('fontFamily'),
                'font_size': style.get('fontSize'),
                'font_weight': style.get('fontWeight'),
                'text_align': style.get('textAlignHorizontal'),
                'line_height': style.get('lineHeightPx')
            })
        
        return properties
    
    def _rgba_to_hex(self, color: Dict[str, float]) -> str:
        """Convert RGBA color to hex"""
        r = int(color.get('r', 0) * 255)
        g = int(color.get('g', 0) * 255)
        b = int(color.get('b', 0) * 255)
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def _filter_elements(self, elements: List[Dict]) -> List[Dict]:
        """Filter out irrelevant elements"""
        filtered = []
        
        for element in elements:
            # Skip very small elements
            if element['width'] < 10 or element['height'] < 10:
                continue
            
            # Skip elements with very low confidence
            if element['confidence'] < 0.3:
                continue
            
            # Skip duplicate containers
            if element['type'] == 'container' and not element['text']:
                # Check if there's a similar container
                is_duplicate = False
                for existing in filtered:
                    if (existing['type'] == 'container' and 
                        abs(existing['x'] - element['x']) < 5 and
                        abs(existing['y'] - element['y']) < 5 and
                        abs(existing['width'] - element['width']) < 10 and
                        abs(existing['height'] - element['height']) < 10):
                        is_duplicate = True
                        break
                
                if is_duplicate:
                    continue
            
            filtered.append(element)
        
        return filtered
    
    def _enhance_elements(self, elements: List[Dict]) -> List[Dict]:
        """Enhance elements with additional information"""
        enhanced = []
        
        for element in elements:
            # Add semantic information based on position and context
            element['semantic_role'] = self._determine_semantic_role(element, elements)
            
            # Add layout information
            element['layout_info'] = self._analyze_layout_position(element, elements)
            
            # Clean up properties
            if 'properties' in element:
                element['properties'] = {k: v for k, v in element['properties'].items() if v is not None}
            
            enhanced.append(element)
        
        return enhanced
    
    def _determine_semantic_role(self, element: Dict, all_elements: List[Dict]) -> str:
        """Determine semantic role of element in the layout"""
        x, y = element['x'], element['y']
        width, height = element['width'], element['height']
        
        # Check if it's likely a header (top of layout)
        if y < 100 and element['type'] in ['text', 'title', 'heading']:
            return 'header'
        
        # Check if it's likely a footer (bottom of layout)
        max_y = max(el['y'] + el['height'] for el in all_elements)
        if y > max_y * 0.8 and element['type'] in ['text', 'container']:
            return 'footer'
        
        # Check if it's likely navigation (horizontal layout of buttons/links)
        if element['type'] == 'button':
            nearby_buttons = [el for el in all_elements 
                            if el['type'] == 'button' and 
                            abs(el['y'] - y) < 20 and 
                            abs(el['x'] - x) < 200]
            if len(nearby_buttons) > 2:
                return 'navigation'
        
        # Check if it's likely a sidebar (narrow vertical container)
        if element['type'] == 'container':
            aspect_ratio = height / width if width > 0 else 1
            if aspect_ratio > 2 and width < 300:
                return 'sidebar'
        
        return 'content'
    
    def _analyze_layout_position(self, element: Dict, all_elements: List[Dict]) -> Dict[str, Any]:
        """Analyze element's position in the overall layout"""
        layout_info = {}
        
        # Calculate relative position
        all_x = [el['x'] for el in all_elements]
        all_y = [el['y'] for el in all_elements]
        
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        if max_x > min_x:
            layout_info['relative_x'] = (element['x'] - min_x) / (max_x - min_x)
        else:
            layout_info['relative_x'] = 0.5
        
        if max_y > min_y:
            layout_info['relative_y'] = (element['y'] - min_y) / (max_y - min_y)
        else:
            layout_info['relative_y'] = 0.5
        
        # Determine grid position (rough estimate)
        layout_info['grid_column'] = int(layout_info['relative_x'] * 12) + 1  # 12-column grid
        layout_info['grid_row'] = int(layout_info['relative_y'] * 20) + 1     # 20-row grid
        
        return layout_info

# Example usage
if __name__ == "__main__":
    processor = FigmaProcessor()
    
    # Sample Figma JSON structure
    sample_figma = {
        "document": {
            "type": "DOCUMENT",
            "children": [
                {
                    "type": "CANVAS",
                    "children": [
                        {
                            "type": "FRAME",
                            "name": "Mobile App",
                            "x": 0,
                            "y": 0,
                            "width": 375,
                            "height": 812,
                            "children": [
                                {
                                    "type": "TEXT",
                                    "name": "Welcome Title",
                                    "x": 50,
                                    "y": 100,
                                    "width": 275,
                                    "height": 40,
                                    "characters": "Welcome to Our App"
                                },
                                {
                                    "type": "RECTANGLE",
                                    "name": "Login Button",
                                    "x": 50,
                                    "y": 700,
                                    "width": 275,
                                    "height": 50,
                                    "fills": [{"type": "SOLID", "color": {"r": 0, "g": 0.5, "b": 1}}]
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    }
    
    try:
        elements = processor.process_figma_file(sample_figma)
        print(f"Extracted {len(elements)} elements from Figma file")
        
        for element in elements:
            print(f"- {element['type']}: '{element['text']}' at ({element['x']}, {element['y']})")
            
    except Exception as e:
        print(f"Error: {e}")
