#!/usr/bin/env python3
"""
Simple PDF Outline Extractor
A more reliable approach combining rule-based detection with machine learning
"""

import os
import json
import fitz  # PyMuPDF
import pdfplumber
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib
import re
from typing import List, Dict, Tuple, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleOutlineExtractor:
    def __init__(self, model_path: str = None):
        """Initialize the Simple Outline Extractor"""
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def extract_text_blocks(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text blocks from PDF with layout information"""
        text_blocks = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict")
                
                for block in blocks.get("blocks", []):
                    if "lines" in block:
                        for line in block["lines"]:
                            # Combine all spans in a line
                            line_text = ""
                            line_font_sizes = []
                            line_font_names = []
                            line_flags = []
                            line_bbox = None
                            
                            for span in line["spans"]:
                                line_text += span["text"]
                                line_font_sizes.append(span["size"])
                                line_font_names.append(span["font"])
                                line_flags.append(span.get("flags", 0))
                                
                                if line_bbox is None:
                                    line_bbox = span["bbox"]
                                else:
                                    x0, y0, x1, y1 = span["bbox"]
                                    lx0, ly0, lx1, ly1 = line_bbox
                                    line_bbox = [min(lx0, x0), min(ly0, y0), max(lx1, x1), max(ly1, y1)]
                            
                            if line_text.strip():
                                # Use dominant font size and name
                                font_size = max(line_font_sizes, key=line_font_sizes.count)
                                font_name = max(line_font_names, key=line_font_names.count)
                                
                                # Check formatting
                                is_bold = any("bold" in name.lower() or flags & 2**4 for name, flags in zip(line_font_names, line_flags))
                                is_italic = any("italic" in name.lower() or flags & 2**1 for name, flags in zip(line_font_names, line_flags))
                                
                                # Calculate position
                                x0, y0, x1, y1 = line_bbox
                                center_x = (x0 + x1) / 2
                                center_y = (y0 + y1) / 2
                                relative_y = center_y / page.rect.height
                                
                                text_blocks.append({
                                    'text': line_text.strip(),
                                    'page': page_num,
                                    'font_size': font_size,
                                    'font_name': font_name,
                                    'is_bold': is_bold,
                                    'is_italic': is_italic,
                                    'center_x': center_x,
                                    'center_y': center_y,
                                    'relative_y': relative_y,
                                    'text_length': len(line_text),
                                    'word_count': len(line_text.split()),
                                    'bbox': line_bbox
                                })
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return []
        
        return text_blocks
    
    def is_likely_heading(self, text: str, font_size: float, is_bold: bool, 
                         relative_y: float, text_length: int, word_count: int,
                         font_sizes: List[float]) -> Tuple[bool, str]:
        """Rule-based heading detection"""
        text_lower = text.lower().strip()
        
        # Skip if too long or too short
        if text_length < 2 or text_length > 100:
            return False, ""
        
        # Skip if too many words
        if word_count > 10:
            return False, ""
        
        # Skip common non-heading patterns
        skip_patterns = [
            'page', 'figure', 'table', 'doi:', 'http', 'www', 'arxiv:',
            'copyright', 'all rights reserved', 'submitted', 'published',
            'volume', 'issue', 'journal', 'conference', 'proceedings',
            'abstract', 'keywords', 'acknowledgments', 'references'
        ]
        
        if any(pattern in text_lower for pattern in skip_patterns):
            return False, ""
        
        # Skip if it's just numbers or special characters
        if re.match(r'^[\d\s\.\-_]+$', text_lower):
            return False, ""
        
        # Common heading words
        h1_keywords = [
            'introduction', 'conclusion', 'abstract', 'methodology', 'results',
            'discussion', 'references', 'acknowledgments', 'appendix',
            'related work', 'background', 'experiments', 'evaluation',
            'setup', 'implementation', 'analysis', 'summary', 'future work',
            'limitations', 'contributions', 'approach', 'system', 'model',
            'architecture', 'design', 'algorithm', 'framework', 'method'
        ]
        
        h2_keywords = [
            'dataset', 'evaluation', 'experimental setup', 'results',
            'case study', 'comparison', 'analysis', 'discussion'
        ]
        
        # Check font size relative to document
        if len(font_sizes) > 0:
            font_80th = np.percentile(font_sizes, 80)
            font_60th = np.percentile(font_sizes, 60)
        else:
            font_80th = 16
            font_60th = 14
        
        # Scoring system
        score = 0
        heading_level = "H2"
        
        # Font size scoring
        if font_size > font_80th:
            score += 3
            heading_level = "H1"
        elif font_size > font_60th:
            score += 2
            heading_level = "H2"
        
        # Bold formatting
        if is_bold:
            score += 2
        
        # Position (headings are usually at top of page)
        if relative_y < 0.3:
            score += 1
        
        # Common heading keywords
        if any(keyword in text_lower for keyword in h1_keywords):
            score += 4
            heading_level = "H1"
        elif any(keyword in text_lower for keyword in h2_keywords):
            score += 3
            heading_level = "H2"
        
        # Numbering patterns
        if (re.match(r'^\d+\.', text_lower) or 
            re.match(r'^[IVX]+\.', text_lower) or
            re.match(r'^[A-Z]\.', text_lower)):
            score += 2
            heading_level = "H2"
        
        # All caps (common in headings)
        if text.isupper() and text_length < 50:
            score += 1
        
        # Colon ending (common in headings)
        if text.endswith(':'):
            score += 1
        
        # Short text (headings are usually concise)
        if text_length < 30:
            score += 1
        
        # Determine if it's a heading (more selective)
        is_heading = score >= 6
        
        return is_heading, heading_level
    
    def extract_outline(self, pdf_path: str) -> Dict[str, Any]:
        """Extract outline from PDF"""
        # Extract text blocks
        text_blocks = self.extract_text_blocks(pdf_path)
        if not text_blocks:
            return {"title": "", "outline": []}
        
        # Calculate font size statistics
        font_sizes = [block['font_size'] for block in text_blocks]
        
        # Find title (usually the largest text on the first page)
        title = ""
        first_page_blocks = [b for b in text_blocks if b['page'] == 0]
        if first_page_blocks:
            # Sort by font size and position
            first_page_blocks.sort(key=lambda x: (-x['font_size'], x['relative_y']))
            
            # Take the largest font size text that's not too long and looks like a title
            for block in first_page_blocks[:10]:  # Check top 10 candidates
                if (block['text_length'] > 10 and 
                    block['text_length'] < 200 and
                    not any(skip in block['text'].lower() for skip in ['page', 'figure', 'table', 'doi', 'arxiv', 'copyright']) and
                    block['font_size'] > np.percentile([b['font_size'] for b in first_page_blocks], 80)):
                    title = block['text']
                    break
        
        # Find headings
        headings = []
        for block in text_blocks:
            is_heading, level = self.is_likely_heading(
                block['text'], block['font_size'], block['is_bold'],
                block['relative_y'], block['text_length'], block['word_count'],
                font_sizes
            )
            
            if is_heading:
                headings.append({
                    "level": level,
                    "text": block['text'],
                    "page": block['page']
                })
        
        # Sort by page and heading level
        headings.sort(key=lambda x: (x['page'], self._heading_level_to_number(x['level'])))
        
        # Remove duplicates and clean up
        seen = set()
        unique_headings = []
        for heading in headings:
            text_key = heading['text'].lower().strip()
            if text_key not in seen:
                seen.add(text_key)
                # Clean up the text (remove extra whitespace, etc.)
                heading['text'] = ' '.join(heading['text'].split())
                unique_headings.append(heading)
        
        return {
            "title": title,
            "outline": unique_headings
        }
    
    def _heading_level_to_number(self, level: str) -> int:
        """Convert heading level to number for sorting"""
        level_map = {'H1': 1, 'H2': 2, 'H3': 3}
        return level_map.get(level, 999)
    
    def save_model(self, model_path: str):
        """Save the model (placeholder for compatibility)"""
        pass
    
    def load_model(self, model_path: str):
        """Load the model (placeholder for compatibility)"""
        pass


def main():
    """Test the simple extractor"""
    extractor = SimpleOutlineExtractor()
    
    # Test on sample files
    test_files = ["sample_dataset/pdfs/1.pdf", "sample_dataset/pdfs/2.pdf"]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            logger.info(f"Testing {test_file}")
            result = extractor.extract_outline(test_file)
            print(f"Result for {test_file}:")
            print(f"Title: {result['title']}")
            print(f"Number of headings: {len(result['outline'])}")
            for heading in result['outline'][:5]:  # Show first 5 headings
                print(f"  {heading['level']}: {heading['text']} (page {heading['page']})")
            print("-" * 50)


if __name__ == "__main__":
    main() 