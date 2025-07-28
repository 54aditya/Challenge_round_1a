import os
import json
import fitz  # PyMuPDF
import pdfplumber
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import re
from typing import List, Dict, Tuple, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFOutlineExtractor:
    def __init__(self, model_path: str = None):
        """
        Initialize the PDF Outline Extractor
        
        Args:
            model_path: Path to pre-trained model (optional)
        """
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.title_model = None
        self.title_vectorizer = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def extract_text_features(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text and layout features from PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of text blocks with features
        """
        text_blocks = []
        
        try:
            # Use PyMuPDF for better text extraction
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get text blocks with font information
                blocks = page.get_text("dict")
                
                for block in blocks.get("blocks", []):
                    if "lines" in block:
                        for line in block["lines"]:
                            # Combine spans in a line to get complete text
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
                                    # Expand bbox to include all spans
                                    x0, y0, x1, y1 = span["bbox"]
                                    lx0, ly0, lx1, ly1 = line_bbox
                                    line_bbox = [min(lx0, x0), min(ly0, y0), max(lx1, x1), max(ly1, y1)]
                            
                            if line_text.strip():
                                # Use dominant font size and name
                                font_size = max(line_font_sizes, key=line_font_sizes.count)
                                font_name = max(line_font_names, key=line_font_names.count)
                                
                                # Check if any span is bold or italic
                                is_bold = any("bold" in name.lower() or flags & 2**4 for name, flags in zip(line_font_names, line_flags))
                                is_italic = any("italic" in name.lower() or flags & 2**1 for name, flags in zip(line_font_names, line_flags))
                                
                                # Calculate position features
                                x0, y0, x1, y1 = line_bbox
                                center_x = (x0 + x1) / 2
                                center_y = (y0 + y1) / 2
                                
                                # Calculate text length and word count
                                text_length = len(line_text)
                                word_count = len(line_text.split())
                                
                                # Check for common heading patterns
                                is_all_caps = line_text.isupper()
                                has_numbers = bool(re.search(r'\d', line_text))
                                ends_with_colon = line_text.endswith(':')
                                starts_with_number = bool(re.match(r'^\d+\.', line_text))
                                
                                # Calculate relative position on page
                                page_height = page.rect.height
                                relative_y = center_y / page_height
                                
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
                                    'text_length': text_length,
                                    'word_count': word_count,
                                    'is_all_caps': is_all_caps,
                                    'has_numbers': has_numbers,
                                    'ends_with_colon': ends_with_colon,
                                    'starts_with_number': starts_with_number,
                                    'bbox': line_bbox
                                })
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            # Fallback to pdfplumber
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        # Extract text blocks instead of individual words
                        text_blocks_page = page.extract_text_blocks()
                        for block in text_blocks_page:
                            text = block.get('text', '').strip()
                            if text:
                                text_blocks.append({
                                    'text': text,
                                    'page': page_num,
                                    'font_size': block.get('size', 12),
                                    'font_name': block.get('fontname', ''),
                                    'is_bold': False,
                                    'is_italic': False,
                                    'center_x': (block.get('x0', 0) + block.get('x1', 0)) / 2,
                                    'center_y': (block.get('y0', 0) + block.get('y1', 0)) / 2,
                                    'relative_y': (block.get('y0', 0) + block.get('y1', 0)) / 2 / page.height,
                                    'text_length': len(text),
                                    'word_count': len(text.split()),
                                    'is_all_caps': text.isupper(),
                                    'has_numbers': bool(re.search(r'\d', text)),
                                    'ends_with_colon': text.endswith(':'),
                                    'starts_with_number': bool(re.match(r'^\d+\.', text)),
                                    'bbox': [block.get('x0', 0), block.get('y0', 0), block.get('x1', 0), block.get('y1', 0)]
                                })
            except Exception as e2:
                logger.error(f"Fallback extraction also failed: {e2}")
                return []
        
        return text_blocks
    
    def create_features(self, text_blocks: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Create feature matrix from text blocks
        
        Args:
            text_blocks: List of text blocks with features
            
        Returns:
            DataFrame with features
        """
        if not text_blocks:
            return pd.DataFrame()
        
        df = pd.DataFrame(text_blocks)
        
        # Create additional features
        df['font_size_normalized'] = df['font_size'] / df['font_size'].max()
        df['text_length_normalized'] = df['text_length'] / df['text_length'].max()
        df['word_count_normalized'] = df['word_count'] / df['word_count'].max()
        
        # Create text-based features
        df['is_short'] = df['text_length'] < 50
        df['is_medium'] = (df['text_length'] >= 50) & (df['text_length'] < 200)
        df['is_long'] = df['text_length'] >= 200
        
        # Position-based features
        df['is_top_third'] = df['relative_y'] < 0.33
        df['is_middle_third'] = (df['relative_y'] >= 0.33) & (df['relative_y'] < 0.67)
        df['is_bottom_third'] = df['relative_y'] >= 0.67
        
        # Common heading patterns
        common_headings = [
            'introduction', 'conclusion', 'abstract', 'methodology', 'results',
            'discussion', 'references', 'acknowledgments', 'appendix',
            'related work', 'background', 'experiments', 'evaluation',
            'setup', 'implementation', 'analysis', 'summary'
        ]
        
        df['is_common_heading'] = df['text'].str.lower().isin(common_headings)
        
        # Numbering patterns
        df['has_roman_numerals'] = df['text'].str.match(r'^[IVX]+\.')
        df['has_letter_numbering'] = df['text'].str.match(r'^[A-Z]\.')
        
        return df
    
    def prepare_training_data(self, pdf_dir: str, json_dir: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare training data from PDF files and their corresponding JSON outputs
        
        Args:
            pdf_dir: Directory containing PDF files
            json_dir: Directory containing JSON output files
            
        Returns:
            Tuple of (features, heading_labels, title_labels)
        """
        all_features = []
        heading_labels = []
        title_labels = []
        
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            json_file = pdf_file.replace('.pdf', '.json')
            json_path = os.path.join(json_dir, json_file)
            
            if not os.path.exists(json_path):
                logger.warning(f"No JSON file found for {pdf_file}")
                continue
            
            try:
                # Load ground truth
                with open(json_path, 'r', encoding='utf-8') as f:
                    ground_truth = json.load(f)
                
                # Extract text features
                text_blocks = self.extract_text_features(pdf_path)
                if not text_blocks:
                    continue
                
                # Create features
                features_df = self.create_features(text_blocks)
                if features_df.empty:
                    continue
                
                # Create labels
                heading_label = []
                title_label = []
                
                # Get title from ground truth
                true_title = ground_truth.get('title', '').lower().strip()
                true_outline = ground_truth.get('outline', [])
                
                # Create mapping of text to heading level with fuzzy matching
                heading_map = {}
                for item in true_outline:
                    text_lower = item['text'].lower().strip()
                    heading_map[text_lower] = item['level']
                
                # Also create partial matches for better coverage
                for item in true_outline:
                    text_lower = item['text'].lower().strip()
                    # Add variations
                    heading_map[text_lower.replace(':', '').strip()] = item['level']
                    heading_map[text_lower.replace('.', '').strip()] = item['level']
                
                for _, row in features_df.iterrows():
                    text_lower = row['text'].lower().strip()
                    
                    # Check if this text is a heading with fuzzy matching
                    heading_found = False
                    for gt_text, level in heading_map.items():
                        # Exact match
                        if text_lower == gt_text:
                            heading_label.append(level)
                            heading_found = True
                            break
                        # Partial match (contains)
                        elif len(text_lower) > 5 and (gt_text in text_lower or text_lower in gt_text):
                            heading_label.append(level)
                            heading_found = True
                            break
                        # Similar match (high similarity)
                        elif self._text_similarity(text_lower, gt_text) > 0.8:
                            heading_label.append(level)
                            heading_found = True
                            break
                    
                    if not heading_found:
                        heading_label.append('NOT_HEADING')
                    
                    # Check if this text is the title with fuzzy matching
                    title_found = False
                    if true_title:
                        if text_lower == true_title:
                            title_label.append('TITLE')
                            title_found = True
                        elif self._text_similarity(text_lower, true_title) > 0.8:
                            title_label.append('TITLE')
                            title_found = True
                    
                    if not title_found:
                        title_label.append('NOT_TITLE')
                
                all_features.append(features_df)
                heading_labels.extend(heading_label)
                title_labels.extend(title_label)
                
                logger.info(f"Processed {pdf_file}: {len(text_blocks)} text blocks")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
                continue
        
        if not all_features:
            raise ValueError("No training data could be extracted")
        
        # Combine all features
        combined_features = pd.concat(all_features, ignore_index=True)
        
        return combined_features, pd.Series(heading_labels), pd.Series(title_labels)
    
    def train_model(self, pdf_dir: str, json_dir: str):
        """
        Train the machine learning model
        
        Args:
            pdf_dir: Directory containing PDF files
            json_dir: Directory containing JSON output files
        """
        logger.info("Preparing training data...")
        features, heading_labels, title_labels = self.prepare_training_data(pdf_dir, json_dir)
        
        logger.info(f"Training data shape: {features.shape}")
        logger.info(f"Heading labels distribution: {heading_labels.value_counts()}")
        logger.info(f"Title labels distribution: {title_labels.value_counts()}")
        
        # Prepare text features for vectorization
        text_features = features['text'].fillna('')
        
        # Train heading classifier
        logger.info("Training heading classifier...")
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2
        )
        
        text_vectors = self.vectorizer.fit_transform(text_features)
        
        # Combine text vectors with numerical features
        numerical_features = features[[
            'font_size_normalized', 'text_length_normalized', 'word_count_normalized',
            'is_bold', 'is_italic', 'is_all_caps', 'has_numbers', 'ends_with_colon',
            'starts_with_number', 'is_short', 'is_medium', 'is_long',
            'is_top_third', 'is_middle_third', 'is_bottom_third',
            'is_common_heading', 'has_roman_numerals', 'has_letter_numbering'
        ]].fillna(0)
        
        # Convert to dense array and combine with text vectors
        numerical_array = numerical_features.values
        text_array = text_vectors.toarray()
        combined_features = np.hstack([text_array, numerical_array])
        
        # Train heading classifier
        self.label_encoder = LabelEncoder()
        heading_labels_encoded = self.label_encoder.fit_transform(heading_labels)
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            combined_features, heading_labels_encoded, test_size=0.2, random_state=42, stratify=heading_labels_encoded
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate heading classifier
        y_pred = self.model.predict(X_val)
        logger.info("Heading Classification Report:")
        logger.info(classification_report(y_val, y_pred, target_names=self.label_encoder.classes_))
        
        # Train title classifier
        logger.info("Training title classifier...")
        title_labels_encoded = (title_labels == 'TITLE').astype(int)
        
        self.title_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            random_state=42,
            class_weight='balanced'
        )
        
        X_train_title, X_val_title, y_train_title, y_val_title = train_test_split(
            combined_features, title_labels_encoded, test_size=0.2, random_state=42, stratify=title_labels_encoded
        )
        
        self.title_model.fit(X_train_title, y_train_title)
        
        # Evaluate title classifier
        y_pred_title = self.title_model.predict(X_val_title)
        logger.info("Title Classification Report:")
        logger.info(classification_report(y_val_title, y_pred_title, target_names=['NOT_TITLE', 'TITLE']))
        
        logger.info("Model training completed!")
    
    def extract_outline(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract outline from a PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with title and outline
        """
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Extract text features
        text_blocks = self.extract_text_features(pdf_path)
        if not text_blocks:
            return {"title": "", "outline": []}
        
        # Create features
        features_df = self.create_features(text_blocks)
        if features_df.empty:
            return {"title": "", "outline": []}
        
        # Prepare features for prediction
        text_features = features_df['text'].fillna('')
        text_vectors = self.vectorizer.transform(text_features)
        
        numerical_features = features_df[[
            'font_size_normalized', 'text_length_normalized', 'word_count_normalized',
            'is_bold', 'is_italic', 'is_all_caps', 'has_numbers', 'ends_with_colon',
            'starts_with_number', 'is_short', 'is_medium', 'is_long',
            'is_top_third', 'is_middle_third', 'is_bottom_third',
            'is_common_heading', 'has_roman_numerals', 'has_letter_numbering'
        ]].fillna(0)
        
        numerical_array = numerical_features.values
        text_array = text_vectors.toarray()
        combined_features = np.hstack([text_array, numerical_array])
        
        # Predict headings and titles
        heading_predictions = self.model.predict(combined_features)
        heading_probs = self.model.predict_proba(combined_features)
        
        title_predictions = self.title_model.predict(combined_features)
        title_probs = self.title_model.predict_proba(combined_features)
        
        # Decode heading predictions
        heading_labels = self.label_encoder.inverse_transform(heading_predictions)
        
        # Extract title
        title_candidates = []
        for i, (text, is_title, title_prob) in enumerate(zip(features_df['text'], title_predictions, title_probs)):
            if is_title and title_prob[1] > 0.7:  # High confidence for title
                title_candidates.append((text, features_df.iloc[i]['page'], title_prob[1]))
        
        # Select the best title (highest confidence, earliest page)
        title = ""
        if title_candidates:
            title_candidates.sort(key=lambda x: (-x[2], x[1]))  # Sort by confidence desc, then page asc
            title = title_candidates[0][0]
        
        # Extract headings with better confidence thresholds
        outline = []
        for i, (text, heading_label, heading_prob) in enumerate(zip(features_df['text'], heading_labels, heading_probs)):
            if heading_label != 'NOT_HEADING':
                # Get confidence for the predicted class
                class_idx = list(self.label_encoder.classes_).index(heading_label)
                confidence = heading_prob[class_idx]
                
                # Use different thresholds for different heading levels
                threshold = 0.5  # Balanced threshold
                if confidence > threshold:
                    outline.append({
                        "level": heading_label,
                        "text": text,
                        "page": int(features_df.iloc[i]['page'])
                    })
        
        # Sort outline by page number and heading level
        outline.sort(key=lambda x: (x['page'], self._heading_level_to_number(x['level'])))
        
        # Post-process to improve heading detection using heuristics
        outline = self._post_process_outline(outline, features_df)
        
        return {
            "title": title,
            "outline": outline
        }
    
    def _heading_level_to_number(self, level: str) -> int:
        """Convert heading level to number for sorting"""
        level_map = {'H1': 1, 'H2': 2, 'H3': 3}
        return level_map.get(level, 999)
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using simple character-based approach"""
        if not text1 or not text2:
            return 0.0
        
        # Convert to sets of characters for simple similarity
        set1 = set(text1.lower())
        set2 = set(text2.lower())
        
        if not set1 or not set2:
            return 0.0
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _post_process_outline(self, outline: List[Dict], features_df: pd.DataFrame) -> List[Dict]:
        """Post-process outline to improve heading detection using heuristics"""
        if not outline:
            return outline
        
        # Common heading patterns
        common_headings = [
            'introduction', 'conclusion', 'abstract', 'methodology', 'results',
            'discussion', 'references', 'acknowledgments', 'appendix',
            'related work', 'background', 'experiments', 'evaluation',
            'setup', 'implementation', 'analysis', 'summary', 'future work',
            'limitations', 'contributions', 'approach', 'system', 'model',
            'architecture', 'design', 'algorithm', 'framework', 'method'
        ]
        
        # Add missing headings based on heuristics (more selective)
        added_headings = []
        
        # Calculate font size statistics
        font_sizes = features_df['font_size'].dropna()
        if len(font_sizes) > 0:
            font_80th_percentile = font_sizes.quantile(0.8)
            font_60th_percentile = font_sizes.quantile(0.6)
            font_median = font_sizes.median()
        else:
            font_80th_percentile = 16
            font_60th_percentile = 14
            font_median = 12
        
        for _, row in features_df.iterrows():
            text_lower = row['text'].lower().strip()
            
            # Skip if already in outline
            if any(item['text'].lower().strip() == text_lower for item in outline):
                continue
            
            # Skip very short or very long text
            if len(text_lower) < 3 or len(text_lower) > 80:  # More restrictive length limit
                continue
            
            # Check if this looks like a heading based on heuristics
            is_heading = False
            heading_level = "H2"  # Default level
            score = 0
            
            # Check font size (larger fonts are likely headings)
            if row['font_size'] > font_80th_percentile:
                score += 3
                heading_level = "H1"
            elif row['font_size'] > font_60th_percentile:
                score += 2
                heading_level = "H2"
            
            # Check if it's a common heading word
            if any(heading in text_lower for heading in common_headings):
                score += 4
                heading_level = "H1"
            
            # Check formatting
            if row['is_bold'] and row['font_size'] > font_median:
                score += 2
            
            # Check position (headings are usually at top of page)
            if row['relative_y'] < 0.3 and row['text_length'] < 80:
                score += 1
            
            # Check numbering patterns
            if (row['starts_with_number'] or 
                bool(re.match(r'^[IVX]+\.', text_lower)) or
                bool(re.match(r'^[A-Z]\.', text_lower))):
                score += 2
                heading_level = "H2"
            
            # Check for all caps (common in headings)
            if row['is_all_caps'] and len(text_lower) < 50:
                score += 1
            
            # Check for colon ending (common in headings)
            if row['ends_with_colon']:
                score += 1
            
            # Only add if score is high enough
            if score >= 4:  # Balanced score requirement
                added_headings.append({
                    "level": heading_level,
                    "text": row['text'],
                    "page": int(row['page'])
                })
        
        # Combine and sort
        all_headings = outline + added_headings
        all_headings.sort(key=lambda x: (x['page'], self._heading_level_to_number(x['level'])))
        
        # Remove duplicates
        seen = set()
        unique_headings = []
        for heading in all_headings:
            text_key = heading['text'].lower().strip()
            if text_key not in seen:
                seen.add(text_key)
                unique_headings.append(heading)
        
        # Limit the number of headings to prevent over-detection
        if len(unique_headings) > 30:  # Balanced limit
            # Keep the most important ones (H1s first, then by position)
            h1_headings = [h for h in unique_headings if h['level'] == 'H1']
            other_headings = [h for h in unique_headings if h['level'] != 'H1']
            
            # Keep all H1s and limit others
            max_others = 50 - len(h1_headings)
            if max_others > 0:
                other_headings = other_headings[:max_others]
            
            unique_headings = h1_headings + other_headings
            unique_headings.sort(key=lambda x: (x['page'], self._heading_level_to_number(x['level'])))
        
        # Final filtering to remove obvious false positives
        filtered_headings = []
        for heading in unique_headings:
            text_lower = heading['text'].lower().strip()
            
            # Skip if it's too long (likely not a heading)
            if len(text_lower) > 100:
                continue
                
            # Skip if it contains too many words (likely not a heading)
            if len(text_lower.split()) > 15:
                continue
                
            # Skip if it's just numbers or special characters
            if re.match(r'^[\d\s\.\-_]+$', text_lower):
                continue
                
            # Skip if it's a common non-heading pattern
            skip_patterns = [
                'page', 'figure', 'table', 'doi:', 'http', 'www', 'arxiv:',
                'copyright', 'all rights reserved', 'submitted', 'published',
                'volume', 'issue', 'journal', 'conference', 'proceedings'
            ]
            
            if any(pattern in text_lower for pattern in skip_patterns):
                continue
                
            filtered_headings.append(heading)
        
        return filtered_headings
    
    def save_model(self, model_path: str):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'title_model': self.title_model
        }
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.label_encoder = model_data['label_encoder']
        self.title_model = model_data['title_model']
        logger.info(f"Model loaded from {model_path}")


def main():
    """Main function for training and testing"""
    # Initialize extractor
    extractor = PDFOutlineExtractor()
    
    # Train model if training data is available
    pdf_dir = "sample_dataset/pdfs"
    json_dir = "sample_dataset/outputs"
    
    if os.path.exists(pdf_dir) and os.path.exists(json_dir):
        logger.info("Training model with provided dataset...")
        extractor.train_model(pdf_dir, json_dir)
        
        # Save the trained model
        extractor.save_model("trained_model.pkl")
        
        # Test on a few samples
        logger.info("Testing model on sample files...")
        test_files = ["sample_dataset/pdfs/1.pdf", "sample_dataset/pdfs/2.pdf"]
        
        for test_file in test_files:
            if os.path.exists(test_file):
                logger.info(f"Testing {test_file}")
                result = extractor.extract_outline(test_file)
                print(f"Result for {test_file}:")
                print(json.dumps(result, indent=2))
                print("-" * 50)
    else:
        logger.info("No training data found. Please provide PDF files and corresponding JSON outputs.")


if __name__ == "__main__":
    main() 