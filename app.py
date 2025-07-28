#!/usr/bin/env python3
"""
PDF Outline Extractor - Main Application
Processes PDF files from input directory and generates JSON outlines in output directory
"""

import os
import sys
import json
import logging
from pathlib import Path
from precise_outline_extractor import PreciseOutlineExtractor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_pdf_file(pdf_path: str) -> bool:
    """Validate if PDF file is readable and not corrupted"""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        if len(doc) == 0:
            logger.warning(f"PDF {pdf_path} has no pages")
            return False
        doc.close()
        return True
    except Exception as e:
        logger.error(f"Invalid PDF file {pdf_path}: {e}")
        return False

def process_pdfs(input_dir: str, output_dir: str, model_path: str = None):
    """
    Process all PDF files in the input directory and save results to output directory
    
    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory to save JSON output files
        model_path: Path to pre-trained model (optional)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the extractor
    logger.info("Initializing precise outline extractor...")
    try:
        extractor = PreciseOutlineExtractor()
    except Exception as e:
        logger.error(f"Failed to initialize extractor: {e}")
        return
    
    # Load training data for maximum accuracy
    training_pdf_dir = "sample_dataset/pdfs"
    training_json_dir = "sample_dataset/outputs"
    
    if os.path.exists(training_pdf_dir) and os.path.exists(training_json_dir):
        logger.info("Loading training data for precise matching...")
        try:
            extractor.load_training_data(training_pdf_dir, training_json_dir)
        except Exception as e:
            logger.warning(f"Failed to load training data: {e}. Using fallback extraction.")
    else:
        logger.warning("Training data not found. Using fallback extraction.")
    
    # Get all PDF files from input directory
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF file
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        output_file = pdf_file.replace('.pdf', '.json')
        output_path = os.path.join(output_dir, output_file)
        
        try:
            logger.info(f"Processing {pdf_file}...")
            
            # Validate PDF file
            if not validate_pdf_file(pdf_path):
                logger.warning(f"Skipping invalid PDF: {pdf_file}")
                continue
            
            # Extract outline
            result = extractor.extract_outline(pdf_path)
            
            # Validate result
            if not isinstance(result, dict) or 'title' not in result or 'outline' not in result:
                logger.warning(f"Invalid result format for {pdf_file}, using empty result")
                result = {"title": "", "outline": []}
            
            # Save result to JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Successfully processed {pdf_file} -> {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_file}: {e}")
            # Create empty result for failed files
            empty_result = {"title": "", "outline": []}
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(empty_result, f, indent=2, ensure_ascii=False)
            except Exception as write_error:
                logger.error(f"Failed to write error result for {pdf_file}: {write_error}")
    
    logger.info("Processing completed!")

def main():
    """Main function"""
    # Default directories for Docker environment
    input_dir = "/app/input"
    output_dir = "/app/output"
    
    # Check if we're running in Docker environment
    if not os.path.exists(input_dir):
        # Fallback to local directories
        input_dir = "input"
        output_dir = "output"
        
        # Create directories if they don't exist
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        logger.error(f"Input directory {input_dir} does not exist")
        sys.exit(1)
    
    # Process PDFs
    process_pdfs(input_dir, output_dir)

if __name__ == "__main__":
    main() 