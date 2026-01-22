"""
AgriPal RAG System - Agricultural Document Processor
Extracts and chunks agricultural PDFs with domain-specific intelligence
"""

import fitz  # PyMuPDF
import re
import json
from pathlib import Path
from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
import config
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgriculturalDocProcessor:
    """Process agricultural PDFs with domain-specific chunking"""
    
    def __init__(self):
        self.chunk_size = config.CHUNK_SIZE
        self.chunk_overlap = config.CHUNK_OVERLAP
        self.crops = config.SUPPORTED_CROPS
        self.states = config.INDIAN_STATES
        
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text page by page from PDF"""
        try:
            doc = fitz.open(pdf_path)
            pages_data = []
            
            logger.info(f"📄 Extracting from: {Path(pdf_path).name}")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                if text.strip():  # Only add pages with content
                    pages_data.append({
                        'page_num': page_num + 1,
                        'text': text
                    })
            
            doc.close()
            logger.info(f"   ✅ Extracted {len(pages_data)} pages")
            return pages_data
            
        except Exception as e:
            logger.error(f"   ❌ Error extracting PDF: {e}")
            return []
    
    def detect_document_type(self, text: str) -> str:
        """Detect type of agricultural document"""
        text_lower = text.lower()
        
        # Pattern matching for document types
        patterns = {
            'scheme': ['scheme', 'yojana', 'subsidy', 'benefit', 'eligibility'],
            'pest_management': ['pest', 'disease', 'ipm', 'pesticide', 'fungicide'],
            'crop_guide': ['cultivation', 'planting', 'sowing', 'harvesting'],
            'fertilizer': ['fertilizer', 'nutrient', 'npk', 'soil health'],
            'weather': ['weather', 'rainfall', 'temperature', 'climate'],
            'market': ['price', 'mandi', 'msp', 'market'],
            'research': ['study', 'research', 'trial', 'experiment']
        }
        
        for doc_type, keywords in patterns.items():
            if sum(1 for kw in keywords if kw in text_lower) >= 2:
                return doc_type
        
        return 'general'
    
    def extract_metadata(self, text: str, filename: str) -> Dict:
        """Extract agricultural metadata from document"""
        metadata = {
            'source_file': filename,
            'document_type': self.detect_document_type(text),
            'crops_mentioned': [],
            'states_mentioned': [],
            'seasons_mentioned': [],
            'year': None
        }
        
        text_lower = text.lower()
        
        # Extract crops mentioned
        for crop in self.crops:
            if crop.lower() in text_lower:
                metadata['crops_mentioned'].append(crop)
        
        # Extract states mentioned
        for state in self.states:
            if state.lower() in text_lower:
                metadata['states_mentioned'].append(state)
        
        # Extract seasons
        for season in config.SEASONS:
            if season.lower() in text_lower:
                metadata['seasons_mentioned'].append(season)
        
        # Extract year
        year_match = re.search(r'20\d{2}', text)
        if year_match:
            metadata['year'] = year_match.group()
        
        return metadata
    
    def detect_structure(self, text: str) -> Dict:
        """Detect document structure for smart chunking"""
        structure = {
            'has_chapters': bool(re.search(r'Chapter\s+\d+', text, re.IGNORECASE)),
            'has_sections': bool(re.search(r'\d+\.\d+\s+[A-Z]', text)),
            'has_tables': bool(re.search(r'Table\s+\d+', text, re.IGNORECASE)),
            'has_figures': bool(re.search(r'Figure\s+\d+', text, re.IGNORECASE)),
            'has_bullet_points': bool(re.search(r'^\s*[•\-\*]', text, re.MULTILINE)),
            'has_numbered_lists': bool(re.search(r'^\s*\d+\.\s', text, re.MULTILINE))
        }
        return structure
    
    def smart_chunk_text(self, text: str, page_num: int, base_metadata: Dict) -> List[Dict]:
        """Intelligent chunking with agricultural context preservation"""
        
        # Custom separators for agricultural documents
        separators = [
            "\n\n## ",           # Markdown headers
            "\nChapter ",        # Chapters
            "\nSection ",        # Sections
            "\n\n",              # Paragraph breaks
            "\n",                # Line breaks
            ". ",                # Sentences
            " ",                 # Words
            ""
        ]
        
        splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        
        chunks = splitter.split_text(text)
        chunk_objects = []
        
        for i, chunk_text in enumerate(chunks):
            if len(chunk_text.strip()) < config.MIN_CHUNK_SIZE:
                continue
            
            structure = self.detect_structure(chunk_text)
            chunk_metadata = self.extract_metadata(chunk_text, base_metadata['source_file'])
            
            # Merge with base metadata
            chunk_metadata.update({
                'page_num': page_num,
                'chunk_id': i,
                'char_count': len(chunk_text),
                'has_tables': structure['has_tables'],
                'has_figures': structure['has_figures'],
                'document_type': base_metadata.get('document_type', 'general')
            })
            
            chunk_objects.append({
                'text': chunk_text.strip(),
                'metadata': chunk_metadata
            })
        
        return chunk_objects
    
    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """Process a single PDF completely"""
        logger.info(f"\n{'='*70}")
        logger.info(f"📄 Processing: {Path(pdf_path).name}")
        logger.info(f"{'='*70}")
        
        # Extract pages
        pages_data = self.extract_text_from_pdf(pdf_path)
        if not pages_data:
            logger.warning("   ⚠️  No pages extracted!")
            return []
        
        # Get base metadata from full document
        full_text = " ".join([p['text'] for p in pages_data[:5]])  # First 5 pages
        base_metadata = self.extract_metadata(full_text, Path(pdf_path).name)
        
        logger.info(f"   📋 Document Type: {base_metadata['document_type']}")
        logger.info(f"   🌾 Crops: {', '.join(base_metadata['crops_mentioned'][:3]) or 'None'}")
        logger.info(f"   📍 States: {', '.join(base_metadata['states_mentioned'][:3]) or 'None'}")
        
        # Chunk all pages
        all_chunks = []
        for page_data in pages_data:
            page_chunks = self.smart_chunk_text(
                page_data['text'],
                page_data['page_num'],
                base_metadata
            )
            all_chunks.extend(page_chunks)
        
        logger.info(f"   ✅ Created {len(all_chunks)} chunks")
        return all_chunks
    
    def save_chunks(self, chunks: List[Dict], output_name: str):
        """Save processed chunks to JSON"""
        output_path = config.CHUNKS_DIR / f"{output_name}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        logger.info(f"   💾 Saved to: {output_path.name}")
        
        # Save summary
        summary = {
            'total_chunks': len(chunks),
            'document_types': {},
            'crops': set(),
            'states': set()
        }
        
        for chunk in chunks:
            doc_type = chunk['metadata'].get('document_type', 'general')
            summary['document_types'][doc_type] = summary['document_types'].get(doc_type, 0) + 1
            summary['crops'].update(chunk['metadata'].get('crops_mentioned', []))
            summary['states'].update(chunk['metadata'].get('states_mentioned', []))
        
        summary['crops'] = list(summary['crops'])
        summary['states'] = list(summary['states'])
        
        summary_path = config.CHUNKS_DIR / f"{output_name}_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)


def main():
    """Main processing pipeline"""
    print("\n" + "="*70)
    print("🌾 AgriPal RAG - Agricultural Document Processing")
    print("="*70)
    
    # Validate configuration
    if not config.validate_config():
        return
    
    # Initialize processor
    processor = AgriculturalDocProcessor()
    
    # Find all PDFs
    pdf_files = list(config.DATA_DIR.glob("*.pdf"))
    
    if not pdf_files:
        logger.error(f"\n❌ No PDF files found in: {config.DATA_DIR}")
        logger.info(f"\n📥 Please download agricultural PDFs and place them in:")
        logger.info(f"   {config.DATA_DIR}")
        logger.info(f"\n💡 See the data collection guide below this script!")
        return
    
    logger.info(f"\n📚 Found {len(pdf_files)} PDF file(s)\n")
    
    # Process all PDFs
    all_chunks = []
    successful = 0
    failed = 0
    
    for pdf_path in pdf_files:
        try:
            chunks = processor.process_pdf(str(pdf_path))
            if chunks:
                all_chunks.extend(chunks)
                output_name = Path(pdf_path).stem
                processor.save_chunks(chunks, output_name)
                successful += 1
            else:
                failed += 1
                
        except Exception as e:
            logger.error(f"   ❌ Failed: {e}")
            failed += 1
    
    # Save combined chunks
    if all_chunks:
        processor.save_chunks(all_chunks, "all_agri_chunks_combined")
        
        print("\n" + "="*70)
        print("✅ PROCESSING COMPLETE!")
        print("="*70)
        print(f"   📊 Total Documents: {len(pdf_files)}")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        print(f"   📦 Total Chunks: {len(all_chunks)}")
        print(f"   📂 Output: {config.CHUNKS_DIR}")
        print("="*70)
    else:
        logger.error("\n❌ No chunks generated!")


if __name__ == "__main__":
    main()