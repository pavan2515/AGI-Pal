"""
AgriPal RAG System - Vector Database Setup
Creates ChromaDB with agricultural document embeddings
"""

import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pathlib import Path
import config
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgriVectorDBSetup:
    """Setup ChromaDB for agricultural documents"""
    
    def __init__(self):
        logger.info("🔧 Initializing Vector Database...")
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=config.VECTOR_DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Load embedding model
        logger.info(f"📦 Loading embedding model: {config.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        logger.info("   ✅ Model loaded\n")
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection("agri_docs")
            logger.info("📚 Using existing collection 'agri_docs'")
            
            # Optionally reset collection
            reset = input("\n⚠️  Collection exists. Reset? (yes/no): ").strip().lower()
            if reset == 'yes':
                self.client.delete_collection("agri_docs")
                self.collection = self.client.create_collection(
                    name="agri_docs",
                    metadata={"description": "Agricultural Knowledge Base"}
                )
                logger.info("♻️  Collection reset and recreated")
        except:
            self.collection = self.client.create_collection(
                name="agri_docs",
                metadata={"description": "Agricultural Knowledge Base"}
            )
            logger.info("📚 Created new collection 'agri_docs'")
    
    def load_chunks(self, json_path: str) -> list:
        """Load processed chunks from JSON"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            logger.info(f"📄 Loaded {len(chunks)} chunks from {Path(json_path).name}")
            return chunks
        except Exception as e:
            logger.error(f"❌ Error loading chunks: {e}")
            return []
    
    def prepare_metadata(self, chunk: dict) -> dict:
        """Clean and prepare metadata for ChromaDB"""
        metadata = chunk.get('metadata', {})
        
        # ChromaDB requirements:
        # - No None values
        # - Lists must be strings
        # - Keep it simple
        
        clean_metadata = {
            'source_file': str(metadata.get('source_file', 'unknown')),
            'page_num': int(metadata.get('page_num', 0)),
            'chunk_id': int(metadata.get('chunk_id', 0)),
            'document_type': str(metadata.get('document_type', 'general')),
            'char_count': int(metadata.get('char_count', 0)),
            'has_tables': bool(metadata.get('has_tables', False)),
            'has_figures': bool(metadata.get('has_figures', False))
        }
        
        # Convert lists to comma-separated strings
        if metadata.get('crops_mentioned'):
            clean_metadata['crops'] = ', '.join(metadata['crops_mentioned'][:5])
        
        if metadata.get('states_mentioned'):
            clean_metadata['states'] = ', '.join(metadata['states_mentioned'][:3])
        
        if metadata.get('seasons_mentioned'):
            clean_metadata['seasons'] = ', '.join(metadata['seasons_mentioned'])
        
        if metadata.get('year'):
            clean_metadata['year'] = str(metadata['year'])
        
        return clean_metadata
    
    def add_chunks_to_db(self, chunks: list, batch_size: int = 100):
        """Add chunks to vector database in batches"""
        if not chunks:
            logger.error("❌ No chunks to add!")
            return
        
        total_chunks = len(chunks)
        total_batches = (total_chunks + batch_size - 1) // batch_size
        
        logger.info(f"\n📦 Adding {total_chunks} chunks in {total_batches} batches...")
        
        successful = 0
        failed = 0
        
        for batch_num in range(0, total_chunks, batch_size):
            batch_end = min(batch_num + batch_size, total_chunks)
            batch_chunks = chunks[batch_num:batch_end]
            
            current_batch = (batch_num // batch_size) + 1
            
            try:
                documents = []
                metadatas = []
                ids = []
                
                for idx, chunk in enumerate(batch_chunks):
                    actual_idx = batch_num + idx
                    
                    # Create unique ID
                    source = chunk['metadata'].get('source_file', 'unknown')
                    page = chunk['metadata'].get('page_num', 0)
                    chunk_id = chunk['metadata'].get('chunk_id', actual_idx)
                    unique_id = f"{source}_{page}_{chunk_id}"
                    
                    documents.append(chunk['text'])
                    metadatas.append(self.prepare_metadata(chunk))
                    ids.append(unique_id)
                
                # Generate embeddings
                logger.info(f"   🔢 Batch {current_batch}/{total_batches}: Generating embeddings...")
                embeddings = self.embedding_model.encode(
                    documents, 
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                
                # Add to database
                logger.info(f"   💾 Adding to database...")
                self.collection.add(
                    embeddings=embeddings.tolist(),
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                
                successful += len(batch_chunks)
                logger.info(f"   ✅ Batch {current_batch} complete ({len(batch_chunks)} chunks)")
                
            except Exception as e:
                logger.error(f"   ❌ Batch {current_batch} failed: {e}")
                failed += len(batch_chunks)
        
        logger.info(f"\n📊 Results:")
        logger.info(f"   ✅ Successful: {successful}")
        logger.info(f"   ❌ Failed: {failed}")
    
    def get_collection_stats(self):
        """Get detailed statistics about the collection"""
        try:
            count = self.collection.count()
            
            # Sample some documents to analyze
            sample = self.collection.get(limit=min(100, count))
            
            stats = {
                'total_chunks': count,
                'document_types': {},
                'sources': set(),
                'crops': set(),
                'states': set()
            }
            
            for metadata in sample['metadatas']:
                # Document types
                doc_type = metadata.get('document_type', 'unknown')
                stats['document_types'][doc_type] = stats['document_types'].get(doc_type, 0) + 1
                
                # Sources
                if 'source_file' in metadata:
                    stats['sources'].add(metadata['source_file'])
                
                # Crops
                if 'crops' in metadata:
                    stats['crops'].update(metadata['crops'].split(', '))
                
                # States
                if 'states' in metadata:
                    stats['states'].update(metadata['states'].split(', '))
            
            # Print statistics
            print("\n" + "="*70)
            print("📊 VECTOR DATABASE STATISTICS")
            print("="*70)
            print(f"Total Chunks: {stats['total_chunks']}")
            print(f"Unique Sources: {len(stats['sources'])}")
            print(f"Unique Crops: {len(stats['crops'])}")
            print(f"Unique States: {len(stats['states'])}")
            
            print("\n📚 Document Types:")
            for doc_type, count in sorted(stats['document_types'].items(), key=lambda x: x[1], reverse=True):
                print(f"   - {doc_type}: {count}")
            
            if stats['crops']:
                print(f"\n🌾 Top Crops: {', '.join(list(stats['crops'])[:10])}")
            
            if stats['states']:
                print(f"📍 States: {', '.join(list(stats['states'])[:10])}")
            
            print("="*70)
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting stats: {e}")
            return None
    
    def test_search(self, query: str, n_results: int = 3):
        """Test the search functionality"""
        try:
            logger.info(f"\n🔍 Testing search: '{query}'")
            
            # Encode query
            query_embedding = self.embedding_model.encode(query)
            
            # Search
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results
            )
            
            print("\n" + "="*70)
            print(f"🔍 Search Results for: '{query}'")
            print("="*70)
            
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ), 1):
                print(f"\n📄 Result {i} (Distance: {distance:.4f})")
                print(f"   Source: {metadata.get('source_file', 'Unknown')}")
                print(f"   Type: {metadata.get('document_type', 'Unknown')}")
                print(f"   Page: {metadata.get('page_num', 'Unknown')}")
                
                if metadata.get('crops'):
                    print(f"   Crops: {metadata['crops']}")
                
                preview = doc[:200] + "..." if len(doc) > 200 else doc
                print(f"\n   Preview:\n   {preview}\n")
                print("   " + "-"*66)
            
            print("="*70)
            
        except Exception as e:
            logger.error(f"❌ Search test failed: {e}")


def main():
    """Main setup pipeline"""
    print("\n" + "="*70)
    print("🌾 AgriPal RAG - Vector Database Setup")
    print("="*70)
    
    # Validate configuration
    if not config.validate_config():
        return
    
    # Initialize database
    db_setup = AgriVectorDBSetup()
    
    # Find chunk files
    chunk_files = list(config.CHUNKS_DIR.glob("*.json"))
    chunk_files = [f for f in chunk_files if not f.name.endswith('_summary.json')]
    
    if not chunk_files:
        logger.error(f"\n❌ No chunk files found in: {config.CHUNKS_DIR}")
        logger.info("\n💡 Please run 1_process_agri_docs.py first!")
        return
    
    print(f"\n📚 Found {len(chunk_files)} chunk file(s):")
    for f in chunk_files:
        print(f"   - {f.name}")
    
    # Load combined file or first available
    combined_file = config.CHUNKS_DIR / "all_agri_chunks_combined.json"
    target_file = combined_file if combined_file.exists() else chunk_files[0]
    
    print(f"\n📂 Loading: {target_file.name}")
    chunks = db_setup.load_chunks(str(target_file))
    
    if not chunks:
        logger.error("❌ No chunks loaded!")
        return
    
    # Add to database
    db_setup.add_chunks_to_db(chunks, batch_size=100)
    
    # Show statistics
    db_setup.get_collection_stats()
    
    # Test searches
    test_queries = [
        "tomato disease management",
        "government schemes for farmers",
        "rice cultivation kharif season"
    ]
    
    print("\n" + "="*70)
    print("🧪 TESTING SEARCH FUNCTIONALITY")
    print("="*70)
    
    for query in test_queries:
        db_setup.test_search(query, n_results=2)
        print("\n")
    
    print("="*70)
    print("✅ VECTOR DATABASE SETUP COMPLETE!")
    print("="*70)
    print(f"📍 Database Location: {config.VECTOR_DB_PATH}")
    print(f"📊 Total Documents: {db_setup.collection.count()}")
    print("="*70)


if __name__ == "__main__":
    main()