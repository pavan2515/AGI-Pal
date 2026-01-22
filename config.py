"""
AgriPal RAG System - Configuration File
UPDATED with correct Gemini 2.5 model names
"""

import os
from pathlib import Path

# ============================================================================
# DIRECTORY STRUCTURE
# ============================================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "agricultural_docs"
CHUNKS_DIR = BASE_DIR / "chunks" / "processed"
VECTOR_DB_PATH = str(BASE_DIR / "chroma_agri_db")

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

# MySQL Configuration for Agricultural Data
MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'pavan',  # ⚠️ Change this to your MySQL password
    'database': 'agripal_rag',
    'port': 3306
}

# ============================================================================
# API KEYS
# ============================================================================

# Gemini API Key (FREE!) - Your current working key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyArxDxmlWNGvk2Gn5jmv6kcd7jYBw5mlks")

# ✅ UPDATED: Use correct Gemini 2.5 model names (from your API test)
# Primary model (fastest and most efficient)
GEMINI_MODEL = "models/gemini-2.5-flash"

# Alternative models (in order of preference)
GEMINI_FALLBACK_MODELS = [
    "models/gemini-2.5-flash",      # Primary - fastest
    "models/gemini-2.5-pro",        # More powerful
    "models/gemini-flash-latest",   # Latest stable flash
    "models/gemini-pro-latest",     # Latest stable pro
    "models/gemini-2.0-flash",      # Fallback to 2.0
]

# Claude API Key (Optional - costs money, not needed for this project)
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ============================================================================
# RAG SYSTEM CONFIGURATION
# ============================================================================

# Text Chunking Parameters
CHUNK_SIZE = 1000              # Characters per chunk
CHUNK_OVERLAP = 150            # Overlap between chunks
MIN_CHUNK_SIZE = 200           # Minimum chunk size to keep

# Embedding Model (for vector search)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ============================================================================
# RAG RETRIEVAL SETTINGS
# ============================================================================

TOP_K_DOCUMENTS = 5            # Number of documents to retrieve
SIMILARITY_THRESHOLD = 0.3     # Minimum similarity score
MAX_CONTEXT_TOKENS = 8000      # Maximum context length

# Gemini API Settings
GEMINI_TEMPERATURE = 0.7       # Creativity (0.0 = deterministic, 1.0 = creative)
GEMINI_TOP_P = 0.9            # Nucleus sampling
GEMINI_TOP_K = 40             # Top-k sampling
GEMINI_MAX_OUTPUT_TOKENS = 800 # Maximum response length

# ============================================================================
# AGRICULTURAL DOMAIN SETTINGS
# ============================================================================

SUPPORTED_CROPS = [
    'Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane',
    'Tomato', 'Potato', 'Onion', 'Chilli', 'Brinjal',
    'Mango', 'Banana', 'Coconut', 'Tea', 'Coffee',
    'Groundnut', 'Soybean', 'Sunflower', 'Mustard',
    'Chickpea', 'Lentil', 'Pigeon Pea', 'Green Gram'
]

INDIAN_STATES = [
    'Karnataka', 'Maharashtra', 'Tamil Nadu', 'Kerala', 'Gujarat',
    'Rajasthan', 'Punjab', 'Haryana', 'Uttar Pradesh', 'Bihar',
    'West Bengal', 'Madhya Pradesh', 'Andhra Pradesh', 'Telangana',
    'Odisha', 'Chhattisgarh', 'Jharkhand', 'Assam', 'Himachal Pradesh'
]

SEASONS = ['Kharif', 'Rabi', 'Zaid', 'Summer', 'Winter', 'Monsoon']

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Enable/disable verbose logging
VERBOSE_LOGGING = True

# ============================================================================
# CHATBOT SETTINGS
# ============================================================================

# Response settings
MAX_CONVERSATION_HISTORY = 10  # Number of previous messages to include
ENABLE_CHAT_HISTORY_DB = True  # Save chats to database
ENABLE_FALLBACK_RESPONSES = True  # Use fallback when API fails

# Intent detection keywords (for better query understanding)
INTENT_KEYWORDS = {
    'scheme': ['scheme', 'subsidy', 'yojana', 'benefit', 'government', 'pm-kisan', 'pmfby'],
    'market_price': ['price', 'mandi', 'market', 'rate', 'sell', 'buy', 'cost'],
    'disease': ['disease', 'pest', 'infection', 'treatment', 'cure', 'control', 'fungus'],
    'crop_guide': ['cultivation', 'farming', 'planting', 'sowing', 'harvest', 'grow'],
    'fertilizer': ['fertilizer', 'nutrient', 'soil', 'compost', 'npk', 'organic'],
    'weather': ['weather', 'rainfall', 'temperature', 'climate', 'forecast'],
}

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration settings"""
    errors = []
    warnings = []
    
    print("\n" + "="*70)
    print("🔍 CONFIGURATION VALIDATION")
    print("="*70)
    
    # Check directories
    print(f"\n📁 Directories:")
    print(f"   Base: {BASE_DIR}")
    print(f"   Data: {DATA_DIR} {'✅' if DATA_DIR.exists() else '⚠️ (will be created)'}")
    print(f"   Chunks: {CHUNKS_DIR} {'✅' if CHUNKS_DIR.exists() else '⚠️ (will be created)'}")
    print(f"   VectorDB: {VECTOR_DB_PATH}")
    
    # Check Gemini API key
    print(f"\n🔑 API Configuration:")
    if GEMINI_API_KEY == "your-key-here" or not GEMINI_API_KEY:
        errors.append("❌ Gemini API key not configured!")
        print(f"   Gemini Key: ❌ NOT CONFIGURED")
    else:
        print(f"   Gemini Key: ✅ {GEMINI_API_KEY[:20]}...{GEMINI_API_KEY[-4:]}")
        print(f"   Primary Model: {GEMINI_MODEL}")
    
    # Check MySQL configuration
    print(f"\n🗄️  MySQL Configuration:")
    print(f"   Host: {MYSQL_CONFIG['host']}")
    print(f"   User: {MYSQL_CONFIG['user']}")
    print(f"   Database: {MYSQL_CONFIG['database']}")
    print(f"   Port: {MYSQL_CONFIG['port']}")
    if MYSQL_CONFIG['password'] == 'pavan':
        warnings.append("⚠️  Using default MySQL password - consider changing it")
        print(f"   Password: ⚠️  DEFAULT (please change for production)")
    else:
        print(f"   Password: ✅ Custom")
    
    # Check RAG settings
    print(f"\n📊 RAG Settings:")
    print(f"   Chunk Size: {CHUNK_SIZE}")
    print(f"   Chunk Overlap: {CHUNK_OVERLAP}")
    print(f"   Embedding Model: {EMBEDDING_MODEL}")
    print(f"   Top-K Documents: {TOP_K_DOCUMENTS}")
    
    # Check agricultural data
    print(f"\n🌾 Agricultural Domain:")
    print(f"   Supported Crops: {len(SUPPORTED_CROPS)}")
    print(f"   Indian States: {len(INDIAN_STATES)}")
    print(f"   Seasons: {len(SEASONS)}")
    
    # Print warnings
    if warnings:
        print(f"\n⚠️  Warnings:")
        for warning in warnings:
            print(f"   - {warning}")
    
    # Print errors
    if errors:
        print(f"\n❌ Errors:")
        for error in errors:
            print(f"   - {error}")
        print("\n💡 Fix these errors before running the system!")
        print("="*70)
        return False
    
    print("\n✅ Configuration validated successfully!")
    print("="*70)
    return True


def print_config_summary():
    """Print a summary of current configuration"""
    print("\n" + "="*70)
    print("🌾 AgriPal RAG System - Configuration Summary")
    print("="*70)
    print(f"\n📂 Paths:")
    print(f"   Base Directory: {BASE_DIR}")
    print(f"   Data Directory: {DATA_DIR}")
    print(f"   Chunks Directory: {CHUNKS_DIR}")
    print(f"   Vector DB: {VECTOR_DB_PATH}")
    
    print(f"\n🤖 AI Configuration:")
    print(f"   Gemini Model: {GEMINI_MODEL}")
    print(f"   Temperature: {GEMINI_TEMPERATURE}")
    print(f"   Max Tokens: {GEMINI_MAX_OUTPUT_TOKENS}")
    
    print(f"\n📚 RAG Configuration:")
    print(f"   Embedding Model: {EMBEDDING_MODEL}")
    print(f"   Chunk Size: {CHUNK_SIZE}")
    print(f"   Chunk Overlap: {CHUNK_OVERLAP}")
    print(f"   Top-K Retrieval: {TOP_K_DOCUMENTS}")
    
    print(f"\n🗄️  Database:")
    print(f"   MySQL Host: {MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}")
    print(f"   Database Name: {MYSQL_CONFIG['database']}")
    
    print(f"\n🌾 Domain Knowledge:")
    print(f"   Crops: {len(SUPPORTED_CROPS)}")
    print(f"   States: {len(INDIAN_STATES)}")
    print(f"   Seasons: {len(SEASONS)}")
    
    print("="*70)


if __name__ == "__main__":
    print_config_summary()
    print()
    validate_config()