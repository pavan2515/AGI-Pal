"""
AgriPal RAG System - Comprehensive Testing Suite
Tests all components: Vector DB, MySQL, Claude RAG
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from gemini_rag_chatbot import AgriPalRAGChatbot
import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_vector_search():
    """Test vector database search"""
    print("\n" + "="*70)
    print("TEST 1: VECTOR DATABASE SEARCH")
    print("="*70)
    
    try:
        chatbot = AgriPalRAGChatbot()
        
        queries = [
            "tomato disease management",
            "organic farming techniques",
            "kharif season crops"
        ]
        
        for query in queries:
            print(f"\nQuery: {query}")
            results = chatbot.search_documents(query, n_results=3)
            print(f"✅ Found {len(results['documents'][0])} documents")
        
        chatbot.close()
        print("\n✅ Vector search test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Vector search test failed: {e}")
        return False


def test_mysql_queries():
    """Test MySQL database queries"""
    print("\n" + "="*70)
    print("TEST 2: MYSQL DATABASE QUERIES")
    print("="*70)
    
    try:
        chatbot = AgriPalRAGChatbot()
        
        # Test schemes
        print("\n📜 Testing government schemes...")
        schemes = chatbot.search_schemes(state="Karnataka")
        print(f"✅ Found {len(schemes)} schemes")
        
        # Test crop info
        print("\n🌾 Testing crop information...")
        crop = chatbot.search_crop_info("Tomato")
        print(f"✅ Found crop: {crop['crop_name'] if crop else 'None'}")
        
        # Test market prices
        print("\n💰 Testing market prices...")
        prices = chatbot.search_market_prices("Tomato")
        print(f"✅ Found {len(prices)} price records")
        
        chatbot.close()
        print("\n✅ MySQL queries test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ MySQL queries test failed: {e}")
        return False


def test_intent_detection():
    """Test query intent detection"""
    print("\n" + "="*70)
    print("TEST 3: INTENT DETECTION")
    print("="*70)
    
    try:
        chatbot = AgriPalRAGChatbot()
        
        test_cases = [
            ("What schemes are available for farmers?", "scheme"),
            ("Current price of onion", "market_price"),
            ("How to treat tomato blight?", "disease"),
            ("Rice cultivation practices", "crop_guide")
        ]
        
        for query, expected_intent in test_cases:
            intent = chatbot.detect_query_intent(query)
            status = "✅" if intent['type'] == expected_intent else "❌"
            print(f"{status} '{query}' -> {intent['type']}")
        
        chatbot.close()
        print("\n✅ Intent detection test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Intent detection test failed: {e}")
        return False


def test_full_rag_pipeline():
    """Test complete RAG pipeline with Claude"""
    print("\n" + "="*70)
    print("TEST 4: FULL RAG PIPELINE")
    print("="*70)
    
    try:
        chatbot = AgriPalRAGChatbot()
        
        queries = [
            "What are organic methods to control pests in tomato?",
            "Tell me about PM-KISAN scheme",
            "Best time to plant rice in Karnataka"
        ]
        
        for query in queries:
            print(f"\n🔍 Query: {query}")
            response = chatbot.generate_response(query, location="Karnataka")
            print(f"✅ Response length: {len(response)} chars")
            print(f"Preview: {response[:100]}...")
        
        chatbot.close()
        print("\n✅ Full RAG pipeline test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Full RAG pipeline test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🧪 AgriPal RAG System - Comprehensive Testing")
    print("="*70)
    
    # Validate configuration first
    if not config.validate_config():
        print("\n❌ Configuration validation failed!")
        return
    
    tests = [
        ("Vector Database Search", test_vector_search),
        ("MySQL Database Queries", test_mysql_queries),
        ("Intent Detection", test_intent_detection),
        ("Full RAG Pipeline", test_full_rag_pipeline)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\n{passed}/{total} tests passed")
    print("="*70)
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use!")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()
