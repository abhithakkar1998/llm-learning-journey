"""
Test the FastAPI server with a simple client request.
"""

import requests
import json
import time
import sys
import os

def detect_initialization_type():
    """
    Detect whether this is a first-time initialization or normal startup.
    
    Returns:
        str: "first_time", "normal", or "unknown"
    """
    # Go up one level from test_scripts to find data directory
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(parent_dir, "data")
    
    # Check if data directory and key files exist
    embeddings_path = os.path.join(data_dir, "wikipedia_embeddings.npy")
    corpus_path = os.path.join(data_dir, "wikipedia_corpus.json")
    
    if not os.path.exists(data_dir):
        return "first_time"
    
    if not os.path.exists(embeddings_path) or not os.path.exists(corpus_path):
        return "first_time"
    
    # Check if files are recent and substantial (not empty/corrupted)
    try:
        if os.path.getsize(embeddings_path) < 1000 or os.path.getsize(corpus_path) < 1000:
            return "first_time"
    except OSError:
        return "first_time"
    
    return "normal"

def print_smart_testing_info():
    """Print information about the smart testing logic."""
    print("\n" + "="*60)
    print("🧠 SMART TESTING LOGIC")
    print("="*60)
    print("This test automatically detects initialization type:")
    print()
    print("🔍 FIRST-TIME INITIALIZATION:")
    print("   • No timeout (waits as long as needed)")
    print("   • Expects 5-10 minutes for full setup")
    print("   • Shows detailed progress messages")
    print()
    print("⚡ NORMAL STARTUP (Cached Data):")
    print("   • 60-second timeout")
    print("   • Should be ready in 30-60 seconds")
    print("   • Quick failure detection")
    print()
    print("Usage:")
    print("   python test_api_client.py        # Full test")
    print("   python test_api_client.py --quick # Skip search tests")
    print("="*60)

def test_api_server(quick_test=False):
    """Test the RAG API endpoints."""
    base_url = "http://localhost:8000"
    
    # Detect initialization type
    init_type = detect_initialization_type()
    
    print("🧪 Testing RAG API Server")
    print("=" * 50)
    
    if init_type == "first_time":
        print("� First-time initialization detected")
        print("📝 This will take 5-10 minutes:")
        print("    • Downloading sentence transformer model (1-3 min)")
        print("    • Fetching Wikipedia articles (1-2 min)")
        print("    • Generating embeddings (2-5 min)")
        print("    • Subsequent runs will be much faster!")
    else:
        print("⚡ Normal startup detected (cached data available)")
        print("📝 Should be ready within 30-60 seconds")
    
    print()
    
    # Test 1: Root endpoint
    print("\n1. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        print(f"✓ Root endpoint: {response.status_code}")
        print(f"  Response: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("❌ Server not running. Start it with: python src/main.py")
        return
    
    # Test 2: Health check
    print("\n2. Testing health endpoint...")
    response = requests.get(f"{base_url}/health")
    health_data = response.json()
    print(f"✓ Health check: {response.status_code}")
    print(f"  Status: {health_data.get('status')}")
    print(f"  Message: {health_data.get('message')}")
    
    # Test 3: Status endpoint
    print("\n3. Testing status endpoint...")
    response = requests.get(f"{base_url}/status")
    status_data = response.json()
    print(f"✓ Status endpoint: {response.status_code}")
    print(f"  Pipeline status: {status_data.get('status')}")
    if 'pipeline' in status_data and status_data['pipeline']:
        pipeline_info = status_data['pipeline']
        print(f"  Documents: {pipeline_info.get('document_count')}")
        print(f"  Model: {pipeline_info.get('model_name')}")
    
    # Test 4: Search endpoint
    print("\n4. Testing search endpoint...")
    
    if quick_test:
        print("⚡ Quick test mode - skipping search tests")
        print("   Start server with --skip-init and run full test for search functionality")
        print("\n✅ Quick API testing completed!")
        print(f"\n🌐 Visit http://localhost:8000/docs for interactive API documentation")
        return
    
    # Smart waiting logic based on initialization type
    if init_type == "first_time":
        print("🔄 First-time initialization - waiting without timeout...")
        print("   (Press Ctrl+C to cancel if needed)")
        max_wait = None  # No timeout for first-time init
        check_interval = 10  # Check every 10 seconds
    else:
        print("⏱️  Normal startup - timeout in 60 seconds...")
        max_wait = 60  # 1 minute timeout for normal startup
        check_interval = 5   # Check every 5 seconds
    
    start_time = time.time()
    
    while True:
        try:
            response = requests.get(f"{base_url}/health")
            health_data = response.json()
            status = health_data.get('status')
            
            if status == 'healthy':
                elapsed = int(time.time() - start_time)
                print(f"✅ Pipeline ready! (took {elapsed}s)")
                break
            
            # Check timeout only for normal startup
            if max_wait is not None and time.time() - start_time > max_wait:
                elapsed = int(time.time() - start_time)
                print(f"❌ Timeout after {elapsed}s")
                print("💡 Try restarting the server or check logs for errors")
                return
            
            # Show appropriate progress message
            elapsed = int(time.time() - start_time)
            
            if status == "first_time_init":
                print(f"   🔄 First-time setup in progress... ({elapsed}s elapsed)")
            elif status == "initializing":
                print(f"   ⚡ Loading cached data... ({elapsed}s elapsed)")
            else:
                print(f"   � Waiting for pipeline... ({status}) - {elapsed}s elapsed")
            
            time.sleep(check_interval)
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error: {e}")
            return
        except KeyboardInterrupt:
            print("\n⏹️  Test cancelled by user")
            return
    
    # Perform search
    search_queries = [
        {
            "query": "machine learning algorithms",
            "top_k": 3,
            "min_similarity": 0.1
        },
        {
            "query": "deep neural networks",
            "top_k": 2,
            "min_similarity": 0.2
        }
    ]
    
    for i, search_request in enumerate(search_queries, 1):
        print(f"\n  Search {i}: '{search_request['query']}'")
        
        response = requests.post(
            f"{base_url}/search",
            json=search_request,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            search_data = response.json()
            print(f"    ✓ Found {search_data['total_results']} results")
            print(f"    ✓ Processing time: {search_data['processing_time_ms']:.1f}ms")
            
            for j, result in enumerate(search_data['results'], 1):
                print(f"      {j}. {result['title']} (score: {result['similarity_score']:.3f})")
        else:
            print(f"    ❌ Search failed: {response.status_code}")
            print(f"    Error: {response.text}")
    
    print("\n✅ API testing completed!")
    print(f"\n🌐 Visit http://localhost:8000/docs for interactive API documentation")


if __name__ == "__main__":
    # Check if user wants to see info about smart testing
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h", "help"]:
        print_smart_testing_info()
        sys.exit(0)
    
    # Check if user wants quick test
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        print("🚀 Running quick test (endpoints only, no search)")
        test_api_server(quick_test=True)
    else:
        print("🔍 Running full test (includes search functionality)")
        print("💡 Use 'python test_api_client.py --quick' for faster testing")
        print("💡 Use 'python test_api_client.py --help' for smart testing info")
        print()
        test_api_server(quick_test=False)
