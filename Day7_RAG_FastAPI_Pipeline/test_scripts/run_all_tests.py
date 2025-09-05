"""
Master test runner for all RAG Pipeline components.

This script runs all individual component tests and provides a comprehensive overview.
"""

import sys
import os
import time
import importlib.util
from pathlib import Path

def load_test_module(test_file_path):
    """Dynamically load a test module."""
    module_name = Path(test_file_path).stem
    spec = importlib.util.spec_from_file_location(module_name, test_file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def run_individual_test(test_file_path, test_name):
    """Run an individual test file."""
    print(f"\n{'='*60}")
    print(f"🧪 Running {test_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Load and run the test module
        test_module = load_test_module(test_file_path)
        
        if hasattr(test_module, 'main'):
            success = test_module.main()
        else:
            print(f"❌ Test module {test_name} has no main() function")
            return False
        
        elapsed_time = time.time() - start_time
        
        if success:
            print(f"✅ {test_name} completed successfully in {elapsed_time:.2f}s")
            return True
        else:
            print(f"❌ {test_name} failed in {elapsed_time:.2f}s")
            return False
            
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"💥 {test_name} crashed in {elapsed_time:.2f}s: {e}")
        return False

def run_all_tests(quick_mode=False):
    """Run all available tests."""
    
    # Get the directory containing this script
    test_scripts_dir = Path(__file__).parent
    
    # Define test files in dependency order (simple to complex)
    test_files = [
        ("test_utils.py", "Utils Module Tests"),
        ("test_models.py", "Models Module Tests"),
        ("test_data_loader.py", "Data Loader Module Tests"),
        ("test_embeddings.py", "Embeddings Module Tests"),
        ("test_retrieval.py", "Retrieval Module Tests"),
        ("test_rag_pipeline.py", "RAG Pipeline Integration Tests"),
    ]
    
    # Add API tests if not in quick mode
    if not quick_mode:
        test_files.append(("test_api_client.py", "API Client Tests"))
    
    print("🚀 RAG Pipeline Test Suite")
    print("=" * 60)
    print(f"Running {len(test_files)} test suites...")
    if quick_mode:
        print("⚡ Quick mode: Skipping API tests")
    print()
    
    # Track results
    results = []
    total_start_time = time.time()
    
    for test_file, test_name in test_files:
        test_path = test_scripts_dir / test_file
        
        if test_path.exists():
            success = run_individual_test(test_path, test_name)
            results.append((test_name, success))
        else:
            print(f"⚠️  Test file not found: {test_file}")
            results.append((test_name, False))
    
    # Summary
    total_elapsed = time.time() - total_start_time
    passed = sum(1 for _, success in results if success)
    failed = len(results) - passed
    
    print(f"\n{'='*60}")
    print("📊 TEST SUITE SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_elapsed:.2f} seconds")
    print(f"Tests run: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print()
    
    # Detailed results
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print()
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! 🎉")
        print("✅ RAG Pipeline is working correctly")
        return True
    else:
        print(f"❌ {failed} test suite(s) failed")
        print("💡 Check individual test outputs above for details")
        return False

def run_specific_test(test_name):
    """Run a specific test by name."""
    test_scripts_dir = Path(__file__).parent
    
    # Map test names to files
    test_mapping = {
        "utils": "test_utils.py",
        "models": "test_models.py", 
        "data": "test_data_loader.py",
        "loader": "test_data_loader.py",
        "embeddings": "test_embeddings.py",
        "retrieval": "test_retrieval.py",
        "pipeline": "test_rag_pipeline.py",
        "api": "test_api_client.py"
    }
    
    test_file = test_mapping.get(test_name.lower())
    
    if not test_file:
        print(f"❌ Unknown test: {test_name}")
        print(f"Available tests: {', '.join(test_mapping.keys())}")
        return False
    
    test_path = test_scripts_dir / test_file
    
    if not test_path.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    return run_individual_test(test_path, f"{test_name.title()} Tests")

def print_help():
    """Print usage help."""
    print("🧪 RAG Pipeline Test Runner")
    print("=" * 40)
    print("Usage:")
    print("  python run_all_tests.py                 # Run all tests")
    print("  python run_all_tests.py --quick         # Run all tests except API")
    print("  python run_all_tests.py --test <name>   # Run specific test")
    print("  python run_all_tests.py --help          # Show this help")
    print()
    print("Available individual tests:")
    print("  utils       - Core utility functions")
    print("  models      - Pydantic data models")
    print("  data        - Data loading and Wikipedia fetching")
    print("  embeddings  - Text embedding generation")
    print("  retrieval   - Document retrieval and search")
    print("  pipeline    - Complete RAG pipeline")
    print("  api         - API endpoint testing")
    print()
    print("Examples:")
    print("  python run_all_tests.py --test utils")
    print("  python run_all_tests.py --quick")

def main():
    """Main entry point."""
    
    # Parse command line arguments
    if len(sys.argv) == 1:
        # Run all tests
        success = run_all_tests(quick_mode=False)
        sys.exit(0 if success else 1)
    
    elif "--help" in sys.argv or "-h" in sys.argv:
        print_help()
        sys.exit(0)
    
    elif "--quick" in sys.argv:
        # Run all tests except API
        success = run_all_tests(quick_mode=True)
        sys.exit(0 if success else 1)
    
    elif "--test" in sys.argv:
        try:
            test_index = sys.argv.index("--test")
            if test_index + 1 < len(sys.argv):
                test_name = sys.argv[test_index + 1]
                success = run_specific_test(test_name)
                sys.exit(0 if success else 1)
            else:
                print("❌ --test requires a test name")
                print_help()
                sys.exit(1)
        except ValueError:
            print("❌ Invalid --test usage")
            print_help()
            sys.exit(1)
    
    else:
        print(f"❌ Unknown arguments: {' '.join(sys.argv[1:])}")
        print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
