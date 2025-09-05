# Test Scripts Directory

This directory contains comprehensive test suites for all components of the RAG Pipeline system.

## 📁 Test Files

### **Individual Component Tests**
- **`test_utils.py`** - Core utility functions (device detection, embeddings normalization, similarity computation)
- **`test_models.py`** - Pydantic data models validation and serialization 
- **`test_data_loader.py`** - Wikipedia data fetching and document management
- **`test_embeddings.py`** - Text embedding generation and caching
- **`test_retrieval.py`** - Document retrieval and semantic search
- **`test_rag_pipeline.py`** - Complete RAG pipeline integration
- **`test_api_client.py`** - API endpoint testing with smart initialization detection

### **Test Runner**
- **`run_all_tests.py`** - Master test runner for all components

## 🚀 Running Tests

### **Run All Tests**
```bash
# From the Day7 directory
cd test_scripts
python run_all_tests.py
```

### **Quick Testing (Skip API)**
```bash
python run_all_tests.py --quick
```

### **Run Specific Component Tests**
```bash
python run_all_tests.py --test utils
python run_all_tests.py --test models
python run_all_tests.py --test data
python run_all_tests.py --test embeddings
python run_all_tests.py --test retrieval
python run_all_tests.py --test pipeline
python run_all_tests.py --test api
```

### **Run Individual Test Files**
```bash
python test_utils.py
python test_models.py
python test_data_loader.py
python test_embeddings.py
python test_retrieval.py
python test_rag_pipeline.py
python test_api_client.py
```

## 📊 Test Coverage

### **Utils Module (`test_utils.py`)**
- ✅ Device detection (CPU/GPU/MPS)
- ✅ Embedding normalization
- ✅ Cosine similarity computation
- ✅ Text truncation
- ✅ Reproducibility setup

### **Models Module (`test_models.py`)**
- ✅ QueryRequest validation
- ✅ DocumentResponse structure
- ✅ SearchResponse formatting
- ✅ HealthResponse status
- ✅ JSON serialization/deserialization
- ✅ Input validation and error handling

### **Data Loader Module (`test_data_loader.py`)**
- ✅ Document class functionality
- ✅ Wikipedia article fetching
- ✅ Corpus creation and caching
- ✅ Error handling for network issues
- ✅ Cache file management

### **Embeddings Module (`test_embeddings.py`)**
- ✅ EmbeddingGenerator initialization
- ✅ Query encoding
- ✅ Batch document encoding
- ✅ Embedding caching and validation
- ✅ Cache invalidation logic

### **Retrieval Module (`test_retrieval.py`)**
- ✅ SearchResult data structure
- ✅ DocumentRetriever initialization
- ✅ Semantic search functionality
- ✅ Top-k and similarity filtering
- ✅ Edge cases and error handling
- ✅ Content preview generation

### **RAG Pipeline (`test_rag_pipeline.py`)**
- ✅ End-to-end pipeline functionality
- ✅ Corpus initialization
- ✅ Search query processing
- ✅ Integration of all components

### **API Client (`test_api_client.py`)**
- ✅ Smart initialization detection
- ✅ Adaptive timeout logic
- ✅ All API endpoint testing
- ✅ Search functionality validation
- ✅ Error handling and user guidance

## 🧠 Smart Testing Features

### **Initialization Detection**
The test suite automatically detects:
- **First-time setup**: No cached data exists (5-10 minute timeout)
- **Normal startup**: Cached embeddings available (60-second timeout)
- **Quick testing**: Endpoint validation only (5-10 seconds)

### **Test Dependencies**
Tests are ordered by complexity and dependencies:
1. **Utils** → Core functions (no dependencies)
2. **Models** → Data validation (no dependencies)
3. **Data Loader** → Uses utils
4. **Embeddings** → Uses utils and data loader
5. **Retrieval** → Uses utils and embeddings
6. **Pipeline** → Integrates all components
7. **API** → Tests complete system

### **Error Resilience**
- Network failures are handled gracefully
- External API limits don't fail tests
- Missing dependencies are clearly reported
- Partial failures don't stop the test suite

## 📈 Example Output

```bash
🚀 RAG Pipeline Test Suite
============================================================
Running 7 test suites...

============================================================
🧪 Running Utils Module Tests
============================================================
🔧 Testing device detection...
✓ Detected device: cpu
✓ Device cpu is functional
✅ test_get_device passed

📐 Testing embedding normalization...
✓ All embeddings normalized to unit length
✅ test_normalize_embeddings passed

============================================================
📊 TEST SUITE SUMMARY
============================================================
Total time: 45.67 seconds
Tests run: 7
Passed: 7
Failed: 0

✅ PASS Utils Module Tests
✅ PASS Models Module Tests
✅ PASS Data Loader Module Tests
✅ PASS Embeddings Module Tests
✅ PASS Retrieval Module Tests
✅ PASS RAG Pipeline Integration Tests
✅ PASS API Client Tests

🎉 ALL TESTS PASSED! 🎉
✅ RAG Pipeline is working correctly
```

## 🔧 Troubleshooting

### **Import Errors**
All test files are configured to import from the parent `src/` directory. Run tests from the `test_scripts` directory.

### **Network Issues**
Data loader and API tests may be affected by network connectivity. Tests are designed to handle this gracefully.

### **Missing Dependencies**
Ensure all required packages are installed:
```bash
pip install -r ../requirements.txt
```

### **Permission Issues**
Make sure the test files have execution permissions and the parent directory is writable for cache files.

## 💡 Tips

- Use `--quick` mode for rapid development testing
- Run specific component tests when debugging
- Check individual test outputs for detailed error information
- The master test runner provides comprehensive coverage reporting

This test suite ensures the reliability and correctness of all RAG Pipeline components! 🎯
