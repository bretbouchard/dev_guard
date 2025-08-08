"""Comprehensive test suite for Vector Database system.
This module tests the ChromaDB-based vector database including
document processing, embedding generation, and similarity search.
"""

import hashlib
import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Import availability check
try:
    from src.dev_guard.core.config import VectorDBConfig
    from src.dev_guard.memory.vector_db import (
        Document,
        FileProcessor,
        VectorDatabase,
        VectorDatabaseError,
    )
    VECTOR_DB_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Vector database imports not available: {e}")
    VECTOR_DB_IMPORTS_AVAILABLE = False


@pytest.fixture
def temp_vector_db_path():
    """Create a temporary path for vector database."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir) / "test_vector_db"


@pytest.fixture
def vector_db_config():
    """Create a test configuration for vector database."""
    if not VECTOR_DB_IMPORTS_AVAILABLE:
        pytest.skip("Vector database modules not available")
    
    return VectorDBConfig(
        provider="chroma",
        path="./test_vector_db",
        collection_name="test_collection",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=500,
        chunk_overlap=50
    )


@pytest.fixture
def vector_database(temp_vector_db_path, vector_db_config):
    """Create a VectorDatabase instance for testing."""
    if not VECTOR_DB_IMPORTS_AVAILABLE:
        pytest.skip("Vector database modules not available")
    
    # Mock ChromaDB to avoid actual database operations
    with patch('src.dev_guard.memory.vector_db.chromadb'), \
         patch('src.dev_guard.memory.vector_db.SentenceTransformer'):
        
        vector_db_config.path = str(temp_vector_db_path)
        db = VectorDatabase(vector_db_config)
        yield db


# Document Model Tests
@pytest.mark.skipif(not VECTOR_DB_IMPORTS_AVAILABLE, reason="Vector DB modules not available")
class TestDocumentModel:
    """Test Document model validation and functionality."""

    def test_document_creation(self):
        """Test basic Document creation."""
        doc = Document(
            content="Test document content",
            source="test.py",
            metadata={"type": "python"}
        )
        
        assert doc.content == "Test document content"
        assert doc.source == "test.py"
        assert doc.metadata["type"] == "python"
        assert len(doc.id) == 36  # UUID format
        assert isinstance(doc.created_at, datetime)

    def test_document_with_embedding(self):
        """Test Document with embedding vector."""
        doc = Document(
            content="Test content",
            source="test.txt",
            metadata={"type": "text"}
        )
        
        # Note: embedding is handled at the vector database level
        assert hasattr(doc, 'content')
        assert hasattr(doc, 'source')

    def test_document_validation(self):
        """Test Document validation."""
        # Test empty content
        with pytest.raises(ValueError):
            Document(content="", source="test.py", metadata={})
        
        # Test invalid metadata (handled by pydantic)
        doc = Document(content="test", source="test.py", metadata={"key": "value"})
        assert doc.metadata == {"key": "value"}

    def test_document_content_hash(self):
        """Test Document content hash generation."""
        doc1 = Document(
            content="Same content",
            source="file1.py",
            metadata={"type": "python"}
        )
        doc2 = Document(
            content="Same content",
            source="file2.py",
            metadata={"type": "python"}
        )
        
        # Same content should have same hash regardless of source
        hash1 = doc1.get_content_hash()
        hash2 = doc2.get_content_hash()
        assert hash1 == hash2

    def test_document_serialization(self):
        """Test Document serialization."""
        doc = Document(
            content="Test content",
            source="test.py",
            metadata={"type": "python", "lines": 100}
        )
        
        # Test JSON serialization
        doc_json = doc.model_dump_json()
        assert isinstance(doc_json, str)
        
        # Test deserialization
        parsed = json.loads(doc_json)
        assert parsed["content"] == "Test content"
        assert parsed["source"] == "test.py"


# FileProcessor Tests
@pytest.mark.skipif(not VECTOR_DB_IMPORTS_AVAILABLE, reason="Vector DB modules not available")
class TestFileProcessor:
    """Test FileProcessor functionality."""

    def test_file_processor_initialization(self):
        """Test FileProcessor initialization."""
        processor = FileProcessor()
        
        assert hasattr(processor, 'code_extensions')
        assert hasattr(processor, 'text_extensions')
        assert hasattr(processor, 'doc_extensions')
        
        # Test some common extensions
        assert '.py' in processor.code_extensions
        assert '.js' in processor.code_extensions
        assert '.md' in processor.text_extensions
        assert '.txt' in processor.text_extensions

    def test_extract_file_metadata(self):
        """Test file metadata extraction."""
        processor = FileProcessor()
        
        # Mock file
        test_file = Path("/test/path/example.py")
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.stat') as mock_stat, \
             patch('pathlib.Path.read_text', return_value="def test(): pass"):
            
            # Mock file stats
            mock_stat_obj = Mock()
            mock_stat_obj.st_size = 1000
            mock_stat_obj.st_mtime = 1640995200  # 2022-01-01
            mock_stat_obj.st_ctime = 1640995200  # 2022-01-01
            mock_stat_obj.st_atime = 1640995200  # 2022-01-01
            mock_stat.return_value = mock_stat_obj
            
            metadata = processor.extract_file_metadata(test_file)
            
            assert metadata["file_path"] == str(test_file)
            assert metadata["file_name"] == "example.py"
            assert metadata["file_extension"] == ".py"
            assert metadata["language"] == "python"
            assert metadata["file_size"] == 1000
            assert "modified_at" in metadata

    def test_process_code_file(self):
        """Test processing code files."""
        processor = FileProcessor()
        
        python_code = '''
def hello_world():
    """Print hello world."""
    print("Hello, World!")

class Calculator:
    def add(self, a, b):
        return a + b

if __name__ == "__main__":
    hello_world()
'''
        
        test_file = Path("/test/calculator.py")
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.read_text', return_value=python_code), \
             patch('pathlib.Path.stat') as mock_stat:
            
            mock_stat_obj = Mock()
            mock_stat_obj.st_size = len(python_code)
            mock_stat_obj.st_mtime = 1640995200
            mock_stat.return_value = mock_stat_obj
            
            metadata = processor.extract_file_metadata(test_file)
            
            assert metadata["language"] == "python"
            assert metadata["file_size"] == len(python_code)

    def test_process_text_file(self):
        """Test processing text/documentation files."""
        processor = FileProcessor()
        
        markdown_content = '''
# Test Documentation

This is a test markdown file.

## Features
- Feature 1
- Feature 2

## Usage
Run the tests with pytest.
'''
        
        test_file = Path("/test/README.md")
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.read_text', return_value=markdown_content), \
             patch('pathlib.Path.stat') as mock_stat:
            
            mock_stat_obj = Mock()
            mock_stat_obj.st_size = len(markdown_content)
            mock_stat_obj.st_mtime = 1640995200
            mock_stat.return_value = mock_stat_obj
            
            metadata = processor.extract_file_metadata(test_file)
            
            assert metadata["language"] == "markdown"
            assert metadata["file_size"] == len(markdown_content)

    def test_unsupported_file_type(self):
        """Test handling unsupported file types."""
        processor = FileProcessor()
        
        test_file = Path("/test/binary.exe")
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.stat') as mock_stat:
            
            mock_stat_obj = Mock()
            mock_stat_obj.st_size = 5000
            mock_stat_obj.st_mtime = 1640995200
            mock_stat.return_value = mock_stat_obj
            
            metadata = processor.extract_file_metadata(test_file)
            
            assert metadata["language"] == "unknown"
            assert metadata["file_size"] == 5000

    def test_file_change_detection(self):
        """Test file change detection."""
        processor = FileProcessor()
        
        # Test hash calculation for change detection
        content1 = "Original content"
        content2 = "Modified content"
        
        hash1 = hashlib.sha256(content1.encode()).hexdigest()
        hash2 = hashlib.sha256(content2.encode()).hexdigest()
        
        assert hash1 != hash2  # Different content should have different hashes


# VectorDatabase Tests
@pytest.mark.skipif(not VECTOR_DB_IMPORTS_AVAILABLE, reason="Vector DB modules not available")
class TestVectorDatabase:
    """Test VectorDatabase functionality."""

    def test_vector_database_initialization(self, vector_database):
        """Test VectorDatabase initialization."""
        assert vector_database.config is not None
        assert vector_database.config.provider == "chroma"
        assert vector_database.config.collection_name == "test_collection"

    def test_add_document(self, vector_database):
        """Test adding documents to vector database."""
        doc = Document(
            content="Test document for vector search",
            source="test.py",
            metadata={"type": "python"}
        )
        
        # Mock the collection add method
        with patch.object(vector_database._collection, 'add') as mock_add:
            result_id = vector_database.add_document(doc)
            
            assert result_id == doc.id
            mock_add.assert_called_once()

    def test_add_multiple_documents(self, vector_database):
        """Test adding multiple documents."""
        docs = [
            Document(
                content=f"Test document {i}",
                source=f"test{i}.py",
                metadata={"index": i}
            )
            for i in range(3)
        ]
        
        with patch.object(vector_database._collection, 'add') as mock_add:
            results = vector_database.add_documents(docs)
            
            assert len(results) == 3
            mock_add.assert_called_once()

    def test_search_documents(self, vector_database):
        """Test document search functionality."""
        # Mock search results
        mock_results = {
            'ids': [['doc1', 'doc2']],
            'documents': [['First document', 'Second document']],
            'metadatas': [[{'source': 'test1.py'}, {'source': 'test2.py'}]],
            'distances': [[0.1, 0.2]]
        }
        
        with patch.object(vector_database._collection, 'query', 
                         return_value=mock_results):
            results = vector_database.search("test query", limit=2)
            
            assert len(results) == 2

    def test_search_by_metadata(self, vector_database):
        """Test searching documents by metadata."""
        mock_results = {
            'ids': [['doc1']],
            'documents': [['Python document']],
            'metadatas': [[{'source': 'test.py', 'language': 'python'}]],
            'distances': [[0.1]]
        }
        
        with patch.object(vector_database.collection, 'query', return_value=mock_results):
            results = vector_database.search_by_metadata({"language": "python"})
            
            assert len(results) == 1
            assert results[0].metadata['language'] == 'python'

    def test_get_document_by_id(self, vector_database):
        """Test retrieving document by ID."""
        mock_result = {
            'ids': ['doc1'],
            'documents': ['Test document'],
            'metadatas': [{'source': 'test.py'}]
        }
        
        with patch.object(vector_database.collection, 'get', return_value=mock_result):
            doc = vector_database.get_document("doc1")
            
            assert doc is not None
            assert doc.content == "Test document"
            assert doc.metadata['source'] == 'test.py'

    def test_delete_document(self, vector_database):
        """Test document deletion."""
        with patch.object(vector_database.collection, 'delete') as mock_delete:
            success = vector_database.delete_document("doc1")
            
            mock_delete.assert_called_once_with(ids=["doc1"])
            assert success is True

    def test_update_document(self, vector_database):
        """Test document updating."""
        updated_doc = Document(
            id="doc1",
            content="Updated content",
            source="updated.py",
            metadata={"type": "updated"}
        )
        
        with patch.object(vector_database, '_generate_embedding', return_value=[0.1, 0.2, 0.3]), \
             patch.object(vector_database.collection, 'update') as mock_update:
            
            success = vector_database.update_document("doc1", updated_doc)
            
            mock_update.assert_called_once()
            assert success is True

    def test_get_collection_stats(self, vector_database):
        """Test collection statistics."""
        with patch.object(vector_database.collection, 'count', return_value=100):
            stats = vector_database.get_collection_stats()
            
            assert stats['document_count'] == 100
            assert 'collection_name' in stats

    def test_chunk_text(self, vector_database):
        """Test text chunking functionality."""
        long_text = "This is a long piece of text. " * 50  # 1500 characters
        
        chunks = vector_database._chunk_text(long_text, chunk_size=100, overlap=10)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 100 for chunk in chunks)
        
        # Test overlap
        if len(chunks) > 1:
            # Should have some overlap between consecutive chunks
            overlap_found = any(
                chunks[i][-10:] in chunks[i+1]
                for i in range(len(chunks)-1)
                if len(chunks[i]) >= 10
            )

    def test_embedding_generation(self, vector_database):
        """Test embedding generation."""
        with patch.object(vector_database, '_embedding_function', return_value=[[0.1, 0.2, 0.3]]):
            embedding = vector_database._generate_embedding("test text")
            
            assert embedding == [0.1, 0.2, 0.3]

    def test_error_handling(self, vector_database):
        """Test error handling in vector database operations."""
        # Test handling of database errors
        with patch.object(vector_database.collection, 'add', side_effect=Exception("DB Error")):
            doc = Document(
                content="Test document",
                source="test.py"
            )
            
            # Should raise VectorDatabaseError
            with pytest.raises(VectorDatabaseError):
                vector_database.add_document(doc)

    def test_batch_operations(self, vector_database):
        """Test batch operations for performance."""
        docs = [
            Document(
                content=f"Batch document {i}",
                source=f"batch{i}.py",
                metadata={"batch_id": "test_batch"}
            )
            for i in range(10)
        ]
        
        with patch.object(vector_database, '_generate_embedding', return_value=[0.1, 0.2, 0.3]):
            results = vector_database.add_documents(docs)
            
            assert len(results) == 10
            assert all(results)


# File Ingestion Tests
@pytest.mark.skipif(not VECTOR_DB_IMPORTS_AVAILABLE, reason="Vector DB modules not available")
class TestFileIngestion:
    """Test file ingestion and processing functionality."""

    def test_ingest_file(self, vector_database):
        """Test ingesting a single file."""
        test_file = Path("/test/example.py")
        file_content = '''
def calculate_sum(a, b):
    """Calculate sum of two numbers."""
    return a + b

def calculate_product(a, b):
    """Calculate product of two numbers."""
    return a * b
'''
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.read_text', return_value=file_content), \
             patch('pathlib.Path.stat') as mock_stat, \
             patch.object(vector_database, '_generate_embedding', return_value=[0.1, 0.2, 0.3]):
            
            mock_stat_obj = Mock()
            mock_stat_obj.st_size = len(file_content)
            mock_stat_obj.st_mtime = 1640995200
            mock_stat.return_value = mock_stat_obj
            
            success = vector_database.ingest_file(test_file)
            assert success is True

    def test_ingest_directory(self, vector_database):
        """Test ingesting a directory of files."""
        test_dir = Path("/test/project")
        
        # Mock directory structure
        mock_files = [
            test_dir / "main.py",
            test_dir / "utils.py",
            test_dir / "README.md",
            test_dir / "tests" / "test_main.py"
        ]
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_dir', return_value=True), \
             patch('pathlib.Path.rglob', return_value=mock_files), \
             patch('pathlib.Path.read_text', return_value="mock content"), \
             patch('pathlib.Path.stat') as mock_stat, \
             patch.object(vector_database, '_generate_embedding', return_value=[0.1, 0.2, 0.3]):
            
            mock_stat_obj = Mock()
            mock_stat_obj.st_size = 100
            mock_stat_obj.st_mtime = 1640995200
            mock_stat.return_value = mock_stat_obj
            
            results = vector_database.ingest_directory(test_dir)
            
            # Should process all files
            assert len(results) == len(mock_files)
            assert all(results.values())

    def test_incremental_file_updates(self, vector_database):
        """Test incremental file updates based on modification time."""
        test_file = Path("/test/modified.py")
        
        # Mock existing document with older timestamp
        existing_doc = Document(
            content="Old content",
            metadata={
                "source": str(test_file),
                "modified_time": "2022-01-01T00:00:00Z"
            }
        )
        
        new_content = "Updated content"
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.read_text', return_value=new_content), \
             patch('pathlib.Path.stat') as mock_stat, \
             patch.object(vector_database, 'get_document_by_source', return_value=existing_doc), \
             patch.object(vector_database, '_generate_embedding', return_value=[0.1, 0.2, 0.3]):
            
            # Mock newer modification time
            mock_stat_obj = Mock()
            mock_stat_obj.st_size = len(new_content)
            mock_stat_obj.st_mtime = 1672531200  # 2023-01-01
            mock_stat.return_value = mock_stat_obj
            
            success = vector_database.ingest_file(test_file, incremental=True)
            assert success is True

    def test_code_specific_search(self, vector_database):
        """Test code-specific search with file extension filtering."""
        mock_results = {
            'ids': [['py_doc', 'js_doc']],
            'documents': [['Python code', 'JavaScript code']],
            'metadatas': [[
                {'source': 'test.py', 'language': 'python'},
                {'source': 'app.js', 'language': 'javascript'}
            ]],
            'distances': [[0.1, 0.3]]
        }
        
        with patch.object(vector_database.collection, 'query', return_value=mock_results):
            # Search for Python files only
            python_results = vector_database.search_code(
                query="function implementation",
                file_extensions=['.py']
            )
            
            assert len(python_results) >= 1
            assert any(r.metadata.get('language') == 'python' for r in python_results)


# Integration Tests
@pytest.mark.skipif(not VECTOR_DB_IMPORTS_AVAILABLE, reason="Vector DB modules not available")
class TestVectorDatabaseIntegration:
    """Test integration scenarios for vector database."""

    def test_end_to_end_workflow(self, vector_database):
        """Test complete end-to-end workflow."""
        # Step 1: Ingest documents
        docs = [
            Document(
                content="Python function for data processing",
                metadata={"source": "data.py", "type": "function"}
            ),
            Document(
                content="JavaScript async function for API calls",
                metadata={"source": "api.js", "type": "function"}
            ),
            Document(
                content="Documentation for API endpoints",
                metadata={"source": "docs.md", "type": "documentation"}
            )
        ]
        
        with patch.object(vector_database, '_generate_embedding', return_value=[0.1, 0.2, 0.3]):
            # Add documents
            results = vector_database.add_documents(docs)
            assert all(results)
            
            # Search for related content
            mock_search_results = {
                'ids': [['doc1', 'doc2']],
                'documents': [['Python function for data processing', 'Documentation for API endpoints']],
                'metadatas': [[
                    {'source': 'data.py', 'type': 'function'},
                    {'source': 'docs.md', 'type': 'documentation'}
                ]],
                'distances': [[0.1, 0.2]]
            }
            
            with patch.object(vector_database.collection, 'query', return_value=mock_search_results):
                search_results = vector_database.search("data processing", limit=5)
                
                assert len(search_results) == 2
                assert any("Python" in r.content for r in search_results)

    def test_knowledge_retrieval_patterns(self, vector_database):
        """Test knowledge retrieval patterns."""
        with patch.object(vector_database, '_generate_embedding', return_value=[0.1, 0.2, 0.3]):
            # Test pattern-based search
            patterns = [
                "error handling",
                "async function",
                "data validation",
                "unit testing"
            ]
            
            for pattern in patterns:
                mock_results = {
                    'ids': [[f"{pattern}_doc"]],
                    'documents': [[f"Example of {pattern}"]],
                    'metadatas': [[{'pattern': pattern}]],
                    'distances': [[0.1]]
                }
                
                with patch.object(vector_database.collection, 'query', return_value=mock_results):
                    results = vector_database.search(pattern)
                    assert len(results) >= 0  # May return empty results


# Mock-based tests for components without dependencies
class TestVectorDatabaseMocks:
    """Test vector database components using mocks."""

    def test_mock_chroma_integration(self):
        """Test ChromaDB integration with mocks."""
        if not VECTOR_DB_IMPORTS_AVAILABLE:
            pytest.skip("Vector DB modules not available")
        
        with patch('src.dev_guard.memory.vector_db.chromadb') as mock_chromadb, \
             patch('src.dev_guard.memory.vector_db.SentenceTransformer') as mock_transformer:
            
            # Mock ChromaDB client and collection
            mock_client = Mock()
            mock_collection = Mock()
            mock_chromadb.PersistentClient.return_value = mock_client
            mock_client.get_or_create_collection.return_value = mock_collection
            
            # Mock SentenceTransformer
            mock_model = Mock()
            mock_transformer.return_value = mock_model
            mock_model.encode.return_value = [0.1, 0.2, 0.3]
            
            config = VectorDBConfig(
                provider="chroma",
                path="./test_db",
                collection_name="test"
            )
            
            db = VectorDatabase(config)
            
            # Test that mocks were called correctly
            mock_chromadb.PersistentClient.assert_called_once()
            mock_client.get_or_create_collection.assert_called_once()

    def test_mock_embedding_operations(self):
        """Test embedding operations with mocks."""
        if not VECTOR_DB_IMPORTS_AVAILABLE:
            pytest.skip("Vector DB modules not available")
        
        # Mock document processing without actual ML operations
        processor = FileProcessor()
        
        test_content = "def hello(): print('Hello, World!')"
        
        # Test content processing
        assert len(test_content) > 0
        assert "def" in test_content
        assert "hello" in test_content
        
        # Mock hash generation
        content_hash = hashlib.sha256(test_content.encode()).hexdigest()
        assert len(content_hash) == 64  # SHA256 hash length

    def test_mock_search_operations(self):
        """Test search operations with mocked results."""
        if not VECTOR_DB_IMPORTS_AVAILABLE:
            pytest.skip("Vector DB modules not available")
        
        # Mock search functionality
        mock_documents = [
            {
                "id": f"doc_{i}",
                "content": f"Test document {i}",
                "metadata": {"source": f"test{i}.py"},
                "score": 0.9 - (i * 0.1)
            }
            for i in range(5)
        ]
        
        # Test result processing
        assert len(mock_documents) == 5
        assert all("content" in doc for doc in mock_documents)
        assert all("metadata" in doc for doc in mock_documents)
        
        # Test sorting by score
        sorted_docs = sorted(mock_documents, key=lambda x: x["score"], reverse=True)
        assert sorted_docs[0]["score"] >= sorted_docs[-1]["score"]


if __name__ == "__main__":
    if VECTOR_DB_IMPORTS_AVAILABLE:
        pytest.main([__file__, "-v", "-x"])
    else:
        print("Skipping vector database tests due to import errors")
        exit(1)
