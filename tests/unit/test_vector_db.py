"""Unit tests for vector database functionality."""

import shutil
import tempfile
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from src.dev_guard.core.config import VectorDBConfig
from src.dev_guard.memory.vector_db import (
    Document,
    FileProcessor,
    TextChunker,
    VectorDatabase,
    VectorDatabaseError,
)


class TestDocument:
    """Test Document model validation and functionality."""
    
    def test_document_creation_valid(self):
        """Test creating a valid document."""
        doc = Document(
            content="Test content",
            source="test.py",
            metadata={"type": "test"}
        )
        
        assert doc.content == "Test content"
        assert doc.source == "test.py"
        assert doc.chunk_index == 0
        assert doc.metadata == {"type": "test"}
        assert isinstance(doc.created_at, datetime)
        assert len(doc.id) == 36  # UUID length
    
    def test_document_content_validation(self):
        """Test document content validation."""
        # Empty content should fail
        with pytest.raises(Exception):  # Pydantic raises ValidationError, not ValueError
            Document(content="", source="test.py")
        
        # Whitespace-only content should fail
        with pytest.raises(ValueError, match="content cannot be empty"):
            Document(content="   ", source="test.py")
        
        # Content should be stripped
        doc = Document(content="  test content  ", source="test.py")
        assert doc.content == "test content"
    
    def test_document_source_validation(self):
        """Test document source validation."""
        # Empty source should fail
        with pytest.raises(Exception):  # Pydantic raises ValidationError, not ValueError
            Document(content="test", source="")
        
        # Whitespace-only source should fail
        with pytest.raises(ValueError, match="source cannot be empty"):
            Document(content="test", source="   ")
        
        # Source should be stripped
        doc = Document(content="test", source="  test.py  ")
        assert doc.source == "test.py"
    
    def test_document_metadata_validation(self):
        """Test document metadata validation."""
        # Valid JSON-serializable metadata should work
        doc = Document(
            content="test",
            source="test.py",
            metadata={"key": "value", "number": 42, "bool": True}
        )
        assert doc.metadata == {"key": "value", "number": 42, "bool": True}
        
        # Non-JSON-serializable metadata should fail
        with pytest.raises(ValueError, match="metadata must be JSON-serializable"):
            Document(
                content="test",
                source="test.py",
                metadata={"func": lambda x: x}  # Functions are not JSON-serializable
            )
    
    def test_document_content_hash(self):
        """Test document content hash generation."""
        doc1 = Document(content="test content", source="test.py")
        doc2 = Document(content="test content", source="other.py")
        doc3 = Document(content="different content", source="test.py")
        
        # Same content should have same hash regardless of source
        assert doc1.get_content_hash() == doc2.get_content_hash()
        
        # Different content should have different hash
        assert doc1.get_content_hash() != doc3.get_content_hash()
        
        # Hash should be consistent
        assert doc1.get_content_hash() == doc1.get_content_hash()


class TestTextChunker:
    """Test TextChunker functionality."""
    
    def test_chunker_initialization(self):
        """Test chunker initialization with valid parameters."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        assert chunker.chunk_size == 100
        assert chunker.chunk_overlap == 20
    
    def test_chunker_initialization_invalid(self):
        """Test chunker initialization with invalid parameters."""
        # Overlap >= chunk_size should fail
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            TextChunker(chunk_size=100, chunk_overlap=100)
        
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            TextChunker(chunk_size=100, chunk_overlap=150)
    
    def test_chunk_text_small(self):
        """Test chunking text smaller than chunk size."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = "This is a small text."
        
        chunks = chunker.chunk_text(text, "test.txt")
        
        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].source == "test.txt"
        assert chunks[0].chunk_index == 0
        assert chunks[0].metadata["total_chunks"] == 1
    
    def test_chunk_text_large(self):
        """Test chunking text larger than chunk size."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "This is a longer text that should be split into multiple chunks because it exceeds the chunk size limit."
        
        chunks = chunker.chunk_text(text, "test.txt")
        
        assert len(chunks) > 1
        
        # Check that all chunks have proper metadata
        for i, chunk in enumerate(chunks):
            assert chunk.source == "test.txt"
            assert chunk.chunk_index == i
            assert chunk.metadata["total_chunks"] == len(chunks)
            assert chunk.metadata["chunk_index"] == i
            assert "start_pos" in chunk.metadata
            assert "end_pos" in chunk.metadata
            assert "chunk_size" in chunk.metadata
    
    def test_chunk_text_empty(self):
        """Test chunking empty or whitespace-only text."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        
        # Empty text
        chunks = chunker.chunk_text("", "test.txt")
        assert len(chunks) == 0
        
        # Whitespace-only text
        chunks = chunker.chunk_text("   ", "test.txt")
        assert len(chunks) == 0
    
    def test_chunk_text_with_metadata(self):
        """Test chunking text with additional metadata."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "This is a test text for chunking with metadata."
        metadata = {"type": "test", "author": "tester"}
        
        chunks = chunker.chunk_text(text, "test.txt", metadata)
        
        for chunk in chunks:
            assert chunk.metadata["type"] == "test"
            assert chunk.metadata["author"] == "tester"
            assert "total_chunks" in chunk.metadata
            # chunk_index is stored in the Document model, not in metadata
            assert chunk.chunk_index >= 0
    
    def test_chunk_python_code(self):
        """Test chunking Python code with function boundaries."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        code = '''def function1():
    """First function."""
    return "hello"

def function2():
    """Second function."""
    return "world"

class TestClass:
    """Test class."""
    def method1(self):
        return "test"
'''
        
        chunks = chunker.chunk_code(code, "test.py", "python")
        
        assert len(chunks) > 0
        
        # Check that chunks have code-specific metadata
        for chunk in chunks:
            assert chunk.metadata["content_type"] == "code"
            assert chunk.metadata["language"] == "python"
            assert chunk.metadata["file_extension"] == ".py"
    
    def test_chunk_javascript_code(self):
        """Test chunking JavaScript code with function boundaries."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        code = '''function test1() {
    return "hello";
}

const test2 = () => {
    return "world";
}

class TestClass {
    constructor() {
        this.value = "test";
    }
}
'''
        
        chunks = chunker.chunk_code(code, "test.js", "javascript")
        
        assert len(chunks) > 0
        
        # Check that chunks have code-specific metadata
        for chunk in chunks:
            assert chunk.metadata["content_type"] == "code"
            assert chunk.metadata["language"] == "javascript"
            assert chunk.metadata["file_extension"] == ".js"
    
    def test_chunk_code_fallback(self):
        """Test that unknown languages fall back to text chunking."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        code = "Some code in an unknown language that should be chunked as text."
        
        chunks = chunker.chunk_code(code, "test.unknown", "unknown")
        
        assert len(chunks) > 0
        assert chunks[0].metadata["content_type"] == "code"
        assert chunks[0].metadata["language"] == "unknown"


@pytest.fixture
def temp_db_path():
    """Create a temporary directory for test database."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def vector_config(temp_db_path):
    """Create a test vector database configuration."""
    return VectorDBConfig(
        path=temp_db_path,
        collection_name="test_collection",
        chunk_size=100,
        chunk_overlap=20,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )


@pytest.fixture
def mock_chromadb():
    """Mock ChromaDB client and collection."""
    with patch('src.dev_guard.memory.vector_db.chromadb') as mock_chromadb:
        # Mock client
        mock_client = Mock()
        mock_chromadb.Client.return_value = mock_client
        
        # Mock collection
        mock_collection = Mock()
        # First call to get_collection should fail (collection doesn't exist)
        # Second call should succeed (after creation)
        mock_client.get_collection.side_effect = [Exception("Collection not found"), mock_collection]
        mock_client.create_collection.return_value = mock_collection
        
        # Mock embedding function
        mock_embedding_func = Mock()
        
        with patch('src.dev_guard.memory.vector_db.embedding_functions') as mock_ef:
            mock_ef.SentenceTransformerEmbeddingFunction.return_value = mock_embedding_func
            
            yield {
                'client': mock_client,
                'collection': mock_collection,
                'embedding_function': mock_embedding_func
            }


class TestFileProcessor:
    """Test FileProcessor functionality."""
    
    def test_initialization(self):
        """Test FileProcessor initialization."""
        processor = FileProcessor()
        
        assert processor.code_extensions is not None
        assert processor.text_extensions is not None
        assert processor.doc_extensions is not None
        assert '.py' in processor.code_extensions
        assert '.md' in processor.text_extensions
        assert '.md' in processor.doc_extensions
    
    def test_should_process_file(self, tmp_path):
        """Test file processing decision logic."""
        processor = FileProcessor()
        
        # Create test files
        py_file = tmp_path / "test.py"
        py_file.write_text("def test(): pass")
        
        md_file = tmp_path / "README.md"
        md_file.write_text("# Test")
        
        binary_file = tmp_path / "test.bin"
        binary_file.write_bytes(b'\x00\x01\x02\x03')
        
        # Should process supported files
        assert processor.should_process_file(py_file)
        assert processor.should_process_file(md_file)
        
        # Should not process binary files
        assert not processor.should_process_file(binary_file)
        
        # Test ignore patterns
        ignore_patterns = ["*.md", "test*"]
        assert not processor.should_process_file(md_file, ignore_patterns)
        assert not processor.should_process_file(py_file, ignore_patterns)
    
    def test_read_file_content(self, tmp_path):
        """Test reading file content."""
        processor = FileProcessor()
        
        # Create test file
        test_file = tmp_path / "test.py"
        content = "def hello():\n    return 'world'"
        test_file.write_text(content)
        
        # Read content
        read_content = processor.read_file_content(test_file)
        
        assert read_content == content
    
    def test_get_file_hash(self, tmp_path):
        """Test file hash generation."""
        processor = FileProcessor()
        
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def test(): pass")
        
        # Get hash
        hash1 = processor.get_file_hash(test_file)
        hash2 = processor.get_file_hash(test_file)
        
        assert hash1 is not None
        assert hash1 == hash2  # Should be consistent
        
        # Modify file and check hash changes
        test_file.write_text("def test(): return 'modified'")
        hash3 = processor.get_file_hash(test_file)
        
        assert hash3 != hash1  # Should change after modification
    
    def test_extract_file_metadata(self, tmp_path):
        """Test extracting file metadata."""
        processor = FileProcessor()
        
        # Create test file
        py_file = tmp_path / "test.py"
        py_file.write_text("def test(): pass")
        
        # Extract metadata
        metadata = processor.extract_file_metadata(py_file)
        
        assert metadata['file_name'] == 'test.py'
        assert metadata['file_extension'] == '.py'
        assert metadata['content_type'] == 'code'
        assert metadata['language'] == 'python'
        assert metadata['is_code'] is True
        assert metadata['is_text'] is True
        assert metadata['is_documentation'] is False
        assert 'file_size' in metadata
        assert 'created_at' in metadata
        assert 'modified_at' in metadata


class TestVectorDatabase:
    """Test VectorDatabase functionality."""
    
    def test_initialization(self, vector_config, mock_chromadb):
        """Test vector database initialization."""
        db = VectorDatabase(vector_config)
        
        assert db.config == vector_config
        assert db._client is not None
        assert db._collection is not None
        assert db._embedding_function is not None
        assert db._chunker is not None
    
    def test_initialization_error(self, vector_config):
        """Test vector database initialization with errors."""
        with patch('src.dev_guard.memory.vector_db.chromadb.Client') as mock_client:
            mock_client.side_effect = Exception("ChromaDB error")
            
            with pytest.raises(VectorDatabaseError, match="Failed to initialize ChromaDB client"):
                VectorDatabase(vector_config)
    
    def test_add_document(self, vector_config, mock_chromadb):
        """Test adding a single document."""
        db = VectorDatabase(vector_config)
        mock_collection = mock_chromadb['collection']
        
        doc = Document(content="Test content", source="test.py")
        
        result = db.add_document(doc)
        
        assert result == doc.id
        mock_collection.add.assert_called_once()
        
        # Check the call arguments
        call_args = mock_collection.add.call_args
        assert call_args[1]['documents'] == [doc.content]
        assert call_args[1]['ids'] == [doc.id]
        assert len(call_args[1]['metadatas']) == 1
        assert call_args[1]['metadatas'][0]['source'] == doc.source
    
    def test_add_document_invalid(self, vector_config, mock_chromadb):
        """Test adding invalid document."""
        db = VectorDatabase(vector_config)
        
        with pytest.raises(VectorDatabaseError, match="Failed to add document"):
            db.add_document("not a document")
    
    def test_add_documents_batch(self, vector_config, mock_chromadb):
        """Test adding multiple documents in batch."""
        db = VectorDatabase(vector_config)
        mock_collection = mock_chromadb['collection']
        
        docs = [
            Document(content="Content 1", source="test1.py"),
            Document(content="Content 2", source="test2.py")
        ]
        
        result = db.add_documents(docs)
        
        assert len(result) == 2
        assert result == [docs[0].id, docs[1].id]
        mock_collection.add.assert_called_once()
        
        # Check batch call arguments
        call_args = mock_collection.add.call_args
        assert len(call_args[1]['documents']) == 2
        assert len(call_args[1]['ids']) == 2
        assert len(call_args[1]['metadatas']) == 2
    
    def test_add_documents_empty(self, vector_config, mock_chromadb):
        """Test adding empty document list."""
        db = VectorDatabase(vector_config)
        
        result = db.add_documents([])
        
        assert result == []
    
    def test_search(self, vector_config, mock_chromadb):
        """Test searching documents."""
        db = VectorDatabase(vector_config)
        mock_collection = mock_chromadb['collection']
        
        # Mock search results
        mock_collection.query.return_value = {
            'ids': [['doc1', 'doc2']],
            'documents': [['Content 1', 'Content 2']],
            'metadatas': [[{'source': 'test1.py'}, {'source': 'test2.py'}]],
            'distances': [[0.1, 0.2]]
        }
        
        results = db.search("test query", n_results=5)
        
        assert len(results) == 2
        assert results[0]['id'] == 'doc1'
        assert results[0]['document'] == 'Content 1'
        assert results[0]['metadata']['source'] == 'test1.py'
        assert results[0]['distance'] == 0.1
        
        mock_collection.query.assert_called_once_with(
            query_texts=['test query'],
            n_results=5,
            where=None
        )
    
    def test_search_empty_query(self, vector_config, mock_chromadb):
        """Test searching with empty query."""
        db = VectorDatabase(vector_config)
        
        with pytest.raises(VectorDatabaseError, match="Failed to search documents"):
            db.search("")
        
        with pytest.raises(VectorDatabaseError, match="Failed to search documents"):
            db.search("   ")
    
    def test_search_code(self, vector_config, mock_chromadb):
        """Test searching code-specific content."""
        db = VectorDatabase(vector_config)
        mock_collection = mock_chromadb['collection']
        
        # Mock search results
        mock_collection.query.return_value = {
            'ids': [['doc1']],
            'documents': [['def test(): pass']],
            'metadatas': [[{'source': 'test.py', 'content_type': 'code'}]],
            'distances': [[0.1]]
        }
        
        results = db.search_code("function", file_extensions=['.py'], n_results=5)
        
        assert len(results) == 1
        mock_collection.query.assert_called_once()
        
        # Check that where condition includes code filter
        call_args = mock_collection.query.call_args
        where_condition = call_args[1]['where']
        assert where_condition['content_type'] == 'code'
        assert where_condition['file_extension'] == {'$in': ['.py']}
    
    def test_get_document(self, vector_config, mock_chromadb):
        """Test getting a specific document by ID."""
        db = VectorDatabase(vector_config)
        mock_collection = mock_chromadb['collection']
        
        # Mock get result
        mock_collection.get.return_value = {
            'ids': ['doc1'],
            'documents': ['Test content'],
            'metadatas': [{'source': 'test.py'}]
        }
        
        result = db.get_document('doc1')
        
        assert result is not None
        assert result['id'] == 'doc1'
        assert result['document'] == 'Test content'
        assert result['metadata']['source'] == 'test.py'
        
        mock_collection.get.assert_called_once_with(
            ids=['doc1'],
            include=['documents', 'metadatas']
        )
    
    def test_get_document_not_found(self, vector_config, mock_chromadb):
        """Test getting a document that doesn't exist."""
        db = VectorDatabase(vector_config)
        mock_collection = mock_chromadb['collection']
        
        # Mock empty result
        mock_collection.get.return_value = {
            'ids': [],
            'documents': [],
            'metadatas': []
        }
        
        result = db.get_document('nonexistent')
        
        assert result is None
    
    def test_update_document(self, vector_config, mock_chromadb):
        """Test updating a document."""
        db = VectorDatabase(vector_config)
        mock_collection = mock_chromadb['collection']
        
        # Mock existing document
        mock_collection.get.return_value = {
            'ids': ['doc1'],
            'documents': ['Old content'],
            'metadatas': [{'source': 'test.py', 'old_key': 'old_value'}]
        }
        
        result = db.update_document(
            'doc1',
            content='New content',
            metadata={'new_key': 'new_value'}
        )
        
        assert result is True
        mock_collection.update.assert_called_once()
        
        # Check update call
        call_args = mock_collection.update.call_args
        assert call_args[1]['ids'] == ['doc1']
        assert call_args[1]['documents'] == ['New content']
        assert 'new_key' in call_args[1]['metadatas'][0]
        assert 'old_key' in call_args[1]['metadatas'][0]  # Should merge metadata
    
    def test_update_document_not_found(self, vector_config, mock_chromadb):
        """Test updating a document that doesn't exist."""
        db = VectorDatabase(vector_config)
        mock_collection = mock_chromadb['collection']
        
        # Mock empty result
        mock_collection.get.return_value = {
            'ids': [],
            'documents': [],
            'metadatas': []
        }
        
        result = db.update_document('nonexistent', content='New content')
        
        assert result is False
        mock_collection.update.assert_not_called()
    
    def test_delete_document(self, vector_config, mock_chromadb):
        """Test deleting a document."""
        db = VectorDatabase(vector_config)
        mock_collection = mock_chromadb['collection']
        
        # Mock existing document
        mock_collection.get.return_value = {
            'ids': ['doc1'],
            'documents': ['Content'],
            'metadatas': [{'source': 'test.py'}]
        }
        
        result = db.delete_document('doc1')
        
        assert result is True
        mock_collection.delete.assert_called_once_with(ids=['doc1'])
    
    def test_delete_document_not_found(self, vector_config, mock_chromadb):
        """Test deleting a document that doesn't exist."""
        db = VectorDatabase(vector_config)
        mock_collection = mock_chromadb['collection']
        
        # Mock empty result
        mock_collection.get.return_value = {
            'ids': [],
            'documents': [],
            'metadatas': []
        }
        
        result = db.delete_document('nonexistent')
        
        assert result is False
        mock_collection.delete.assert_not_called()
    
    def test_delete_documents_by_source(self, vector_config, mock_chromadb):
        """Test deleting all documents from a source."""
        db = VectorDatabase(vector_config)
        mock_collection = mock_chromadb['collection']
        
        # Mock documents from source
        mock_collection.get.return_value = {
            'ids': ['doc1', 'doc2'],
            'documents': ['Content 1', 'Content 2'],
            'metadatas': [{'source': 'test.py'}, {'source': 'test.py'}]
        }
        
        result = db.delete_documents_by_source('test.py')
        
        assert result == 2
        mock_collection.delete.assert_called_once_with(where={'source': 'test.py'})
    
    def test_add_file_content(self, vector_config, mock_chromadb, tmp_path):
        """Test adding file content with automatic chunking."""
        db = VectorDatabase(vector_config)
        mock_collection = mock_chromadb['collection']
        
        # Create a real temporary file for the test
        file_path = tmp_path / 'test.py'
        content = 'def test():\n    return "hello world"'
        file_path.write_text(content)
        
        result = db.add_file_content(file_path, content)
        
        assert len(result) > 0  # Should return document IDs
        mock_collection.add.assert_called_once()
        
        # Check that code-specific metadata was added
        call_args = mock_collection.add.call_args
        metadata = call_args[1]['metadatas'][0]
        assert metadata['file_name'] == 'test.py'
        assert metadata['file_extension'] == '.py'
        assert metadata['is_code'] is True
    
    def test_update_file_content(self, vector_config, mock_chromadb, tmp_path):
        """Test updating file content."""
        db = VectorDatabase(vector_config)
        mock_collection = mock_chromadb['collection']
        
        # Create a real temporary file for the test
        file_path = tmp_path / 'test.py'
        content = 'def new_test():\n    return "new content"'
        file_path.write_text(content)
        
        # Mock existing documents
        mock_collection.get.return_value = {
            'ids': ['doc1'],
            'documents': ['Old content'],
            'metadatas': [{'source': str(file_path)}]
        }
        
        result = db.update_file_content(file_path, content)
        
        assert len(result) > 0
        
        # Should delete old documents and add new ones
        mock_collection.delete.assert_called_once_with(where={'source': str(file_path)})
        mock_collection.add.assert_called_once()
    
    def test_get_collection_stats(self, vector_config, mock_chromadb):
        """Test getting collection statistics."""
        db = VectorDatabase(vector_config)
        mock_collection = mock_chromadb['collection']
        
        # Mock collection data
        mock_collection.count.return_value = 10
        mock_collection.get.return_value = {
            'ids': ['doc1', 'doc2'],
            'metadatas': [
                {'source': 'test1.py', 'content_type': 'code', 'file_extension': '.py'},
                {'source': 'test2.js', 'content_type': 'code', 'file_extension': '.js'}
            ]
        }
        
        stats = db.get_collection_stats()
        
        assert stats['total_documents'] == 10
        assert stats['unique_sources'] == 2
        assert stats['content_types']['code'] == 2
        assert stats['file_extensions']['.py'] == 1
        assert stats['file_extensions']['.js'] == 1
        assert stats['collection_name'] == vector_config.collection_name
        assert stats['embedding_model'] == vector_config.embedding_model
    
    def test_cleanup_old_documents(self, vector_config, mock_chromadb):
        """Test cleaning up old documents."""
        db = VectorDatabase(vector_config)
        mock_collection = mock_chromadb['collection']
        
        # Mock old documents
        old_date = (datetime.now(UTC) - timedelta(days=35)).isoformat()
        mock_collection.get.return_value = {
            'ids': ['doc1', 'doc2'],
            'metadatas': [
                {'created_at': old_date},
                {'created_at': old_date}
            ]
        }
        
        result = db.cleanup_old_documents(days=30)
        
        assert result == 2
        mock_collection.delete.assert_called_once()
        
        # Check that the where condition uses correct date comparison
        call_args = mock_collection.delete.call_args
        where_condition = call_args[1]['where']
        assert 'created_at' in where_condition
        assert '$lt' in where_condition['created_at']
    
    def test_reset_collection(self, vector_config, mock_chromadb):
        """Test resetting the collection."""
        db = VectorDatabase(vector_config)
        mock_client = mock_chromadb['client']
        
        # For reset, we need to mock get_collection to fail first (during reset)
        # then succeed (after recreation)
        mock_client.get_collection.side_effect = [Exception("Collection not found"), mock_chromadb['collection']]
        
        # Reset the mock call count since initialization already called create_collection
        mock_client.create_collection.reset_mock()
        
        db.reset_collection()
        
        # Should delete and recreate collection
        mock_client.delete_collection.assert_called_once_with(vector_config.collection_name)
        # create_collection should be called once in reset (during _initialize_collection)
        assert mock_client.create_collection.call_count == 1
    
    def test_error_handling(self, vector_config, mock_chromadb):
        """Test error handling in vector database operations."""
        db = VectorDatabase(vector_config)
        mock_collection = mock_chromadb['collection']
        
        # Test search error
        mock_collection.query.side_effect = Exception("Search error")
        
        with pytest.raises(VectorDatabaseError, match="Failed to search documents"):
            db.search("test query")
        
        # Test add document error
        mock_collection.add.side_effect = Exception("Add error")
        doc = Document(content="Test", source="test.py")
        
        with pytest.raises(VectorDatabaseError, match="Failed to add document"):
            db.add_document(doc)


if __name__ == "__main__":
    pytest.main([__file__])