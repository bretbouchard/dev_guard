"""Integration tests for vector database file ingestion and search functionality."""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from src.dev_guard.core.config import VectorDBConfig
from src.dev_guard.memory.vector_db import FileProcessor, VectorDatabase


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace with sample files."""
    temp_dir = tempfile.mkdtemp()
    workspace = Path(temp_dir)
    
    # Create sample Python files
    (workspace / "main.py").write_text('''
def main():
    """Main function for the application."""
    print("Hello, World!")
    return 0

if __name__ == "__main__":
    main()
''')
    
    (workspace / "utils.py").write_text('''
import os
import sys

def get_config():
    """Get configuration from environment."""
    return {
        "debug": os.getenv("DEBUG", "false").lower() == "true",
        "port": int(os.getenv("PORT", "8000"))
    }

class Logger:
    """Simple logging utility."""
    
    def __init__(self, name):
        self.name = name
    
    def info(self, message):
        print(f"[INFO] {self.name}: {message}")
    
    def error(self, message):
        print(f"[ERROR] {self.name}: {message}")
''')
    
    # Create sample JavaScript files
    (workspace / "app.js").write_text('''
const express = require('express');
const app = express();

app.get('/', (req, res) => {
    res.json({ message: 'Hello World' });
});

app.listen(3000, () => {
    console.log('Server running on port 3000');
});
''')
    
    # Create sample documentation
    (workspace / "README.md").write_text('''
# Sample Project

This is a sample project for testing vector database functionality.

## Features

- Python backend
- JavaScript frontend
- Comprehensive documentation

## Usage

Run the application with:

```bash
python main.py
```
''')
    
    # Create package.json for project metadata
    package_json = {
        "name": "sample-project",
        "version": "1.0.0",
        "description": "A sample project for testing",
        "main": "app.js",
        "scripts": {
            "start": "node app.js"
        }
    }
    (workspace / "package.json").write_text(json.dumps(package_json, indent=2))
    
    # Create subdirectory with more files
    subdir = workspace / "lib"
    subdir.mkdir()
    
    (subdir / "helpers.py").write_text('''
def format_date(date):
    """Format a date object as string."""
    return date.strftime("%Y-%m-%d")

def parse_json(json_str):
    """Parse JSON string safely."""
    import json
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None
''')
    
    yield workspace
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_vector_db():
    """Create a temporary vector database."""
    temp_dir = tempfile.mkdtemp()
    config = VectorDBConfig(
        path=temp_dir,
        collection_name="test_integration",
        chunk_size=200,
        chunk_overlap=50
    )
    
    # Skip actual ChromaDB initialization for integration tests
    # In real integration tests, you would use a real ChromaDB instance
    try:
        db = VectorDatabase(config)
        yield db
    except Exception as e:
        # If ChromaDB is not available, skip the test
        pytest.skip(f"ChromaDB not available for integration test: {e}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class TestFileProcessor:
    """Test FileProcessor functionality with real files."""
    
    def test_extract_file_metadata(self, temp_workspace):
        """Test extracting metadata from real files."""
        processor = FileProcessor()
        
        # Test Python file metadata
        python_file = temp_workspace / "main.py"
        metadata = processor.extract_file_metadata(python_file)
        
        assert metadata['file_name'] == 'main.py'
        assert metadata['file_extension'] == '.py'
        assert metadata['content_type'] == 'code'
        assert metadata['language'] == 'python'
        assert metadata['is_code'] is True
        assert metadata['is_text'] is True
        assert metadata['is_documentation'] is False
        assert 'file_size' in metadata
        assert 'created_at' in metadata
        assert 'modified_at' in metadata
    
    def test_extract_project_metadata(self, temp_workspace):
        """Test extracting project metadata."""
        processor = FileProcessor()
        
        # Test file in project with package.json
        js_file = temp_workspace / "app.js"
        metadata = processor.extract_file_metadata(js_file)
        
        assert metadata['project_type'] == 'nodejs'
        assert metadata['project_name'] == 'sample-project'
        assert metadata['project_version'] == '1.0.0'
        assert metadata['project_description'] == 'A sample project for testing'
        assert 'project_root' in metadata
    
    def test_should_process_file(self, temp_workspace):
        """Test file processing decision logic."""
        processor = FileProcessor()
        
        # Should process code files
        assert processor.should_process_file(temp_workspace / "main.py")
        assert processor.should_process_file(temp_workspace / "app.js")
        assert processor.should_process_file(temp_workspace / "README.md")
        
        # Should not process package.json (not in supported extensions)
        # This might be debatable - JSON files could be supported
        package_json = temp_workspace / "package.json"
        # The processor should handle JSON files as they have .json extension
        assert processor.should_process_file(package_json)
        
        # Test ignore patterns
        ignore_patterns = ["*.md", "lib/*"]
        assert not processor.should_process_file(temp_workspace / "README.md", ignore_patterns)
        assert not processor.should_process_file(temp_workspace / "lib" / "helpers.py", ignore_patterns)
        assert processor.should_process_file(temp_workspace / "main.py", ignore_patterns)
    
    def test_read_file_content(self, temp_workspace):
        """Test reading file content."""
        processor = FileProcessor()
        
        # Test reading Python file
        python_file = temp_workspace / "main.py"
        content = processor.read_file_content(python_file)
        
        assert content is not None
        assert "def main():" in content
        assert "Hello, World!" in content
        
        # Test reading markdown file
        md_file = temp_workspace / "README.md"
        content = processor.read_file_content(md_file)
        
        assert content is not None
        assert "# Sample Project" in content
        assert "## Features" in content
    
    def test_get_file_hash(self, temp_workspace):
        """Test file hash generation."""
        processor = FileProcessor()
        
        python_file = temp_workspace / "main.py"
        hash1 = processor.get_file_hash(python_file)
        hash2 = processor.get_file_hash(python_file)
        
        assert hash1 is not None
        assert hash1 == hash2  # Should be consistent
        
        # Modify file and check hash changes
        original_content = python_file.read_text()
        python_file.write_text(original_content + "\n# Modified")
        
        hash3 = processor.get_file_hash(python_file)
        assert hash3 != hash1  # Should change after modification


@pytest.mark.integration
class TestVectorDatabaseIntegration:
    """Integration tests for VectorDatabase with real file operations."""
    
    def test_ingest_single_file(self, temp_vector_db, temp_workspace):
        """Test ingesting a single file."""
        python_file = temp_workspace / "main.py"
        
        document_ids = temp_vector_db.ingest_file(python_file)
        
        assert len(document_ids) > 0
        
        # Verify file was ingested
        chunks = temp_vector_db.get_file_chunks(python_file)
        assert len(chunks) > 0
        
        # Check that chunks contain expected content
        all_content = " ".join([chunk['content'] for chunk in chunks])
        assert "def main():" in all_content
        assert "Hello, World!" in all_content
    
    def test_ingest_directory(self, temp_vector_db, temp_workspace):
        """Test ingesting an entire directory."""
        stats = temp_vector_db.ingest_directory(temp_workspace)
        
        assert stats['total_files'] > 0
        assert stats['processed_files'] > 0
        assert stats['total_documents'] > 0
        assert stats['failed_files'] == 0
        
        # Verify specific files were processed
        python_chunks = temp_vector_db.get_file_chunks(temp_workspace / "main.py")
        js_chunks = temp_vector_db.get_file_chunks(temp_workspace / "app.js")
        md_chunks = temp_vector_db.get_file_chunks(temp_workspace / "README.md")
        
        assert len(python_chunks) > 0
        assert len(js_chunks) > 0
        assert len(md_chunks) > 0
    
    def test_search_code_functionality(self, temp_vector_db, temp_workspace):
        """Test code-specific search functionality."""
        # First ingest the files
        temp_vector_db.ingest_directory(temp_workspace)
        
        # Search for Python-specific content
        python_results = temp_vector_db.search_files(
            "def main function",
            file_extensions=['.py']
        )
        
        assert len(python_results) > 0
        
        # Verify results are from Python files
        for result in python_results:
            assert result['file_info']['file_extension'] == '.py'
            assert result['file_info']['is_code'] is True
        
        # Search for JavaScript-specific content
        js_results = temp_vector_db.search_files(
            "express server",
            file_extensions=['.js']
        )
        
        assert len(js_results) > 0
        
        # Verify results are from JavaScript files
        for result in js_results:
            assert result['file_info']['file_extension'] == '.js'
    
    def test_search_by_file_type(self, temp_vector_db, temp_workspace):
        """Test searching by file type."""
        # First ingest the files
        temp_vector_db.ingest_directory(temp_workspace)
        
        # Search for code files
        code_results = temp_vector_db.search_files(
            "function",
            file_types=['code']
        )
        
        assert len(code_results) > 0
        
        # Verify all results are code files
        for result in code_results:
            assert result['file_info']['is_code'] is True
        
        # Search for documentation
        doc_results = temp_vector_db.search_files(
            "project sample",
            file_types=['documentation']
        )
        
        # Should find README.md
        assert len(doc_results) > 0
    
    def test_incremental_updates(self, temp_vector_db, temp_workspace):
        """Test incremental file updates."""
        python_file = temp_workspace / "main.py"
        
        # Initial ingestion
        initial_ids = temp_vector_db.ingest_file(python_file)
        assert len(initial_ids) > 0
        
        # Ingest again without changes - should skip
        second_ids = temp_vector_db.ingest_file(python_file)
        assert len(second_ids) == 0  # Should be skipped
        
        # Modify file and ingest again
        original_content = python_file.read_text()
        modified_content = original_content + '\n\ndef helper():\n    """Helper function."""\n    pass'
        python_file.write_text(modified_content)
        
        updated_ids = temp_vector_db.ingest_file(python_file)
        assert len(updated_ids) > 0
        
        # Verify new content is searchable
        results = temp_vector_db.search("helper function")
        assert len(results) > 0
        
        # Check that old chunks were replaced
        chunks = temp_vector_db.get_file_chunks(python_file)
        all_content = " ".join([chunk['content'] for chunk in chunks])
        assert "def helper():" in all_content
    
    def test_find_similar_files(self, temp_vector_db, temp_workspace):
        """Test finding similar files."""
        # First ingest the files
        temp_vector_db.ingest_directory(temp_workspace)
        
        # Find files similar to main.py
        python_file = temp_workspace / "main.py"
        similar_files = temp_vector_db.find_similar_files(python_file)
        
        # Should find other Python files as similar
        assert len(similar_files) > 0
        
        # Check that similar files are actually different files
        for similar_file in similar_files:
            assert similar_file['file_path'] != str(python_file)
            assert 'similarity_score' in similar_file
            assert 'matching_content' in similar_file
    
    def test_get_files_by_project(self, temp_vector_db, temp_workspace):
        """Test getting files by project."""
        # First ingest the files
        temp_vector_db.ingest_directory(temp_workspace)
        
        # Get files for the sample project
        project_files = temp_vector_db.get_files_by_project("sample-project")
        
        assert len(project_files) > 0
        
        # Verify project files contain expected files
        file_names = [f['file_name'] for f in project_files]
        assert 'main.py' in file_names or 'app.js' in file_names
        
        # Check file information
        for file_info in project_files:
            assert 'file_path' in file_info
            assert 'file_name' in file_info
            assert 'content_type' in file_info
    
    def test_collection_stats_with_real_data(self, temp_vector_db, temp_workspace):
        """Test collection statistics with real ingested data."""
        # First ingest the files
        stats = temp_vector_db.ingest_directory(temp_workspace)
        
        # Get collection statistics
        collection_stats = temp_vector_db.get_collection_stats()
        
        assert collection_stats['total_documents'] == stats['total_documents']
        assert collection_stats['unique_sources'] >= stats['processed_files']
        
        # Check content type distribution
        content_types = collection_stats['content_types']
        assert 'code' in content_types or 'text' in content_types
        
        # Check file extension distribution
        file_extensions = collection_stats['file_extensions']
        assert '.py' in file_extensions
        assert '.js' in file_extensions or '.md' in file_extensions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])