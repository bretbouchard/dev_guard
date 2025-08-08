"""Vector database system for DevGuard knowledge management."""

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from datetime import datetime, timezone
import hashlib
import json
import re
import os
import mimetypes

import chromadb
from chromadb.utils import embedding_functions
from pydantic import BaseModel, Field, field_validator, ConfigDict
from sentence_transformers import SentenceTransformer

from ..core.config import VectorDBConfig

logger = logging.getLogger(__name__)


class VectorDatabaseError(Exception):
    """Base exception for vector database operations."""
    pass


class FileProcessor:
    """File content processor with metadata extraction and change detection."""
    
    def __init__(self):
        """Initialize file processor."""
        self.logger = logging.getLogger(__name__)
        
        # Supported file extensions for code content
        self.code_extensions = {
            '.py': 'python',
            '.js': 'javascript', 
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.rb': 'ruby',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.r': 'r',
            '.m': 'objective-c',
            '.sh': 'bash',
            '.ps1': 'powershell',
            '.sql': 'sql',
            '.html': 'html',
            '.css': 'css',
            '.scss': 'scss',
            '.less': 'less',
            '.xml': 'xml',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.toml': 'toml',
            '.ini': 'ini',
            '.cfg': 'ini',
            '.conf': 'ini'
        }
        
        # Text file extensions
        self.text_extensions = {
            '.md': 'markdown',
            '.txt': 'text',
            '.rst': 'restructuredtext',
            '.tex': 'latex',
            '.log': 'log',
            '.csv': 'csv'
        }
        
        # Documentation file extensions
        self.doc_extensions = {
            '.md': 'markdown',
            '.rst': 'restructuredtext',
            '.txt': 'text',
            '.tex': 'latex'
        }
    
    def extract_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing file metadata
        """
        try:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            stat = file_path.stat()
            file_extension = file_path.suffix.lower()
            
            # Basic file metadata
            metadata = {
                'file_name': file_path.name,
                'file_path': str(file_path),
                'file_extension': file_extension,
                'file_size': stat.st_size,
                'created_at': datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat(),
                'modified_at': datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                'accessed_at': datetime.fromtimestamp(stat.st_atime, tz=timezone.utc).isoformat(),
            }
            
            # Determine content type and language
            if file_extension in self.code_extensions:
                metadata.update({
                    'content_type': 'code',
                    'language': self.code_extensions[file_extension],
                    'is_code': True,
                    'is_text': True,
                    'is_documentation': False
                })
            elif file_extension in self.text_extensions:
                metadata.update({
                    'content_type': 'text',
                    'language': self.text_extensions[file_extension],
                    'is_code': False,
                    'is_text': True,
                    'is_documentation': file_extension in self.doc_extensions
                })
            else:
                # Try to determine MIME type
                mime_type, _ = mimetypes.guess_type(str(file_path))
                is_text = mime_type and mime_type.startswith('text/')
                
                metadata.update({
                    'content_type': 'binary' if not is_text else 'text',
                    'language': 'unknown',
                    'mime_type': mime_type,
                    'is_code': False,
                    'is_text': is_text,
                    'is_documentation': False
                })
            
            # Add repository context if in a git repo
            git_metadata = self._extract_git_metadata(file_path)
            if git_metadata:
                metadata.update(git_metadata)
            
            # Add project context
            project_metadata = self._extract_project_metadata(file_path)
            if project_metadata:
                metadata.update(project_metadata)
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to extract metadata for {file_path}: {e}")
            return {
                'file_name': file_path.name,
                'file_path': str(file_path),
                'file_extension': file_path.suffix.lower(),
                'error': str(e)
            }
    
    def _extract_git_metadata(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Extract Git repository metadata if file is in a Git repo."""
        try:
            # Find the git repository root
            current_path = file_path.parent
            while current_path != current_path.parent:
                git_dir = current_path / '.git'
                if git_dir.exists():
                    # Found git repository
                    relative_path = file_path.relative_to(current_path)
                    return {
                        'git_repo_root': str(current_path),
                        'git_relative_path': str(relative_path),
                        'in_git_repo': True
                    }
                current_path = current_path.parent
            
            return {'in_git_repo': False}
            
        except Exception as e:
            self.logger.debug(f"Could not extract git metadata for {file_path}: {e}")
            return None
    
    def _extract_project_metadata(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Extract project-specific metadata based on common project files."""
        try:
            metadata = {}
            current_path = file_path.parent
            
            # Look for project indicators
            project_files = {
                'package.json': 'nodejs',
                'pyproject.toml': 'python',
                'requirements.txt': 'python',
                'setup.py': 'python',
                'Cargo.toml': 'rust',
                'go.mod': 'go',
                'pom.xml': 'java',
                'build.gradle': 'java',
                'Gemfile': 'ruby',
                'composer.json': 'php',
                'Dockerfile': 'docker'
            }
            
            # Search up the directory tree for project files
            while current_path != current_path.parent:
                for project_file, project_type in project_files.items():
                    if (current_path / project_file).exists():
                        metadata.update({
                            'project_type': project_type,
                            'project_root': str(current_path),
                            'project_file': project_file
                        })
                        
                        # Extract additional info from specific project files
                        if project_file == 'package.json':
                            package_metadata = self._extract_package_json_metadata(current_path / project_file)
                            if package_metadata:
                                metadata.update(package_metadata)
                        elif project_file == 'pyproject.toml':
                            pyproject_metadata = self._extract_pyproject_metadata(current_path / project_file)
                            if pyproject_metadata:
                                metadata.update(pyproject_metadata)
                        
                        return metadata
                
                current_path = current_path.parent
            
            return metadata if metadata else None
            
        except Exception as e:
            self.logger.debug(f"Could not extract project metadata for {file_path}: {e}")
            return None
    
    def _extract_package_json_metadata(self, package_json_path: Path) -> Optional[Dict[str, Any]]:
        """Extract metadata from package.json file."""
        try:
            with open(package_json_path, 'r', encoding='utf-8') as f:
                package_data = json.load(f)
            
            metadata = {}
            if 'name' in package_data:
                metadata['project_name'] = package_data['name']
            if 'version' in package_data:
                metadata['project_version'] = package_data['version']
            if 'description' in package_data:
                metadata['project_description'] = package_data['description']
            
            return metadata
            
        except Exception as e:
            self.logger.debug(f"Could not parse package.json: {e}")
            return None
    
    def _extract_pyproject_metadata(self, pyproject_path: Path) -> Optional[Dict[str, Any]]:
        """Extract metadata from pyproject.toml file."""
        try:
            import tomllib
            
            with open(pyproject_path, 'rb') as f:
                pyproject_data = tomllib.load(f)
            
            metadata = {}
            if 'project' in pyproject_data:
                project = pyproject_data['project']
                if 'name' in project:
                    metadata['project_name'] = project['name']
                if 'version' in project:
                    metadata['project_version'] = project['version']
                if 'description' in project:
                    metadata['project_description'] = project['description']
            
            return metadata
            
        except Exception as e:
            self.logger.debug(f"Could not parse pyproject.toml: {e}")
            return None
    
    def should_process_file(self, file_path: Path, ignore_patterns: Optional[List[str]] = None) -> bool:
        """
        Determine if a file should be processed based on extension and patterns.
        
        Args:
            file_path: Path to the file
            ignore_patterns: List of glob patterns to ignore
            
        Returns:
            True if file should be processed
        """
        try:
            if not file_path.exists() or not file_path.is_file():
                return False
            
            # Check file size (skip very large files)
            if file_path.stat().st_size > 10 * 1024 * 1024:  # 10MB limit
                self.logger.debug(f"Skipping large file: {file_path}")
                return False
            
            # Check if it's a supported file type
            file_extension = file_path.suffix.lower()
            if file_extension not in self.code_extensions and file_extension not in self.text_extensions:
                # Check if it's a text file by MIME type
                mime_type, _ = mimetypes.guess_type(str(file_path))
                if not mime_type or not mime_type.startswith('text/'):
                    return False
            
            # Check ignore patterns
            if ignore_patterns:
                import fnmatch
                file_str = str(file_path)
                for pattern in ignore_patterns:
                    if fnmatch.fnmatch(file_str, pattern) or fnmatch.fnmatch(file_path.name, pattern):
                        self.logger.debug(f"Ignoring file due to pattern {pattern}: {file_path}")
                        return False
            
            # Skip common non-content files
            skip_patterns = [
                '*.pyc', '*.pyo', '*.pyd', '__pycache__',
                '*.class', '*.jar', '*.war',
                '*.o', '*.so', '*.dll', '*.dylib',
                '*.exe', '*.bin',
                '.git', '.svn', '.hg',
                'node_modules', '.venv', 'venv',
                '*.log', '*.tmp', '*.temp',
                '.DS_Store', 'Thumbs.db',
                '*.min.js', '*.min.css'
            ]
            
            import fnmatch
            file_str = str(file_path)
            for pattern in skip_patterns:
                if fnmatch.fnmatch(file_str, pattern) or fnmatch.fnmatch(file_path.name, pattern):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking if file should be processed {file_path}: {e}")
            return False
    
    def read_file_content(self, file_path: Path) -> Optional[str]:
        """
        Read file content with encoding detection and error handling.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content as string or None if failed
        """
        try:
            # Try common encodings
            encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    
                    # Validate that we got reasonable text content
                    if len(content.strip()) == 0:
                        return None
                    
                    # Check for binary content indicators
                    if '\x00' in content:
                        self.logger.debug(f"File appears to be binary: {file_path}")
                        return None
                    
                    return content
                    
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    self.logger.debug(f"Error reading file with {encoding}: {e}")
                    continue
            
            self.logger.warning(f"Could not read file with any encoding: {file_path}")
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to read file {file_path}: {e}")
            return None
    
    def get_file_hash(self, file_path: Path) -> Optional[str]:
        """
        Generate a hash of the file for change detection.
        
        Args:
            file_path: Path to the file
            
        Returns:
            SHA256 hash of file content and metadata
        """
        try:
            if not file_path.exists():
                return None
            
            # Hash file content and modification time
            stat = file_path.stat()
            hash_input = f"{file_path}:{stat.st_size}:{stat.st_mtime}"
            
            # Add content hash for small files
            if stat.st_size < 1024 * 1024:  # 1MB
                content = self.read_file_content(file_path)
                if content:
                    hash_input += f":{hashlib.sha256(content.encode('utf-8')).hexdigest()}"
            
            return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()
            
        except Exception as e:
            self.logger.error(f"Failed to generate file hash for {file_path}: {e}")
            return None


class Document(BaseModel):
    """Document model for vector database storage."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(..., min_length=1, max_length=50000)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source: str = Field(..., min_length=1, max_length=1000)
    chunk_index: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v):
        """Validate content is not empty after stripping."""
        if not v or not v.strip():
            raise ValueError('content cannot be empty')
        return v.strip()
    
    @field_validator('source')
    @classmethod
    def validate_source(cls, v):
        """Validate source path format."""
        if not v or not v.strip():
            raise ValueError('source cannot be empty')
        return v.strip()
    
    @field_validator('metadata')
    @classmethod
    def validate_metadata(cls, v):
        """Validate metadata contains only JSON-serializable values."""
        try:
            json.dumps(v)
        except (TypeError, ValueError) as e:
            raise ValueError(f'metadata must be JSON-serializable: {e}')
        return v
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
    
    def get_content_hash(self) -> str:
        """Generate a hash of the document content for deduplication."""
        return hashlib.sha256(self.content.encode('utf-8')).hexdigest()


class SearchResult:
    """Search result wrapper with attribute access."""
    def __init__(self, data: Dict[str, Any]):
        """Initialize search result from dictionary data."""
        self.id = data.get('id')
        self.content = data.get('document')  # ChromaDB returns 'document'
        self.document = data.get('document')  # Keep both for compatibility
        self.metadata = data.get('metadata', {})
        self.distance = data.get('distance')


class TextChunker:
    """Text chunking utility with overlap handling."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize text chunker with size and overlap parameters."""
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(__name__)
    
    def chunk_text(self, text: str, source: str = "", metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Split text into overlapping chunks and create Document objects.
        
        Args:
            text: Text content to chunk
            source: Source identifier for the text
            metadata: Additional metadata to attach to each chunk
            
        Returns:
            List of Document objects representing text chunks
        """
        if not text or not text.strip():
            return []
        
        if metadata is None:
            metadata = {}
        
        text = text.strip()
        
        # If text is smaller than chunk size, return as single document
        if len(text) <= self.chunk_size:
            return [Document(
                content=text,
                source=source,
                chunk_index=0,
                metadata={**metadata, "total_chunks": 1}
            )]
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If this is not the last chunk, try to break at word boundary
            if end < len(text):
                # Look for the last space within the chunk to avoid breaking words
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk_content = text[start:end].strip()
            
            if chunk_content:  # Only add non-empty chunks
                chunk_metadata = {
                    **metadata,
                    "chunk_index": chunk_index,
                    "start_pos": start,
                    "end_pos": end,
                    "chunk_size": len(chunk_content)
                }
                
                chunks.append(Document(
                    content=chunk_content,
                    source=source,
                    chunk_index=chunk_index,
                    metadata=chunk_metadata
                ))
                
                chunk_index += 1
            
            # Move start position, accounting for overlap
            start = max(start + 1, end - self.chunk_overlap)
        
        # Add total chunks count to all chunks
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)
        
        self.logger.debug(f"Chunked text from {source} into {len(chunks)} chunks")
        return chunks
    
    def chunk_code(self, code: str, source: str = "", language: str = "", metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Chunk code content with language-aware splitting.
        
        Args:
            code: Code content to chunk
            source: Source file path
            language: Programming language for syntax-aware chunking
            metadata: Additional metadata
            
        Returns:
            List of Document objects representing code chunks
        """
        if metadata is None:
            metadata = {}
        
        # Add language-specific metadata
        code_metadata = {
            **metadata,
            "content_type": "code",
            "language": language,
            "file_extension": Path(source).suffix if source else ""
        }
        
        # For code, try to split on function/class boundaries when possible
        if language.lower() in ['python', 'py']:
            return self._chunk_python_code(code, source, code_metadata)
        elif language.lower() in ['javascript', 'js', 'typescript', 'ts']:
            return self._chunk_js_code(code, source, code_metadata)
        else:
            # Fall back to regular text chunking for other languages
            return self.chunk_text(code, source, code_metadata)
    
    def _chunk_python_code(self, code: str, source: str, metadata: Dict[str, Any]) -> List[Document]:
        """Chunk Python code at function/class boundaries when possible."""
        lines = code.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0
        
        for line in lines:
            line_size = len(line) + 1  # +1 for newline
            
            # Check if this line starts a new function or class
            is_def_line = re.match(r'^(class |def |async def )', line.strip())
            
            # If adding this line would exceed chunk size and we have content
            if current_size + line_size > self.chunk_size and current_chunk:
                # If this is a definition line, end the current chunk here
                if is_def_line:
                    chunk_content = '\n'.join(current_chunk).strip()
                    if chunk_content:
                        chunks.append(Document(
                            content=chunk_content,
                            source=source,
                            chunk_index=chunk_index,
                            metadata={**metadata, "chunk_index": chunk_index}
                        ))
                        chunk_index += 1
                    
                    # Start new chunk with this line
                    current_chunk = [line]
                    current_size = line_size
                else:
                    # Add this line to current chunk and then split
                    current_chunk.append(line)
                    chunk_content = '\n'.join(current_chunk).strip()
                    if chunk_content:
                        chunks.append(Document(
                            content=chunk_content,
                            source=source,
                            chunk_index=chunk_index,
                            metadata={**metadata, "chunk_index": chunk_index}
                        ))
                        chunk_index += 1
                    
                    # Start new chunk
                    current_chunk = []
                    current_size = 0
            else:
                current_chunk.append(line)
                current_size += line_size
        
        # Add remaining content
        if current_chunk:
            chunk_content = '\n'.join(current_chunk).strip()
            if chunk_content:
                chunks.append(Document(
                    content=chunk_content,
                    source=source,
                    chunk_index=chunk_index,
                    metadata={**metadata, "chunk_index": chunk_index}
                ))
        
        # If no chunks were created, fall back to regular text chunking
        if not chunks:
            return self.chunk_text(code, source, metadata)
        
        # Add total chunks count
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)
        
        return chunks
    
    def _chunk_js_code(self, code: str, source: str, metadata: Dict[str, Any]) -> List[Document]:
        """Chunk JavaScript/TypeScript code at function boundaries when possible."""
        lines = code.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0
        
        for line in lines:
            line_size = len(line) + 1  # +1 for newline
            
            # Check if this line starts a new function, class, or export
            is_def_line = re.match(r'^(function |class |export |const .* = |let .* = |var .* = )', line.strip())
            
            # If adding this line would exceed chunk size and we have content
            if current_size + line_size > self.chunk_size and current_chunk:
                if is_def_line:
                    chunk_content = '\n'.join(current_chunk).strip()
                    if chunk_content:
                        chunks.append(Document(
                            content=chunk_content,
                            source=source,
                            chunk_index=chunk_index,
                            metadata={**metadata, "chunk_index": chunk_index}
                        ))
                        chunk_index += 1
                    
                    current_chunk = [line]
                    current_size = line_size
                else:
                    current_chunk.append(line)
                    chunk_content = '\n'.join(current_chunk).strip()
                    if chunk_content:
                        chunks.append(Document(
                            content=chunk_content,
                            source=source,
                            chunk_index=chunk_index,
                            metadata={**metadata, "chunk_index": chunk_index}
                        ))
                        chunk_index += 1
                    
                    current_chunk = []
                    current_size = 0
            else:
                current_chunk.append(line)
                current_size += line_size
        
        # Add remaining content
        if current_chunk:
            chunk_content = '\n'.join(current_chunk).strip()
            if chunk_content:
                chunks.append(Document(
                    content=chunk_content,
                    source=source,
                    chunk_index=chunk_index,
                    metadata={**metadata, "chunk_index": chunk_index}
                ))
        
        # If no chunks were created, fall back to regular text chunking
        if not chunks:
            return self.chunk_text(code, source, metadata)
        
        # Add total chunks count
        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)
        
        return chunks


class VectorDatabase:
    """ChromaDB-based vector database for DevGuard knowledge management."""
    
    def __init__(self, config: VectorDBConfig):
        """Initialize vector database with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._client = None
        self._collection = None
        self._embedding_function = None
        self._chunker = TextChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        self._file_processor = FileProcessor()
        
        self._initialize_client()
        self._initialize_embedding_function()
        self._initialize_collection()
    
    def _initialize_client(self) -> None:
        """Initialize ChromaDB client."""
        try:
            # Ensure the database directory exists
            db_path = Path(self.config.path)
            db_path.mkdir(parents=True, exist_ok=True)
            
            # Use PersistentClient for the updated ChromaDB API
            self._client = chromadb.PersistentClient(path=str(db_path))
            self.logger.info(f"ChromaDB PersistentClient initialized with path: {db_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise VectorDatabaseError(f"Failed to initialize ChromaDB client: {e}")
    
    def _initialize_embedding_function(self) -> None:
        """Initialize embedding function with sentence-transformers."""
        try:
            # Use ChromaDB's sentence transformer embedding function
            self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.config.embedding_model
            )
            self.logger.info(f"Embedding function initialized with model: {self.config.embedding_model}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding function: {e}")
            raise VectorDatabaseError(f"Failed to initialize embedding function: {e}")
    
    def _initialize_collection(self) -> None:
        """Initialize or get the ChromaDB collection."""
        try:
            # Try to get existing collection
            try:
                self._collection = self._client.get_collection(
                    name=self.config.collection_name,
                    embedding_function=self._embedding_function
                )
                self.logger.info(f"Retrieved existing collection: {self.config.collection_name}")
            except Exception:
                # Create new collection if it doesn't exist
                self._collection = self._client.create_collection(
                    name=self.config.collection_name,
                    embedding_function=self._embedding_function,
                    metadata={"description": "DevGuard knowledge base"}
                )
                self.logger.info(f"Created new collection: {self.config.collection_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize collection: {e}")
            raise VectorDatabaseError(f"Failed to initialize collection: {e}")
    
    @property
    def collection(self):
        """Get the ChromaDB collection."""
        return self._collection
    
    @property
    def embedding_function(self):
        """Get the embedding function."""
        return self._embedding_function
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using the embedding function."""
        embeddings = self._embedding_function([text])
        return embeddings[0] if embeddings else []
    
    def _chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Chunk text into smaller pieces."""
        if chunk_size is None:
            chunk_size = self.config.chunk_size
        if overlap is None:
            overlap = self.config.chunk_overlap
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk = text[start:end]
            chunks.append(chunk)
            
            if end >= text_len:
                break
                
            start = end - overlap
            
        return chunks
    
    def add_document(self, document: Document) -> str:
        """
        Add a single document to the vector database.
        
        Args:
            document: Document to add
            
        Returns:
            Document ID
        """
        try:
            # Validate document
            if not isinstance(document, Document):
                raise ValueError("document must be a Document instance")
            
            # Prepare metadata for ChromaDB (must be JSON serializable)
            metadata = {
                "source": document.source,
                "chunk_index": document.chunk_index,
                "created_at": document.created_at.isoformat(),
                "content_hash": document.get_content_hash(),
                **document.metadata
            }
            
            # Ensure all metadata values are JSON serializable
            clean_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    clean_metadata[key] = value
                else:
                    clean_metadata[key] = str(value)
            
            # Add to collection
            self._collection.add(
                documents=[document.content],
                metadatas=[clean_metadata],
                ids=[document.id]
            )
            
            self.logger.debug(f"Added document {document.id} from {document.source}")
            return document.id
            
        except Exception as e:
            self.logger.error(f"Failed to add document: {e}")
            raise VectorDatabaseError(f"Failed to add document: {e}")
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add multiple documents to the vector database in batch.
        
        Args:
            documents: List of documents to add
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        try:
            # Prepare batch data
            doc_ids = []
            doc_contents = []
            doc_metadatas = []
            
            for document in documents:
                if not isinstance(document, Document):
                    raise ValueError("All items must be Document instances")
                
                doc_ids.append(document.id)
                doc_contents.append(document.content)
                
                # Prepare metadata
                metadata = {
                    "source": document.source,
                    "chunk_index": document.chunk_index,
                    "created_at": document.created_at.isoformat(),
                    "content_hash": document.get_content_hash(),
                    **document.metadata
                }
                
                # Ensure all metadata values are JSON serializable
                clean_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        clean_metadata[key] = value
                    else:
                        clean_metadata[key] = str(value)
                
                doc_metadatas.append(clean_metadata)
            
            # Add batch to collection
            self._collection.add(
                documents=doc_contents,
                metadatas=doc_metadatas,
                ids=doc_ids
            )
            
            self.logger.info(f"Added {len(documents)} documents to vector database")
            return doc_ids
            
        except Exception as e:
            self.logger.error(f"Failed to add documents: {e}")
            raise VectorDatabaseError(f"Failed to add documents: {e}")
    
    def search(
        self,
        query: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents in the vector database.
        
        Args:
            query: Search query text
            n_results: Maximum number of results to return
            where: Optional metadata filter conditions
            limit: Alternative name for n_results (for compatibility)
            
        Returns:
            List of search results with documents, metadata, and distances
        """
        # Use limit if provided, otherwise use n_results
        if limit is not None:
            n_results = limit
        try:
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")
            
            # Perform similarity search
            results = self._collection.query(
                query_texts=[query.strip()],
                n_results=min(n_results, 100),  # Limit to reasonable maximum
                where=where
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result_data = {
                        'id': results['ids'][0][i],
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if results['distances'] else None
                    }
                    formatted_results.append(SearchResult(result_data))
            
            self.logger.debug(f"Search for '{query}' returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Failed to search documents: {e}")
            raise VectorDatabaseError(f"Failed to search documents: {e}")
    
    def search_by_metadata(self, metadata_filter: Dict[str, Any], n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search documents by metadata criteria only (no text query).
        
        Args:
            metadata_filter: Metadata filter conditions
            n_results: Maximum number of results to return
            
        Returns:
            List of documents matching the metadata filter
        """
        try:
            if not metadata_filter:
                raise ValueError("metadata_filter cannot be empty")
            
            # Use ChromaDB's get method to filter by metadata without text query
            results = self._collection.query(
                query_texts=[""],  # Empty query to get all docs matching metadata
                n_results=n_results,
                where=metadata_filter
            )
            
            # Format results similar to search method
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result_data = {
                        'id': results['ids'][0][i],
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if results['distances'] else None
                    }
                    formatted_results.append(SearchResult(result_data))
            
            self.logger.debug(f"Search by metadata returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Failed to search by metadata: {e}")
            raise VectorDatabaseError(f"Failed to search by metadata: {e}")
    
    def search_code(self, query: str, file_extensions: Optional[List[str]] = None, n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for code-specific content with file extension filtering.
        
        Args:
            query: Search query text
            file_extensions: List of file extensions to filter by (e.g., ['.py', '.js'])
            n_results: Maximum number of results to return
            
        Returns:
            List of search results filtered for code content
        """
        try:
            # Build metadata filter for code content
            where_conditions = {"content_type": "code"}
            
            if file_extensions:
                # ChromaDB uses $in operator for list matching
                where_conditions["file_extension"] = {"$in": file_extensions}
            
            return self.search(query, n_results, where_conditions)
            
        except Exception as e:
            self.logger.error(f"Failed to search code: {e}")
            raise VectorDatabaseError(f"Failed to search code: {e}")
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific document by ID.
        
        Args:
            document_id: Document ID to retrieve
            
        Returns:
            Document data or None if not found
        """
        try:
            results = self._collection.get(
                ids=[document_id],
                include=['documents', 'metadatas']
            )
            
            if results['documents'] and results['documents'][0]:
                result_data = {
                    'id': results['ids'][0],
                    'document': results['documents'][0],
                    'metadata': results['metadatas'][0]
                }
                return SearchResult(result_data)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get document {document_id}: {e}")
            raise VectorDatabaseError(f"Failed to get document: {e}")
    
    def update_document(self, document_id: str, content: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an existing document's content or metadata.
        
        Args:
            document_id: Document ID to update
            content: New content (optional)
            metadata: New metadata (optional)
            
        Returns:
            True if document was updated, False if not found
        """
        try:
            # Check if document exists
            existing = self.get_document(document_id)
            if not existing:
                return False
            
            # Prepare update data
            update_data = {}
            
            if content is not None:
                update_data['documents'] = [content]
            
            if metadata is not None:
                # Merge with existing metadata
                current_metadata = existing['metadata']
                updated_metadata = {**current_metadata, **metadata}
                
                # Ensure JSON serializable
                clean_metadata = {}
                for key, value in updated_metadata.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        clean_metadata[key] = value
                    else:
                        clean_metadata[key] = str(value)
                
                update_data['metadatas'] = [clean_metadata]
            
            if update_data:
                self._collection.update(
                    ids=[document_id],
                    **update_data
                )
                
                self.logger.debug(f"Updated document {document_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to update document {document_id}: {e}")
            raise VectorDatabaseError(f"Failed to update document: {e}")
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the vector database.
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            True if document was deleted, False if not found
        """
        try:
            # Check if document exists first
            existing = self.get_document(document_id)
            if not existing:
                return False
            
            self._collection.delete(ids=[document_id])
            self.logger.debug(f"Deleted document {document_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete document {document_id}: {e}")
            raise VectorDatabaseError(f"Failed to delete document: {e}")
    
    def delete_documents_by_source(self, source: str) -> int:
        """
        Delete all documents from a specific source.
        
        Args:
            source: Source identifier to delete documents for
            
        Returns:
            Number of documents deleted
        """
        try:
            # Get all documents from this source
            results = self._collection.get(
                where={"source": source},
                include=['documents', 'metadatas']
            )
            
            if not results['ids']:
                return 0
            
            # Delete all documents from this source
            self._collection.delete(where={"source": source})
            
            deleted_count = len(results['ids'])
            self.logger.info(f"Deleted {deleted_count} documents from source: {source}")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to delete documents from source {source}: {e}")
            raise VectorDatabaseError(f"Failed to delete documents from source: {e}")
    
    def add_file_content(self, file_path: Path, content: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Add file content to the vector database with automatic chunking and metadata extraction.
        
        Args:
            file_path: Path to the source file
            content: File content to add (if None, will read from file)
            metadata: Additional metadata for the file
            
        Returns:
            List of document IDs created
        """
        try:
            if metadata is None:
                metadata = {}
            
            # Read content if not provided
            if content is None:
                content = self._file_processor.read_file_content(file_path)
                if content is None:
                    self.logger.warning(f"Could not read content from file: {file_path}")
                    return []
            
            # Extract comprehensive file metadata
            file_metadata = self._file_processor.extract_file_metadata(file_path)
            
            # Merge with provided metadata
            combined_metadata = {**file_metadata, **metadata}
            
            # Add content hash for change detection
            combined_metadata['content_hash'] = hashlib.sha256(content.encode('utf-8')).hexdigest()
            combined_metadata['file_hash'] = self._file_processor.get_file_hash(file_path)
            combined_metadata['ingestion_timestamp'] = datetime.now(timezone.utc).isoformat()
            
            # Chunk the content appropriately
            if combined_metadata.get('is_code', False):
                language = combined_metadata.get('language', 'text')
                documents = self._chunker.chunk_code(
                    content, 
                    str(file_path), 
                    language, 
                    combined_metadata
                )
            else:
                documents = self._chunker.chunk_text(
                    content, 
                    str(file_path), 
                    combined_metadata
                )
            
            # Add documents to vector database
            if documents:
                document_ids = self.add_documents(documents)
                self.logger.info(f"Added {len(documents)} chunks for file: {file_path}")
                return document_ids
            else:
                self.logger.warning(f"No chunks created for file: {file_path}")
                return []
                
        except Exception as e:
            self.logger.error(f"Failed to add file content for {file_path}: {e}")
            raise VectorDatabaseError(f"Failed to add file content: {e}")
    
    def ingest_file(self, file_path: Path, ignore_patterns: Optional[List[str]] = None, force_update: bool = False) -> List[str]:
        """
        Ingest a single file with change detection and metadata extraction.
        
        Args:
            file_path: Path to the file to ingest
            ignore_patterns: List of glob patterns to ignore
            force_update: Force update even if file hasn't changed
            
        Returns:
            List of document IDs created
        """
        try:
            if not self._file_processor.should_process_file(file_path, ignore_patterns):
                self.logger.debug(f"Skipping file: {file_path}")
                return []
            
            # Check if file has changed since last ingestion
            if not force_update:
                current_hash = self._file_processor.get_file_hash(file_path)
                if current_hash and self._is_file_up_to_date(file_path, current_hash):
                    self.logger.debug(f"File unchanged, skipping: {file_path}")
                    return []
            
            # Read and process file content
            content = self._file_processor.read_file_content(file_path)
            if content is None:
                return []
            
            # Update file content (this will delete old chunks and add new ones)
            return self.update_file_content(file_path, content)
            
        except Exception as e:
            self.logger.error(f"Failed to ingest file {file_path}: {e}")
            raise VectorDatabaseError(f"Failed to ingest file: {e}")
    
    def ingest_directory(self, directory_path: Path, ignore_patterns: Optional[List[str]] = None, 
                        recursive: bool = True, max_files: Optional[int] = None) -> Dict[str, Any]:
        """
        Ingest all files in a directory with progress tracking.
        
        Args:
            directory_path: Path to the directory to ingest
            ignore_patterns: List of glob patterns to ignore
            recursive: Whether to process subdirectories
            max_files: Maximum number of files to process
            
        Returns:
            Dictionary with ingestion statistics
        """
        try:
            if not directory_path.exists() or not directory_path.is_dir():
                raise ValueError(f"Directory does not exist: {directory_path}")
            
            stats = {
                'total_files': 0,
                'processed_files': 0,
                'skipped_files': 0,
                'failed_files': 0,
                'total_documents': 0,
                'errors': []
            }
            
            # Get all files to process
            if recursive:
                files = list(directory_path.rglob('*'))
            else:
                files = list(directory_path.iterdir())
            
            # Filter to only files
            files = [f for f in files if f.is_file()]
            stats['total_files'] = len(files)
            
            # Limit number of files if specified
            if max_files and len(files) > max_files:
                files = files[:max_files]
                self.logger.info(f"Limited processing to {max_files} files")
            
            # Process each file
            for file_path in files:
                try:
                    document_ids = self.ingest_file(file_path, ignore_patterns)
                    if document_ids:
                        stats['processed_files'] += 1
                        stats['total_documents'] += len(document_ids)
                    else:
                        stats['skipped_files'] += 1
                        
                except Exception as e:
                    stats['failed_files'] += 1
                    error_msg = f"Failed to process {file_path}: {e}"
                    stats['errors'].append(error_msg)
                    self.logger.error(error_msg)
            
            self.logger.info(f"Directory ingestion complete: {stats}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to ingest directory {directory_path}: {e}")
            raise VectorDatabaseError(f"Failed to ingest directory: {e}")
    
    def _is_file_up_to_date(self, file_path: Path, current_hash: str) -> bool:
        """
        Check if a file is up to date in the vector database.
        
        Args:
            file_path: Path to the file
            current_hash: Current file hash
            
        Returns:
            True if file is up to date
        """
        try:
            # Search for existing documents from this file
            results = self._collection.get(
                where={"source": str(file_path)},
                include=['metadatas'],
                limit=1
            )
            
            if not results['metadatas']:
                return False
            
            # Check if file hash matches
            stored_hash = results['metadatas'][0].get('file_hash')
            return stored_hash == current_hash
            
        except Exception as e:
            self.logger.debug(f"Error checking file status for {file_path}: {e}")
            return False
    
    def search_files(self, query: str, file_extensions: Optional[List[str]] = None, 
                    file_types: Optional[List[str]] = None, n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for files with enhanced filtering options.
        
        Args:
            query: Search query text
            file_extensions: List of file extensions to filter by (e.g., ['.py', '.js'])
            file_types: List of file types to filter by (e.g., ['code', 'text', 'documentation'])
            n_results: Maximum number of results to return
            
        Returns:
            List of search results with file information
        """
        try:
            # Build metadata filter
            where_conditions = {}
            
            if file_extensions:
                where_conditions["file_extension"] = {"$in": file_extensions}
            
            if file_types:
                # Map file types to metadata conditions
                type_conditions = []
                for file_type in file_types:
                    if file_type == 'code':
                        type_conditions.append({"is_code": True})
                    elif file_type == 'documentation':
                        type_conditions.append({"is_documentation": True})
                    elif file_type == 'text':
                        type_conditions.append({"is_text": True})
                
                if len(type_conditions) == 1:
                    where_conditions.update(type_conditions[0])
                elif len(type_conditions) > 1:
                    where_conditions["$or"] = type_conditions
            
            # Perform search
            results = self.search(query, n_results, where_conditions if where_conditions else None)
            
            # Enhance results with file information
            enhanced_results = []
            for result in results:
                metadata = result['metadata']
                enhanced_result = {
                    **result,
                    'file_info': {
                        'file_name': metadata.get('file_name'),
                        'file_path': metadata.get('file_path'),
                        'file_extension': metadata.get('file_extension'),
                        'language': metadata.get('language'),
                        'content_type': metadata.get('content_type'),
                        'is_code': metadata.get('is_code', False),
                        'project_type': metadata.get('project_type'),
                        'project_name': metadata.get('project_name')
                    }
                }
                enhanced_results.append(enhanced_result)
            
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"Failed to search files: {e}")
            raise VectorDatabaseError(f"Failed to search files: {e}")
    
    def get_file_chunks(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of document chunks for the file
        """
        try:
            results = self._collection.get(
                where={"source": str(file_path)},
                include=['documents', 'metadatas']
            )
            
            chunks = []
            if results['documents']:
                for i, doc in enumerate(results['documents']):
                    chunk = {
                        'id': results['ids'][i],
                        'content': doc,
                        'metadata': results['metadatas'][i],
                        'chunk_index': results['metadatas'][i].get('chunk_index', 0)
                    }
                    chunks.append(chunk)
            
            # Sort by chunk index
            chunks.sort(key=lambda x: x['chunk_index'])
            return chunks
            
        except Exception as e:
            self.logger.error(f"Failed to get file chunks for {file_path}: {e}")
            raise VectorDatabaseError(f"Failed to get file chunks: {e}")
    
    def get_files_by_project(self, project_name: str) -> List[Dict[str, Any]]:
        """
        Get all files belonging to a specific project.
        
        Args:
            project_name: Name of the project
            
        Returns:
            List of unique files in the project
        """
        try:
            results = self._collection.get(
                where={"project_name": project_name},
                include=['metadatas']
            )
            
            # Get unique files
            files = {}
            for metadata in results['metadatas']:
                file_path = metadata.get('file_path')
                if file_path and file_path not in files:
                    files[file_path] = {
                        'file_path': file_path,
                        'file_name': metadata.get('file_name'),
                        'file_extension': metadata.get('file_extension'),
                        'language': metadata.get('language'),
                        'content_type': metadata.get('content_type'),
                        'project_type': metadata.get('project_type'),
                        'modified_at': metadata.get('modified_at')
                    }
            
            return list(files.values())
            
        except Exception as e:
            self.logger.error(f"Failed to get files for project {project_name}: {e}")
            raise VectorDatabaseError(f"Failed to get files for project: {e}")
    
    def find_similar_files(self, file_path: Path, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Find files similar to the given file based on content.
        
        Args:
            file_path: Path to the reference file
            n_results: Maximum number of similar files to return
            
        Returns:
            List of similar files with similarity scores
        """
        try:
            # Get content from the file
            chunks = self.get_file_chunks(file_path)
            if not chunks:
                return []
            
            # Use the first chunk as the query (could be enhanced to use multiple chunks)
            query_content = chunks[0]['content']
            
            # Search for similar content, excluding the same file
            results = self.search(query_content, n_results * 2)  # Get more results to filter
            
            # Filter out chunks from the same file and group by file
            similar_files = {}
            for result in results:
                result_file_path = result['metadata'].get('file_path')
                if result_file_path and result_file_path != str(file_path):
                    if result_file_path not in similar_files:
                        similar_files[result_file_path] = {
                            'file_path': result_file_path,
                            'file_name': result['metadata'].get('file_name'),
                            'language': result['metadata'].get('language'),
                            'content_type': result['metadata'].get('content_type'),
                            'similarity_score': result['distance'],
                            'matching_content': result['document'][:200] + '...' if len(result['document']) > 200 else result['document']
                        }
                    else:
                        # Keep the best similarity score
                        if result['distance'] < similar_files[result_file_path]['similarity_score']:
                            similar_files[result_file_path]['similarity_score'] = result['distance']
                            similar_files[result_file_path]['matching_content'] = result['document'][:200] + '...' if len(result['document']) > 200 else result['document']
            
            # Sort by similarity score and limit results
            similar_files_list = list(similar_files.values())
            similar_files_list.sort(key=lambda x: x['similarity_score'])
            
            return similar_files_list[:n_results]
            
        except Exception as e:
            self.logger.error(f"Failed to find similar files for {file_path}: {e}")
            raise VectorDatabaseError(f"Failed to find similar files: {e}")
    
    def update_file_content(self, file_path: Path, content: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Update file content in the vector database by replacing existing chunks.
        
        Args:
            file_path: Path to the source file
            content: New file content
            metadata: Additional metadata for the file
            
        Returns:
            List of new document IDs created
        """
        try:
            # Delete existing documents for this file
            deleted_count = self.delete_documents_by_source(str(file_path))
            self.logger.debug(f"Deleted {deleted_count} existing chunks for {file_path}")
            
            # Add new content
            return self.add_file_content(file_path, content, metadata)
            
        except Exception as e:
            self.logger.error(f"Failed to update file content for {file_path}: {e}")
            raise VectorDatabaseError(f"Failed to update file content: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            # Get collection count
            count_result = self._collection.count()
            
            # Get sample of documents to analyze
            sample_results = self._collection.get(
                limit=min(100, count_result),
                include=['metadatas']
            )
            
            # Analyze metadata
            sources = set()
            content_types = {}
            file_extensions = {}
            
            if sample_results['metadatas']:
                for metadata in sample_results['metadatas']:
                    if 'source' in metadata:
                        sources.add(metadata['source'])
                    
                    content_type = metadata.get('content_type', 'text')
                    content_types[content_type] = content_types.get(content_type, 0) + 1
                    
                    file_ext = metadata.get('file_extension', 'unknown')
                    file_extensions[file_ext] = file_extensions.get(file_ext, 0) + 1
            
            return {
                'total_documents': count_result,
                'document_count': count_result,  # Alias for compatibility
                'unique_sources': len(sources),
                'content_types': content_types,
                'file_extensions': file_extensions,
                'collection_name': self.config.collection_name,
                'embedding_model': self.config.embedding_model
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            raise VectorDatabaseError(f"Failed to get collection stats: {e}")
    
    def cleanup_old_documents(self, days: int = 30) -> int:
        """
        Remove documents older than specified days.
        
        Args:
            days: Number of days to keep documents
            
        Returns:
            Number of documents deleted
        """
        try:
            from datetime import timedelta
            
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            cutoff_iso = cutoff_date.isoformat()
            
            # Get documents older than cutoff
            old_docs = self._collection.get(
                where={"created_at": {"$lt": cutoff_iso}},
                include=['metadatas']
            )
            
            if not old_docs['ids']:
                return 0
            
            # Delete old documents
            self._collection.delete(
                where={"created_at": {"$lt": cutoff_iso}}
            )
            
            deleted_count = len(old_docs['ids'])
            self.logger.info(f"Cleaned up {deleted_count} documents older than {days} days")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old documents: {e}")
            raise VectorDatabaseError(f"Failed to cleanup old documents: {e}")
    
    def reset_collection(self) -> None:
        """Reset the collection by deleting all documents."""
        try:
            # Delete the collection
            self._client.delete_collection(self.config.collection_name)
            
            # Recreate the collection
            self._initialize_collection()
            
            self.logger.info(f"Reset collection: {self.config.collection_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to reset collection: {e}")
            raise VectorDatabaseError(f"Failed to reset collection: {e}")
    
    def reset_collection(self) -> None:
        """Reset the collection by deleting all documents."""
        try:
            # Delete the collection
            self._client.delete_collection(self.config.collection_name)
            
            # Recreate the collection
            self._initialize_collection()
            
            self.logger.info(f"Reset collection: {self.config.collection_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to reset collection: {e}")
            raise VectorDatabaseError(f"Failed to reset collection: {e}")