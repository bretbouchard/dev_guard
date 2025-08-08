"""
Unit tests for Task 3.3: GoosePatch and AST metadata functionality in shared memory.

Tests the enhanced MemoryEntry model with goose_patch, ast_summary, and goose_strategy fields,
and verifies Goose tool output logging and AST metadata linking.
"""

import os
import tempfile
from typing import Any

import pytest

from dev_guard.memory.shared_memory import MemoryEntry, SharedMemory


class TestGoosePatchMemory:
    """Test GoosePatch and AST metadata functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    @pytest.fixture
    def shared_memory(self, temp_db):
        """Create SharedMemory instance with temporary database."""
        return SharedMemory(db_path=temp_db)
    
    @pytest.fixture
    def sample_goose_patch(self) -> dict[str, Any]:
        """Sample Goose patch data."""
        return {
            "operation": "refactor_function",
            "source_file": "test.py",
            "target_function": "calculate_sum",
            "changes": [
                {"type": "parameter_rename", "from": "x", "to": "first_number"},
                {"type": "add_docstring", "content": "Calculate sum of two numbers"}
            ],
            "confidence": 0.95,
            "tool_version": "goose-v1.2.0"
        }
    
    @pytest.fixture
    def sample_ast_summary(self) -> dict[str, Any]:
        """Sample AST analysis data."""
        return {
            "functions": ["calculate_sum", "validate_input"],
            "classes": [],
            "imports": ["typing"],
            "complexity_score": 3,
            "lines_of_code": 25,
            "potential_issues": ["missing_docstring", "parameter_naming"],
            "refactor_suggestions": ["add_type_hints", "extract_validation"]
        }

    def test_enhanced_memory_entry_with_goose_fields(self, sample_goose_patch, sample_ast_summary):
        """Test MemoryEntry with new GoosePatch and AST fields."""
        entry = MemoryEntry(
            agent_id="code_agent",
            type="result",
            content={"action": "code_refactored", "status": "success"},
            tags={"goose", "refactor"},
            goose_patch=sample_goose_patch,
            ast_summary=sample_ast_summary,
            goose_strategy="function_modernization",
            file_path="/path/to/test.py"
        )
        
        # Verify all fields are set correctly
        assert entry.goose_patch == sample_goose_patch
        assert entry.ast_summary == sample_ast_summary
        assert entry.goose_strategy == "function_modernization"
        assert entry.file_path == "/path/to/test.py"
        assert "goose" in entry.tags
        assert "refactor" in entry.tags

    def test_memory_entry_optional_goose_fields(self):
        """Test MemoryEntry with optional GoosePatch fields as None."""
        entry = MemoryEntry(
            agent_id="planner",
            type="observation",
            content={"analysis": "code_review_needed"},
            goose_patch=None,
            ast_summary=None,
            goose_strategy=None,
            file_path=None
        )
        
        # Verify None values are handled correctly
        assert entry.goose_patch is None
        assert entry.ast_summary is None
        assert entry.goose_strategy is None
        assert entry.file_path is None

    def test_add_memory_with_goose_fields(self, shared_memory, sample_goose_patch, sample_ast_summary):
        """Test adding memory entry with GoosePatch fields to database."""
        entry = MemoryEntry(
            agent_id="code_agent",
            type="result",
            content={"action": "goose_patch_applied", "file": "calculator.py"},
            tags={"goose", "patch"},
            goose_patch=sample_goose_patch,
            ast_summary=sample_ast_summary,
            goose_strategy="function_extraction",
            file_path="/src/calculator.py"
        )
        
        # Add memory entry
        entry_id = shared_memory.add_memory(entry)
        assert entry_id == entry.id
        
        # Retrieve and verify
        retrieved_entry = shared_memory.get_memory_by_id(entry_id)
        assert retrieved_entry is not None
        assert retrieved_entry.goose_patch == sample_goose_patch
        assert retrieved_entry.ast_summary == sample_ast_summary
        assert retrieved_entry.goose_strategy == "function_extraction"
        assert retrieved_entry.file_path == "/src/calculator.py"

    def test_log_goose_patch_memory_helper(self, shared_memory, sample_goose_patch, sample_ast_summary):
        """Test log_goose_patch_memory helper method."""
        entry_id = shared_memory.log_goose_patch_memory(
            agent_id="code_agent",
            goose_patch=sample_goose_patch,
            ast_summary=sample_ast_summary,
            goose_strategy="refactor_for_readability",
            file_path="/src/utils.py"
        )
        
        # Retrieve and verify entry
        entry = shared_memory.get_memory_by_id(entry_id)
        assert entry is not None
        assert entry.type == "result"
        assert entry.goose_patch == sample_goose_patch
        assert entry.ast_summary == sample_ast_summary
        assert entry.goose_strategy == "refactor_for_readability"
        assert entry.file_path == "/src/utils.py"
        assert "goose" in entry.tags
        assert "patch" in entry.tags
        assert "refactor" in entry.tags

    def test_log_ast_analysis_memory_helper(self, shared_memory, sample_ast_summary):
        """Test log_ast_analysis_memory helper method."""
        file_path = "/src/models/user.py"
        entry_id = shared_memory.log_ast_analysis_memory(
            agent_id="planner",
            file_path=file_path,
            ast_summary=sample_ast_summary,
            refactor_strategy="class_decomposition"
        )
        
        # Retrieve and verify entry
        entry = shared_memory.get_memory_by_id(entry_id)
        assert entry is not None
        assert entry.type == "observation"
        assert entry.ast_summary == sample_ast_summary
        assert entry.goose_strategy == "class_decomposition"
        assert entry.file_path == file_path
        assert "ast" in entry.tags
        assert "analysis" in entry.tags
        assert "refactor" in entry.tags

    def test_get_goose_patches_for_file(self, shared_memory, sample_goose_patch):
        """Test retrieving Goose patches for a specific file."""
        file_path = "/src/calculator.py"
        
        # Add multiple patches for the same file
        patch1_id = shared_memory.log_goose_patch_memory(
            agent_id="code_agent",
            goose_patch={**sample_goose_patch, "version": "1"},
            goose_strategy="initial_refactor",
            file_path=file_path
        )
        
        patch2_id = shared_memory.log_goose_patch_memory(
            agent_id="code_agent", 
            goose_patch={**sample_goose_patch, "version": "2"},
            goose_strategy="optimization",
            file_path=file_path
        )
        
        # Add patch for different file (should not be retrieved)
        shared_memory.log_goose_patch_memory(
            agent_id="code_agent",
            goose_patch=sample_goose_patch,
            file_path="/src/other.py"
        )
        
        # Retrieve patches for target file
        patches = shared_memory.get_goose_patches_for_file(file_path)
        
        # Verify results
        assert len(patches) == 2
        patch_ids = {p.id for p in patches}
        assert patch1_id in patch_ids
        assert patch2_id in patch_ids
        
        # Verify all patches are for correct file
        for patch in patches:
            assert patch.file_path == file_path
            assert patch.goose_patch is not None

    def test_get_ast_summaries_by_strategy(self, shared_memory, sample_ast_summary):
        """Test retrieving AST summaries by Goose strategy."""
        strategy = "function_extraction"
        
        # Add multiple AST analyses with same strategy
        analysis1_id = shared_memory.log_ast_analysis_memory(
            agent_id="planner",
            file_path="/src/file1.py",
            ast_summary={**sample_ast_summary, "file": "file1.py"},
            refactor_strategy=strategy
        )
        
        analysis2_id = shared_memory.log_ast_analysis_memory(
            agent_id="planner",
            file_path="/src/file2.py", 
            ast_summary={**sample_ast_summary, "file": "file2.py"},
            refactor_strategy=strategy
        )
        
        # Add analysis with different strategy (should not be retrieved)
        shared_memory.log_ast_analysis_memory(
            agent_id="planner",
            file_path="/src/file3.py",
            ast_summary=sample_ast_summary,
            refactor_strategy="class_refactoring"
        )
        
        # Retrieve analyses for target strategy
        analyses = shared_memory.get_ast_summaries_by_strategy(strategy)
        
        # Verify results
        assert len(analyses) == 2
        analysis_ids = {a.id for a in analyses}
        assert analysis1_id in analysis_ids
        assert analysis2_id in analysis_ids
        
        # Verify all analyses use correct strategy
        for analysis in analyses:
            assert analysis.goose_strategy == strategy
            assert analysis.ast_summary is not None

    def test_goose_field_validation(self):
        """Test validation of GoosePatch and AST fields."""
        # Test that empty strings pass through unchanged (falsy values allowed)
        entry_strategy = MemoryEntry(
            agent_id="test",
            type="result",
            content={"test": "data"},
            parent_id=None,
            goose_patch=None,
            ast_summary=None,
            goose_strategy="",  # Empty string is allowed
            file_path=None
        )
        assert entry_strategy.goose_strategy == ""
        
        # Test that empty file_path strings pass through unchanged
        entry_path = MemoryEntry(
            agent_id="test",
            type="result", 
            content={"test": "data"},
            parent_id=None,
            goose_patch=None,
            ast_summary=None,
            goose_strategy=None,
            file_path=""  # Empty string is allowed
        )
        assert entry_path.file_path == ""
        
        # Test that whitespace-only strings raise validation errors
        with pytest.raises(ValueError, match="goose_strategy cannot be empty string"):
            MemoryEntry(
                agent_id="test",
                type="result",
                content={"test": "data"},
                parent_id=None,
                goose_patch=None,
                ast_summary=None,
                goose_strategy="   ",  # Whitespace-only should be invalid
                file_path=None
            )
        
        with pytest.raises(ValueError, match="file_path cannot be empty string"):
            MemoryEntry(
                agent_id="test",
                type="result",
                content={"test": "data"},
                parent_id=None,
                goose_patch=None,
                ast_summary=None,
                goose_strategy=None,
                file_path="  \t  "  # Whitespace-only should be invalid
            )
        
        # Test that valid values are preserved
        entry_valid = MemoryEntry(
            agent_id="test",
            type="result",
            content={"test": "data"},
            parent_id=None,
            goose_patch=None,
            ast_summary=None,
            goose_strategy="function_refactor",
            file_path="/path/to/file.py"
        )
        assert entry_valid.goose_strategy == "function_refactor"
        assert entry_valid.file_path == "/path/to/file.py"

    def test_memory_search_with_goose_tags(self, shared_memory, sample_goose_patch):
        """Test memory search functionality with Goose-related tags."""
        # Add memory with goose tags
        shared_memory.log_goose_patch_memory(
            agent_id="code_agent",
            goose_patch=sample_goose_patch,
            goose_strategy="modernization"
        )
        
        # Add regular memory without goose tags
        regular_entry = MemoryEntry(
            agent_id="qa_agent",
            type="observation",
            content={"test": "regular_observation"},
            parent_id=None,
            goose_patch=None,
            ast_summary=None,
            goose_strategy=None,
            file_path=None
        )
        shared_memory.add_memory(regular_entry)
        
        # Search by goose tag
        goose_memories = shared_memory.get_memory_by_tags(tags={"goose"})
        assert len(goose_memories) == 1
        assert goose_memories[0].goose_patch == sample_goose_patch
        
        # Search by multiple goose tags
        refactor_memories = shared_memory.get_memory_by_tags(
            tags={"goose", "refactor"}, 
            match_all=True
        )
        assert len(refactor_memories) == 1

    def test_database_schema_migration(self, shared_memory):
        """Test that database schema includes new GoosePatch columns."""
        # This test verifies the schema migration worked correctly
        with shared_memory._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(memory_entries)")
            columns = [col[1] for col in cursor.fetchall()]
            
            # Verify new columns exist
            assert "goose_patch" in columns
            assert "ast_summary" in columns
            assert "goose_strategy" in columns
            assert "file_path" in columns

    def test_linking_memory_to_files_and_strategies(self, shared_memory, sample_goose_patch, sample_ast_summary):
        """Test linking memory entries to original files and refactoring strategies."""
        file_path = "/src/complex_module.py"
        strategy = "extract_methods"
        
        # Log AST analysis
        ast_entry_id = shared_memory.log_ast_analysis_memory(
            agent_id="planner",
            file_path=file_path,
            ast_summary=sample_ast_summary,
            refactor_strategy=strategy
        )
        
        # Log Goose patch with same file and strategy
        patch_entry_id = shared_memory.log_goose_patch_memory(
            agent_id="code_agent",
            goose_patch=sample_goose_patch,
            goose_strategy=strategy,
            file_path=file_path,
            parent_id=ast_entry_id  # Link to AST analysis
        )
        
        # Verify linking through file path
        file_patches = shared_memory.get_goose_patches_for_file(file_path)
        assert len(file_patches) == 1
        assert file_patches[0].id == patch_entry_id
        
        # Verify linking through strategy
        strategy_analyses = shared_memory.get_ast_summaries_by_strategy(strategy)
        assert len(strategy_analyses) == 1
        assert strategy_analyses[0].id == ast_entry_id
        
        # Verify parent-child relationship
        patch_entry = shared_memory.get_memory_by_id(patch_entry_id)
        assert patch_entry.parent_id == ast_entry_id

    def test_auditability_and_traceability(self, shared_memory, sample_goose_patch, sample_ast_summary):
        """Test that Goose operations are fully auditable and traceable."""
        file_path = "/src/payment_processor.py"
        
        # Create audit trail: Analysis -> Planning -> Implementation
        
        # 1. AST Analysis
        analysis_id = shared_memory.log_ast_analysis_memory(
            agent_id="repo_auditor",
            file_path=file_path,
            ast_summary=sample_ast_summary
        )
        
        # 2. Planning decision
        planning_entry = MemoryEntry(
            agent_id="planner",
            type="decision",
            content={"decision": "refactor_recommended", "file": file_path},
            parent_id=analysis_id,
            file_path=file_path
        )
        planning_id = shared_memory.add_memory(planning_entry)
        
        # 3. Goose patch implementation  
        patch_id = shared_memory.log_goose_patch_memory(
            agent_id="code_agent",
            goose_patch=sample_goose_patch,
            goose_strategy="security_improvements",
            file_path=file_path,
            parent_id=planning_id
        )
        
        # Verify full audit trail can be reconstructed
        patch_entry = shared_memory.get_memory_by_id(patch_id)
        planning_entry = shared_memory.get_memory_by_id(patch_entry.parent_id)
        analysis_entry = shared_memory.get_memory_by_id(planning_entry.parent_id)
        
        # Verify chain of traceability
        assert analysis_entry.id == analysis_id
        assert planning_entry.id == planning_id  
        assert patch_entry.id == patch_id
        
        # All entries should link to same file
        assert analysis_entry.file_path == file_path
        assert planning_entry.file_path == file_path
        assert patch_entry.file_path == file_path
