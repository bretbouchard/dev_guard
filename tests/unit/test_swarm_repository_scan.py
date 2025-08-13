"""Unit tests for DevGuardSwarm repository scanning."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.dev_guard.core.config import AgentConfig, Config, RepositoryConfig, VectorDBConfig
from src.dev_guard.core.swarm import DevGuardSwarm


@pytest.fixture
def temp_dirs():
    data_dir = tempfile.mkdtemp()
    repo_dir = tempfile.mkdtemp()

    # Create a small sample repository
    files = [
        "src/main.py",
        "src/utils.py",
        "README.md",
        "ignored.bin",
    ]
    for rel in files:
        p = Path(repo_dir) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == ".bin":
            p.write_bytes(b"\x00\x01\x02")
        else:
            p.write_text(f"# content for {rel}")

    yield data_dir, repo_dir


@pytest.fixture
def vector_config(temp_dirs):
    data_dir, _ = temp_dirs
    return VectorDBConfig(
        path=str(Path(data_dir) / "vector_db"),
        collection_name="test_collection",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=128,
        chunk_overlap=16,
    )


@pytest.fixture
def mock_chromadb():
    """Mock chromadb and embedding functions used by VectorDatabase."""
    with patch("src.dev_guard.memory.vector_db.chromadb") as mock_chroma:
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chroma.Client.return_value = mock_client
        # get_collection fails first, then returns collection (align with code paths)
        mock_client.get_collection.side_effect = [Exception("nope"), mock_collection]
        mock_client.create_collection.return_value = mock_collection

        # Provide sane defaults used by get_collection_stats and search
        mock_collection.count.return_value = 3
        mock_collection.get.return_value = {
            "ids": ["id1", "id2", "id3"],
            "metadatas": [
                {"source": "src/main.py", "content_type": "code", "file_extension": ".py"},
                {"source": "src/utils.py", "content_type": "code", "file_extension": ".py"},
                {"source": "README.md", "content_type": "text", "file_extension": ".md"},
            ],
        }
        mock_collection.query.return_value = {
            "ids": [["doc1"]],
            "documents": [["# content for src/main.py"]],
            "metadatas": [[{"source": "src/main.py", "file_extension": ".py", "content_type": "code"}]],
            "distances": [[0.1]],
        }

        with patch("src.dev_guard.memory.vector_db.embedding_functions") as mock_ef:
            mock_ef.SentenceTransformerEmbeddingFunction.return_value = MagicMock()
            yield {"client": mock_client, "collection": mock_collection}


@pytest.mark.asyncio
async def test_scan_repository_ingests_matching_files(temp_dirs, vector_config, mock_chromadb):
    data_dir, repo_dir = temp_dirs

    # Configure swarm with all commonly referenced agents enabled and patched
    cfg = Config(
        data_dir=data_dir,
        debug=True,
        vector_db=vector_config,
        agents={
            "commander": AgentConfig(enabled=True),
            "planner": AgentConfig(enabled=True),
            "code": AgentConfig(enabled=True),
            "qa_test": AgentConfig(enabled=True),
            "docs": AgentConfig(enabled=True),
            "git_watcher": AgentConfig(enabled=True),
            "impact_mapper": AgentConfig(enabled=True),
            "repo_auditor": AgentConfig(enabled=True),
            "dep_manager": AgentConfig(enabled=True),
        },
        repositories=[
            RepositoryConfig(
                path=repo_dir,
                branch="main",
                watch_files=["*.py", "*.md"],
                ignore_patterns=["*.bin", "__pycache__/*"],
            )
        ],
    )

    with patch("src.dev_guard.agents.commander.CommanderAgent", MagicMock()), \
        patch("src.dev_guard.agents.planner.PlannerAgent", MagicMock()), \
        patch("src.dev_guard.agents.code_agent.CodeAgent", MagicMock()), \
        patch("src.dev_guard.agents.qa_test.QATestAgent", MagicMock()), \
        patch("src.dev_guard.agents.docs.DocsAgent", MagicMock()), \
        patch("src.dev_guard.agents.git_watcher.GitWatcherAgent", MagicMock()), \
        patch("src.dev_guard.agents.impact_mapper.ImpactMapperAgent", MagicMock()), \
        patch("src.dev_guard.agents.repo_auditor.RepoAuditorAgent", MagicMock()), \
        patch("src.dev_guard.agents.dep_manager.DepManagerAgent", MagicMock()):
        swarm = DevGuardSwarm(cfg)

    # Use the repo config we passed into the swarm
    repo_cfg = cfg.repositories[0]

    # Run the scan
    await swarm._scan_repository(Path(repo_dir), repo_cfg)

    # Validate stats
    stats = swarm.vector_db.get_collection_stats()
    assert stats.get("total_documents", 0) > 0

    # Validate we can find a specific file by name
    results = swarm.vector_db.search_files("main.py", n_results=5)
    assert isinstance(results, list)
    # We mocked chroma, so ensure the search call path is exercised but may be empty
    # Instead, check that add() was called on the collection for our text files
    collection = mock_chromadb["collection"]
    assert collection.add.call_count >= 1

    # Ensure the ignored file pattern wasn't ingested
    # No direct API to query by file extension here without real data; assert calls only include non-binary sources
    for call in collection.add.call_args_list:
        metadatas = call.kwargs.get("metadatas") or call[1].get("metadatas")
        if metadatas:
            for md in metadatas:
                assert md.get("file_extension") in {".py", ".md"}

