import asyncio
import os
from pathlib import Path

import requests

from src.dev_guard.core.config import Config
from src.dev_guard.memory.shared_memory import SharedMemory
from src.dev_guard.memory.vector_db import VectorDatabase
from src.dev_guard.agents.qa_agent import QATestAgent


def _ollama_available(base_url: str) -> bool:
    try:
        r = requests.get(base_url.rstrip("/") + "/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


async def generate_tests_for(target_files: list[str]) -> dict[str, dict]:
    # Prefer local Ollama per user's preference
    cfg = Config()
    cfg.llm.provider = "ollama"
    cfg.llm.model = "gpt-oss:20b"
    cfg.llm.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    shared_memory = SharedMemory(db_path=str(Path(cfg.shared_memory.db_path)))
    vector_db = VectorDatabase(cfg.vector_db)

    qa = QATestAgent(
        agent_id="qa_test",
        config=cfg,
        shared_memory=shared_memory,
        vector_db=vector_db,
    )

    # If Ollama isn't reachable, force template generation to avoid hanging
    if not _ollama_available(cfg.llm.base_url or "http://localhost:11434"):
        qa.llm_provider = None
        print("[warn] Ollama not reachable; falling back to template test generation.")

    results: dict[str, dict] = {}
    for file in target_files:
        file_path = Path(file)
        if not file_path.exists():
            print(f"[skip] {file} does not exist")
            continue
        print(f"[info] Generating tests for {file}...")
        task = {
            "type": "generate_tests",
            "target_file": str(file_path),
            "test_type": "comprehensive",
            "coverage_target": 80,
            "test_framework": "pytest",
        }
        res = await qa.execute_task(task)
        print(f"[result] {file}: success={res.get('success')} test_file={res.get('test_file')} error={res.get('error')}")
        results[str(file_path)] = res
    return results


def main():
    # Reduce tokenizers fork warning noise
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    targets = [
        "src/dev_guard/core/config.py",
        "src/dev_guard/core/swarm.py",
    ]

    results = asyncio.run(generate_tests_for(targets))
    # Print a concise summary
    print("\nSummary:")
    for f, r in results.items():
        print(f"- {f}: success={r.get('success')} -> {r.get('test_file')}")


if __name__ == "__main__":
    main()

