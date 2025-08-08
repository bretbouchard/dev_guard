"""
Quick debug script to check QA agent capabilities.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dev_guard.agents.qa_test import QATestAgent


class MockConfig:
    def get_agent_config(self, agent_type):
        return {"max_retries": 3, "timeout": 300}


class MockSharedMemory:
    def update_agent_state(self, *args, **kwargs):
        pass
    
    def add_memory(self, *args, **kwargs):
        return "test-memory-id"


class MockVectorDB:
    pass


if __name__ == "__main__":
    config = MockConfig()
    memory = MockSharedMemory()
    vector_db = MockVectorDB()
    
    qa_agent = QATestAgent(
        agent_id="test_qa_agent",
        config=config,
        shared_memory=memory,
        vector_db=vector_db
    )
    
    capabilities = qa_agent.get_capabilities()
    print("Available capabilities:")
    for cap in sorted(capabilities):
        print(f"  - {cap}")
    
    print(f"\nTotal capabilities: {len(capabilities)}")
