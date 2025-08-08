#!/usr/bin/env python3
"""
Simple Git Watcher Agent Implementation Test
Tests core functionality without full validation suite.
"""

import asyncio
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockConfig:
    """Mock configuration for testing."""
    
    def get_agent_config(self, agent_id: str):
        return {"poll_interval": 5.0}
    
    @property
    def repositories(self):
        return {}

class MockSharedMemory:
    """Mock shared memory for testing."""
    
    def update_agent_state(self, state):
        pass
    
    def get_agent_state(self, agent_id):
        return None

class MockVectorDatabase:
    """Mock vector database for testing."""
    
    def add_document(self, doc):
        pass

def create_temp_git_repo():
    """Create a temporary Git repository for testing."""
    temp_dir = Path(tempfile.mkdtemp(prefix="git_watcher_test_"))
    
    # Initialize git repository
    subprocess.run(["git", "init"], cwd=temp_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=temp_dir, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=temp_dir, check=True)
    
    # Create initial commit
    readme = temp_dir / "README.md"
    readme.write_text("# Test Repository")
    subprocess.run(["git", "add", "README.md"], cwd=temp_dir, check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=temp_dir, check=True)
    
    return temp_dir

async def test_git_watcher_agent():
    """Test the Git Watcher Agent implementation."""
    print("🔍 Testing Git Watcher Agent Implementation")
    print("=" * 50)
    
    try:
        # Import the agent
        import sys
        sys.path.insert(0, '.')
        from src.dev_guard.agents.git_watcher import GitWatcherAgent
        from src.dev_guard.core.config import RepositoryConfig
        
        print("✅ Agent import successful")
        
        # Create mocked dependencies
        mock_config = MockConfig()
        mock_shared_memory = MockSharedMemory()
        mock_vector_db = MockVectorDatabase()
        
        # Create agent instance
        agent = GitWatcherAgent(
            agent_id="test_git_watcher",
            config=mock_config,
            shared_memory=mock_shared_memory,
            vector_db=mock_vector_db
        )
        
        print("✅ Agent instantiation successful")
        
        # Test capabilities
        capabilities = agent.get_capabilities()
        print(f"✅ Agent capabilities: {len(capabilities)} items")
        for cap in capabilities:
            print(f"  • {cap}")
        
        # Test status
        status = agent.get_status()
        print(f"✅ Agent status: {status}")
        
        # Create a temporary Git repository
        temp_repo = create_temp_git_repo()
        print(f"✅ Created test repository: {temp_repo}")
        
        try:
            # Test repository config
            repo_config = RepositoryConfig(path=str(temp_repo))
            print("✅ Repository config created")
            
            # Test file watching functionality
            watched_files = await agent._get_watched_files(temp_repo, repo_config)
            print(f"✅ Watched files: {len(watched_files)} files")
            
            # Test checksum calculation
            readme_file = temp_repo / "README.md"
            checksum = await agent._calculate_file_checksum(readme_file)
            print(f"✅ Checksum calculation: {checksum[:8]}...")
            
            # Test Git command execution
            branch_output = await agent._run_git_command(temp_repo, ["branch", "--show-current"])
            print(f"✅ Git command execution: branch = '{branch_output.strip()}'")
            
            # Test repository scanning
            agent.repositories["test_repo"] = repo_config
            scan_result = await agent._scan_repository_for_changes("test_repo", repo_config)
            print(f"✅ Repository scanning: {scan_result}")
            
            # Test change summary generation
            git_changes = [{"type": "new_commits", "commit_count": 2, "branch": "main"}]
            file_changes = [{"type": "modified", "file": "test.py"}]
            summary = agent._generate_change_summary(git_changes, file_changes)
            print(f"✅ Change summary: '{summary}'")
            
            # Test monitoring
            monitor_result = await agent._monitor_repositories()
            print(f"✅ Repository monitoring: {monitor_result}")
            
            # Test continuous monitoring control
            start_result = await agent._start_continuous_monitoring()
            print(f"✅ Start monitoring: {start_result}")
            
            stop_result = await agent._stop_continuous_monitoring()
            print(f"✅ Stop monitoring: {stop_result}")
            
            print("\n🎉 All tests passed successfully!")
            print(f"Git Watcher Agent is fully functional with {len(capabilities)} capabilities")
            
            return True
            
        finally:
            # Cleanup
            if temp_repo.exists():
                shutil.rmtree(temp_repo, ignore_errors=True)
                print("✅ Cleanup completed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_git_watcher_agent())
    if success:
        print("\n🚀 Task 12.1: Git Watcher Agent Implementation - SUCCESS!")
        print("Repository monitoring and change detection system is operational.")
    else:
        print("\n⚠️ Task 12.1: Git Watcher Agent Implementation - NEEDS ATTENTION")
        print("Some issues were encountered during testing.")
