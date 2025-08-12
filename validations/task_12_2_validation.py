#!/usr/bin/env python3
"""
Task 12.2 Validation: Multi-Repository Monitoring and Git Integration

This script validates the implementation of Task 12.2 by testing the enhanced 
Git Watcher Agent with comprehensive multi-repository coordination capabilities.
"""

import asyncio
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dev_guard.agents.git_watcher import GitWatcherAgent, RepositoryConfig, RepositoryRelationship
from dev_guard.core.config import Config, VectorDBConfig
from dev_guard.memory.shared_memory import SharedMemory
from dev_guard.memory.vector_db import VectorDatabase


class Task12_2Validator:
    """Validates Task 12.2 multi-repository monitoring implementation."""
    
    def __init__(self):
        self.temp_dir = None
        self.test_repos = {}
        self.shared_memory = None
        self.config = None
        self.git_watcher = None
        
    async def setup_test_environment(self):
        """Set up test repositories and agent."""
        print("🔧 Setting up test environment...")
        
        # Create temporary directory for test repositories
        self.temp_dir = Path(tempfile.mkdtemp(prefix="dev_guard_task12_2_"))
        print(f"   Test directory: {self.temp_dir}")
        
        # Create multiple test repositories
        await self._create_test_repositories()
        
        # Initialize shared memory and vector database
        self.shared_memory = SharedMemory()
        
        # Create vector database config
        vector_config = VectorDBConfig(
            path=str(self.temp_dir / "vector_db"),
            collection_name="test_collection"
        )
        vector_db = VectorDatabase(vector_config)
        
        # Create config (repositories will be initialized directly in the agent)
        self.config = Config()
        
        # Initialize Git Watcher Agent
        self.git_watcher = GitWatcherAgent(
            agent_id="test_git_watcher_12_2",
            config=self.config,
            shared_memory=self.shared_memory,
            vector_db=vector_db
        )
        
        # Initialize repositories directly in the agent
        self.git_watcher.repositories = {}
        for repo_name, repo_path in self.test_repos.items():
            repo_config = RepositoryConfig(
                path=str(repo_path),
                branch="main",
                auto_commit=False,
                auto_push=False
            )
            self.git_watcher.repositories[repo_name] = repo_config
            
        print("   ✅ Test environment ready")
        
    async def _create_test_repositories(self):
        """Create test Git repositories."""
        repo_names = ["frontend", "backend", "shared-lib", "docs"]
        
        for repo_name in repo_names:
            repo_path = self.temp_dir / repo_name
            repo_path.mkdir()
            
            # Initialize Git repository
            await self._run_git_command(repo_path, ["init"])
            await self._run_git_command(repo_path, ["config", "user.name", "Test User"])
            await self._run_git_command(repo_path, ["config", "user.email", "test@example.com"])
            
            # Create initial files
            (repo_path / "README.md").write_text(f"# {repo_name.title()} Repository\n\nTest repository for multi-repo coordination.")
            (repo_path / ".gitignore").write_text("*.log\n*.tmp\n__pycache__/\n")
            
            if repo_name == "frontend":
                (repo_path / "package.json").write_text("""{
  "name": "frontend-app",
  "version": "1.0.0",
  "dependencies": {
    "react": "^18.0.0",
    "axios": "^1.0.0"
  }
}""")
            elif repo_name == "backend":
                (repo_path / "requirements.txt").write_text("fastapi>=0.100.0\nuvicorn>=0.20.0\nrequests>=2.28.0\n")
            elif repo_name == "shared-lib":
                (repo_path / "pyproject.toml").write_text("""[project]
name = "shared-lib"
version = "1.0.0"
dependencies = [
    "pydantic>=2.0.0",
    "httpx>=0.25.0"
]
""")
            
            # Create initial commit
            await self._run_git_command(repo_path, ["add", "."])
            await self._run_git_command(repo_path, ["commit", "-m", f"Initial commit for {repo_name}"])
            
            # Create main branch explicitly
            await self._run_git_command(repo_path, ["branch", "-M", "main"])
            
            self.test_repos[repo_name] = repo_path
    
    async def _run_git_command(self, repo_path, command):
        """Run a git command in a repository."""
        try:
            result = subprocess.run(
                ["git"] + command,
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Git command failed: {e}")
            return ""
    
    async def test_repository_group_creation(self):
        """Test Task 12.2: Repository group creation and management."""
        print("\n🧪 Testing repository group creation...")
        
        # Create a microservices group
        task = {
            "task_type": "create_repository_group",
            "group_name": "microservices",
            "repositories": ["test_repo_1", "test_repo_2", "test_repo_3"],
            "relationship_type": RepositoryRelationship.MICROSERVICE.value,
            "sync_strategy": "parallel",
            "priority": 1
        }
        
        result = await self.git_watcher.execute_task(task)
        
        if result.get("success"):
            print("   ✅ Repository group created successfully")
            print(f"   📊 Group: {result.get('group_name')}")
            print(f"   📊 Repositories: {len(result.get('repositories', []))}")
            print(f"   📊 Strategy: {result.get('sync_strategy')}")
            return True
        else:
            print(f"   ❌ Repository group creation failed: {result.get('error')}")
            return False
    
    async def test_cross_repository_impact_analysis(self):
        """Test Task 12.2: Cross-repository impact analysis."""
        print("\n🧪 Testing cross-repository impact analysis...")
        
        task = {
            "task_type": "analyze_cross_repository_impact",
            "repository": "test_repo_1",
            "change_type": "commit"
        }
        
        result = await self.git_watcher.execute_task(task)
        
        if result.get("success"):
            print("   ✅ Cross-repository impact analysis completed")
            print(f"   📊 Primary repository: {result.get('primary_repository')}")
            print(f"   📊 Potentially affected: {len(result.get('potentially_affected', []))}")
            print(f"   📊 High impact: {len(result.get('high_impact', []))}")
            print(f"   📊 Medium impact: {len(result.get('medium_impact', []))}")
            return True
        else:
            print(f"   ❌ Cross-repository impact analysis failed: {result.get('error')}")
            return False
    
    async def test_batch_repository_operations(self):
        """Test Task 12.2: Batch repository operations."""
        print("\n🧪 Testing batch repository operations...")
        
        task = {
            "task_type": "execute_batch_repository_operation",
            "operation": "status",
            "repositories": ["test_repo_1", "test_repo_2"],
            "parallel": True
        }
        
        result = await self.git_watcher.execute_task(task)
        
        if result.get("success"):
            print("   ✅ Batch repository operation completed")
            print(f"   📊 Operation: {result.get('operation')}")
            print(f"   📊 Total repositories: {result.get('total_repositories')}")
            print(f"   📊 Successful operations: {result.get('successful_operations')}")
            print(f"   📊 Parallel execution: {result.get('parallel_execution')}")
            return True
        else:
            print(f"   ❌ Batch repository operation failed: {result.get('error')}")
            return False
    
    async def test_repository_group_synchronization(self):
        """Test Task 12.2: Repository group synchronization."""
        print("\n🧪 Testing repository group synchronization...")
        
        task = {
            "task_type": "sync_repository_group",
            "group_name": "microservices"
        }
        
        result = await self.git_watcher.execute_task(task)
        
        if result.get("success"):
            print("   ✅ Repository group synchronization completed")
            print(f"   📊 Group: {result.get('group_name')}")
            print(f"   📊 Sync strategy: {result.get('sync_strategy')}")
            print(f"   📊 Total repositories: {result.get('total_repositories')}")
            print(f"   📊 Successful syncs: {result.get('successful_syncs')}")
            return True
        else:
            print(f"   ❌ Repository group synchronization failed: {result.get('error')}")
            return False
    
    async def test_git_workflow_monitoring(self):
        """Test Task 12.2: Git workflow monitoring."""
        print("\n🧪 Testing Git workflow monitoring...")
        
        # First make some changes to create workflow events
        await self._create_test_workflow_events()
        
        task = {
            "task_type": "monitor_git_workflows",
            "repositories": ["test_repo_1", "test_repo_2"]
        }
        
        result = await self.git_watcher.execute_task(task)
        
        if result.get("success"):
            print("   ✅ Git workflow monitoring completed")
            print(f"   📊 Monitored repositories: {result.get('monitored_repositories')}")
            print(f"   📊 Workflow events detected: {result.get('workflow_events_detected')}")
            return True
        else:
            print(f"   ❌ Git workflow monitoring failed: {result.get('error')}")
            return False
    
    async def _create_test_workflow_events(self):
        """Create some workflow events for testing."""
        # Add a test tag to the first repository
        first_repo = list(self.test_repos.values())[0]
        await self._run_git_command(first_repo, ["tag", "v1.0.0", "-m", "First release"])
        
        # Create a new branch in the second repository
        if len(self.test_repos) > 1:
            second_repo = list(self.test_repos.values())[1]
            await self._run_git_command(second_repo, ["checkout", "-b", "feature/new-feature"])
            
            # Add a commit on the feature branch
            (second_repo / "feature.txt").write_text("New feature implementation")
            await self._run_git_command(second_repo, ["add", "feature.txt"])
            await self._run_git_command(second_repo, ["commit", "-m", "feat: add new feature"])
    
    async def test_repository_health_analysis(self):
        """Test Task 12.2: Repository health analysis."""
        print("\n🧪 Testing repository health analysis...")
        
        task = {
            "task_type": "analyze_repository_health",
            "repositories": list(self.git_watcher.repositories.keys())
        }
        
        result = await self.git_watcher.execute_task(task)
        
        if result.get("success"):
            print("   ✅ Repository health analysis completed")
            print(f"   📊 Overall health score: {result.get('overall_health_score')}")
            print(f"   📊 Repositories analyzed: {result.get('repositories_analyzed')}")
            print(f"   📊 Healthy repos: {result.get('healthy_repos')}")
            print(f"   📊 Attention needed: {result.get('attention_needed')}")
            return True
        else:
            print(f"   ❌ Repository health analysis failed: {result.get('error')}")
            return False
    
    async def test_dependency_management(self):
        """Test Task 12.2: Cross-repository dependency management."""
        print("\n🧪 Testing dependency management...")
        
        task = {
            "task_type": "manage_cross_repo_dependencies",
            "operation": "analyze",
            "repositories": list(self.git_watcher.repositories.keys())
        }
        
        result = await self.git_watcher.execute_task(task)
        
        if result.get("success"):
            print("   ✅ Dependency management analysis completed")
            print(f"   📊 Operation: {result.get('operation')}")
            print(f"   📊 Repositories analyzed: {result.get('repositories_analyzed')}")
            return True
        else:
            print(f"   ❌ Dependency management failed: {result.get('error')}")
            return False
    
    async def test_repository_consistency_validation(self):
        """Test Task 12.2: Repository consistency validation."""
        print("\n🧪 Testing repository consistency validation...")
        
        task = {
            "task_type": "validate_repository_consistency",
            "repositories": list(self.git_watcher.repositories.keys()),
            "rules": {
                "branch_naming": True,
                "file_structure": True,
                "dependency_versions": True
            },
            "required_files": ["README.md", ".gitignore"]
        }
        
        result = await self.git_watcher.execute_task(task)
        
        if result.get("success"):
            print("   ✅ Repository consistency validation completed")
            print(f"   📊 Overall compliance score: {result.get('overall_compliance_score')}")
            print(f"   📊 Repositories validated: {result.get('repositories_validated')}")
            print(f"   📊 Compliant repos: {result.get('compliant_repos')}")
            return True
        else:
            print(f"   ❌ Repository consistency validation failed: {result.get('error')}")
            return False
    
    async def test_enhanced_capabilities(self):
        """Test Task 12.2: Enhanced agent capabilities."""
        print("\n🧪 Testing enhanced agent capabilities...")
        
        capabilities = self.git_watcher.get_capabilities()
        status = self.git_watcher.get_status()
        
        # Check for new multi-repository capabilities
        required_capabilities = [
            "repository_group_management",
            "cross_repository_analysis", 
            "batch_operations",
            "git_workflow_monitoring",
            "repository_health_analysis",
            "dependency_management",
            "consistency_validation"
        ]
        
        missing_capabilities = [cap for cap in required_capabilities if cap not in capabilities]
        
        if not missing_capabilities:
            print("   ✅ All required capabilities present")
            print(f"   📊 Total capabilities: {len(capabilities)}")
            print(f"   📊 Repository groups: {status.get('repository_groups_count', 0)}")
            print(f"   📊 Cross-repo changes tracked: {status.get('cross_repo_changes_tracked', 0)}")
            print(f"   📊 Workflow events tracked: {status.get('workflow_events_tracked', 0)}")
            return True
        else:
            print(f"   ❌ Missing capabilities: {missing_capabilities}")
            return False
    
    async def run_validation(self):
        """Run complete Task 12.2 validation."""
        print("="*70)
        print("🚀 TASK 12.2 VALIDATION: Multi-Repository Monitoring & Git Integration")
        print("="*70)
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Setup test environment
            await self.setup_test_environment()
            
            # Run all tests
            tests = [
                ("Repository Group Creation", self.test_repository_group_creation()),
                ("Cross-Repository Impact Analysis", self.test_cross_repository_impact_analysis()),
                ("Batch Repository Operations", self.test_batch_repository_operations()),
                ("Repository Group Synchronization", self.test_repository_group_synchronization()),
                ("Git Workflow Monitoring", self.test_git_workflow_monitoring()),
                ("Repository Health Analysis", self.test_repository_health_analysis()),
                ("Dependency Management", self.test_dependency_management()),
                ("Repository Consistency Validation", self.test_repository_consistency_validation()),
                ("Enhanced Capabilities", self.test_enhanced_capabilities()),
            ]
            
            results = []
            for test_name, test_coro in tests:
                try:
                    result = await test_coro
                    results.append((test_name, result))
                except Exception as e:
                    print(f"   ❌ {test_name} failed with exception: {e}")
                    results.append((test_name, False))
            
            # Print summary
            print("\n" + "="*70)
            print("📊 TASK 12.2 VALIDATION SUMMARY")
            print("="*70)
            
            passed_tests = sum(1 for _, result in results if result)
            total_tests = len(results)
            
            for test_name, result in results:
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"{status} {test_name}")
            
            print(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
            
            if passed_tests == total_tests:
                print("🎉 TASK 12.2 IMPLEMENTATION: COMPLETE ✅")
                print("\n✨ Multi-repository monitoring and Git integration successfully implemented!")
                print("   • Repository group management")
                print("   • Cross-repository impact analysis") 
                print("   • Batch repository operations")
                print("   • Git workflow monitoring")
                print("   • Repository health analysis")
                print("   • Dependency management")
                print("   • Consistency validation")
                print("   • Enhanced agent capabilities")
                return True
            else:
                failed_tests = total_tests - passed_tests
                print(f"⚠️  TASK 12.2 IMPLEMENTATION: PARTIAL ({failed_tests} issues)")
                return False
                
        except Exception as e:
            print(f"\n❌ VALIDATION FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up test environment."""
        if self.temp_dir and self.temp_dir.exists():
            print(f"\n🧹 Cleaning up test directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir)


async def main():
    """Main validation function."""
    validator = Task12_2Validator()
    success = await validator.run_validation()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
