"""
Integration tests for end-to-end DevGuard workflows.
Tests complete user workflows from request to completion.
"""

from unittest.mock import AsyncMock, patch
from types import SimpleNamespace

import pytest
import pytest_asyncio
from git import Repo

from dev_guard.core.config import Config
from dev_guard.core.swarm import DevGuardSwarm
from dev_guard.memory.shared_memory import SharedMemory
from dev_guard.memory.vector_db import VectorDatabase


class TestEndToEndWorkflows:
    """Test suite for complete end-to-end DevGuard workflows."""

    @pytest_asyncio.fixture
    async def test_environment(self, tmp_path):
        """Set up complete test environment with all components."""
        # Create test configuration
        config_data = {
            "llm": {
                "provider": "openrouter",
                "model": "qwen/qwen-2.5-coder-32b-instruct",
                "api_key": "mock-api-key"
            },
            "vector_db": {
                "provider": "chroma",
                "path": str(tmp_path / "vector_db")
            },
            "shared_memory": {
                "provider": "sqlite",
                "db_path": str(tmp_path / "memory.db")
            },
            "notifications": {
                "enabled": False
            },
            "agents": {
                "commander": {"enabled": True},
                "planner": {"enabled": True},
                "code": {"enabled": True},
                "qa_test": {"enabled": True},
                "git_watcher": {"enabled": True},
                "impact_mapper": {"enabled": True},
                "repo_auditor": {"enabled": True},
                "dependency_manager": {"enabled": True},
                "red_team": {"enabled": True},
                "docs": {"enabled": True}
            },
            "repositories": [
                {
                    "name": "test-repo",
                    "path": str(tmp_path / "test-repo"),
                    "enabled": True,
                    "branch": "main"
                }
            ]
        }
        
        # Create test repository
        test_repo_path = tmp_path / "test-repo"
        test_repo_path.mkdir()
        repo = Repo.init(test_repo_path)
        
        # Add some test files
        (test_repo_path / "README.md").write_text("# Test Repository")
        (test_repo_path / "src").mkdir()
        (test_repo_path / "src" / "__init__.py").write_text("")
        (test_repo_path / "src" / "calculator.py").write_text('''
def add(a, b):
    """Add two numbers."""
    return a + b

def divide(a, b):
    """Divide two numbers."""
    return a / b  # Potential division by zero
''')
        (test_repo_path / "requirements.txt").write_text(
            "requests==2.28.0\npandas==1.5.0"
        )
        
        # Commit files
        repo.index.add([
            str(test_repo_path / "README.md"),
            str(test_repo_path / "src" / "__init__.py"),
            str(test_repo_path / "src" / "calculator.py"),
            str(test_repo_path / "requirements.txt"),
        ])
        repo.index.commit("Initial commit")

        config = Config.load_from_dict(config_data)
        
        # Initialize core components
        shared_memory = SharedMemory(db_path=str(tmp_path / "memory.db"))
        vector_db = VectorDatabase(config.vector_db)
        
        # Mock LLM interface (SmartLLM-compatible)
        mock_llm = AsyncMock()

        return {
            "config": config,
            "shared_memory": shared_memory,
            "vector_db": vector_db,
            "mock_llm": mock_llm,
            "test_repo_path": test_repo_path,
            "repo": repo,
            "tmp_path": tmp_path
        }

    @pytest.mark.asyncio
    async def test_code_generation_workflow(self, test_environment):
        """Test complete code generation workflow from user request to completion."""
        env = test_environment
        
        # Initialize swarm with mocked LLM interface factory
        with patch('dev_guard.llm.factory.get_llm_interface', return_value=env["mock_llm"]):
            swarm = DevGuardSwarm(env["config"])
            await swarm.initialize()
            
            # Mock LLM responses for different agents
            env["mock_llm"].generate.side_effect = [
                SimpleNamespace(content='{"task_type": "code_generation", "agent": "code", "priority": "high", "description": "Generate unit tests for calculator.py"}'),
                SimpleNamespace(content='{"subtasks": [{"type": "analyze_code", "agent": "code"}, {"type": "generate_tests", "agent": "qa_test"}], "dependencies": [], "estimated_time": 300}'),
                SimpleNamespace(content='{"success": true, "files_modified": ["tests/test_calculator.py"], "summary": "Generated comprehensive unit tests"}'),
                SimpleNamespace(content='{"success": true, "tests_passed": 8, "coverage": 95, "summary": "All tests passing with high coverage"}')
            ]

            # Submit user request
            user_request = {
                "type": "code_generation",
                "description": "Generate comprehensive unit tests for the calculator.py file",
                "target_files": ["src/calculator.py"],
                "requirements": ["Include edge cases", "Achieve >90% coverage"]
            }
            
            # Process request through swarm
            result = await swarm.process_user_request(user_request)
            
            # Verify workflow completion
            assert result["success"], f"Workflow failed: {result.get('error')}"
            assert "task_id" in result
            assert result["status"] == "completed"
            
            # Verify agent coordination
            memory_entries = env["shared_memory"].get_recent_entries(limit=20)
            agent_types = {entry.agent_id for entry in memory_entries}
            
            # Should involve Commander, Planner, Code, and QA agents
            expected_agents = {"commander", "planner", "code", "qa_test"}
            assert expected_agents.intersection(agent_types), "Expected agents not involved in workflow"
            
            print("✅ Code generation workflow completed successfully")

    @pytest.mark.asyncio
    async def test_security_scan_workflow(self, test_environment):
        """Test complete security scanning workflow."""
        env = test_environment
        
        with patch('dev_guard.llm.factory.get_llm_interface', return_value=env["mock_llm"]):
            swarm = DevGuardSwarm(env["config"])
            await swarm.initialize()
            
            # Mock security scan responses
            env["mock_llm"].generate.side_effect = [
                SimpleNamespace(content='{"task_type": "security_scan", "agent": "red_team", "priority": "high"}'),
                SimpleNamespace(content='{"success": true, "vulnerabilities_found": 2, "critical": 0, "high": 1, "medium": 1, "risk_score": 65}'),
                SimpleNamespace(content='{"remediation_tasks": [{"type": "fix_vulnerability", "file": "src/calculator.py", "issue": "division_by_zero"}]}')
            ]

            # Submit security scan request
            security_request = {
                "type": "security_scan",
                "description": "Perform comprehensive security scan of the repository",
                "scan_types": ["sast", "secrets", "dependencies"],
                "target_path": str(env["test_repo_path"])
            }
            
            result = await swarm.process_user_request(security_request)
            
            assert result["success"], f"Security workflow failed: {result.get('error')}"
            
            # Verify security findings were logged
            security_entries = env["shared_memory"].search_entries(tags={"security", "vulnerability"})
            assert len(security_entries) > 0, "No security findings logged"
            
            print("✅ Security scan workflow completed successfully")

    @pytest.mark.asyncio  
    async def test_cross_repository_impact_analysis_workflow(self, test_environment):
        """Test cross-repository impact analysis workflow."""
        env = test_environment
        
        # Create second repository for impact analysis
        repo2_path = env["tmp_path"] / "dependent-repo"
        repo2_path.mkdir()
        repo2 = Repo.init(repo2_path)
        
        (repo2_path / "main.py").write_text('''
from test_repo.src.calculator import add, divide

def calculate_total(values):
    total = 0
    for val in values:
        total = add(total, val)
    return total
''')
        repo2.index.add_items([repo2_path / "main.py"])
        repo2.index.commit("Add dependency on test-repo")
        
        # Update config to include both repositories
        env["config"].repositories.append({
            "name": "dependent-repo", 
            "path": str(repo2_path),
            "enabled": True,
            "branch": "main"
        })
        
        with patch('dev_guard.core.swarm.OpenRouterClient', return_value=env["mock_llm"]):
            swarm = DevGuardSwarm(env["config"])
            await swarm.initialize()
            
            # Mock impact analysis responses
            env["mock_llm"].chat_completion.side_effect = [
                # Commander routing to impact mapper
                AsyncMock(content='{"task_type": "impact_analysis", "agent": "impact_mapper", "priority": "medium"}'),
                # Impact Mapper analysis
                AsyncMock(content='{"affected_repositories": ["dependent-repo"], "impact_level": "medium", "breaking_changes": false, "coordination_required": true}'),
                # Planner creates coordination tasks
                AsyncMock(content='{"coordination_tasks": [{"type": "update_dependencies", "repository": "dependent-repo"}]}')
            ]
            
            # Simulate API change in calculator.py
            impact_request = {
                "type": "impact_analysis",
                "description": "Analyze impact of changing add() function signature",
                "source_repository": "test-repo",
                "changes": [
                    {
                        "file": "src/calculator.py",
                        "change_type": "modification",
                        "old_content": "def add(a, b):",
                        "new_content": "def add(a, b, c=0):"
                    }
                ]
            }
            
            result = await swarm.process_user_request(impact_request)
            
            assert result["success"], f"Impact analysis failed: {result.get('error')}"
            
            # Verify impact analysis was performed
            impact_entries = env["shared_memory"].search_entries(tags={"impact", "analysis"})
            assert len(impact_entries) > 0, "No impact analysis logged"
            
            print("✅ Cross-repository impact analysis workflow completed successfully")

    @pytest.mark.asyncio
    async def test_documentation_generation_workflow(self, test_environment):
        """Test automated documentation generation workflow."""
        env = test_environment
        
        with patch('dev_guard.core.swarm.OpenRouterClient', return_value=env["mock_llm"]):
            swarm = DevGuardSwarm(env["config"])
            await swarm.initialize()
            
            # Mock documentation responses
            env["mock_llm"].chat_completion.side_effect = [
                # Commander routes to docs agent
                AsyncMock(content='{"task_type": "generate_documentation", "agent": "docs", "priority": "medium"}'),
                # Docs Agent generates documentation
                AsyncMock(content='{"success": true, "files_created": ["docs/calculator.md", "docs/api.md"], "coverage": 100}'),
                # QA Agent validates documentation  
                AsyncMock(content='{"success": true, "documentation_quality": "excellent", "completeness": 95}')
            ]
            
            docs_request = {
                "type": "generate_documentation",
                "description": "Generate comprehensive documentation for calculator module",
                "target_files": ["src/calculator.py"],
                "doc_types": ["api", "usage", "examples"]
            }
            
            result = await swarm.process_user_request(docs_request)
            
            assert result["success"], f"Documentation workflow failed: {result.get('error')}"
            
            # Verify documentation entries
            doc_entries = env["shared_memory"].search_entries(tags={"documentation", "generation"})
            assert len(doc_entries) > 0, "No documentation generation logged"
            
            print("✅ Documentation generation workflow completed successfully")

    @pytest.mark.asyncio
    async def test_dependency_management_workflow(self, test_environment):
        """Test dependency management and security scanning workflow."""
        env = test_environment
        
        with patch('dev_guard.core.swarm.OpenRouterClient', return_value=env["mock_llm"]):
            swarm = DevGuardSwarm(env["config"])
            await swarm.initialize()
            
            # Mock dependency management responses
            env["mock_llm"].chat_completion.side_effect = [
                # Commander routes to dependency manager
                AsyncMock(content='{"task_type": "dependency_audit", "agent": "dependency_manager", "priority": "high"}'),
                # Dependency Manager audit results
                AsyncMock(content='{"vulnerabilities": [{"package": "requests", "version": "2.28.0", "cve": "CVE-2023-32681", "severity": "medium"}], "outdated_packages": 2}'),
                # Planner creates update plan
                AsyncMock(content='{"update_plan": [{"package": "requests", "current": "2.28.0", "target": "2.31.0", "breaking_changes": false}]}')
            ]
            
            dependency_request = {
                "type": "dependency_audit",
                "description": "Audit dependencies for security vulnerabilities and updates",
                "target_files": ["requirements.txt"],
                "include_transitive": True
            }
            
            result = await swarm.process_user_request(dependency_request)
            
            assert result["success"], f"Dependency workflow failed: {result.get('error')}"
            
            # Verify dependency audit was logged
            dep_entries = env["shared_memory"].search_entries(tags={"dependency", "audit"})
            assert len(dep_entries) > 0, "No dependency audit logged"
            
            print("✅ Dependency management workflow completed successfully")

    @pytest.mark.asyncio
    async def test_complete_development_lifecycle_workflow(self, test_environment):
        """Test complete development lifecycle from feature request to deployment."""
        env = test_environment
        
        with patch('dev_guard.core.swarm.OpenRouterClient', return_value=env["mock_llm"]):
            swarm = DevGuardSwarm(env["config"])
            await swarm.initialize()
            
            # Mock responses for complete lifecycle
            responses = [
                # Commander initial routing
                AsyncMock(content='{"task_type": "feature_development", "requires_coordination": true, "agents": ["planner", "code", "qa_test", "docs", "red_team"]}'),
                # Planner creates development plan
                AsyncMock(content='{"development_plan": {"phases": ["analysis", "implementation", "testing", "security", "documentation"], "estimated_time": 1800}}'),
                # Code Agent implements feature
                AsyncMock(content='{"success": true, "files_modified": ["src/calculator.py", "src/advanced_math.py"], "lines_added": 150}'),
                # QA Agent runs tests
                AsyncMock(content='{"success": true, "tests_created": 12, "coverage": 92, "all_passed": true}'),
                # Red Team security scan
                AsyncMock(content='{"success": true, "vulnerabilities": 0, "risk_score": 15, "security_grade": "A"}'),
                # Docs Agent updates documentation
                AsyncMock(content='{"success": true, "docs_updated": ["README.md", "docs/api.md"], "examples_added": 3}'),
                # Final coordination
                AsyncMock(content='{"success": true, "feature_ready": true, "all_checks_passed": true}')
            ]
            
            env["mock_llm"].chat_completion.side_effect = responses
            
            # Submit complete feature request
            feature_request = {
                "type": "feature_development", 
                "description": "Add advanced mathematical operations (sqrt, pow, factorial) to calculator",
                "requirements": [
                    "Implement sqrt, pow, and factorial functions",
                    "Add comprehensive unit tests",
                    "Update documentation with examples",
                    "Ensure security compliance",
                    "Maintain >90% test coverage"
                ],
                "acceptance_criteria": [
                    "All functions handle edge cases",
                    "Performance benchmarks met",
                    "Security scan passes",
                    "Documentation is complete"
                ]
            }
            
            result = await swarm.process_user_request(feature_request)
            
            assert result["success"], f"Complete lifecycle failed: {result.get('error')}"
            
            # Verify all phases were executed
            lifecycle_entries = env["shared_memory"].get_recent_entries(limit=50)
            involved_agents = {entry.agent_id for entry in lifecycle_entries}
            
            expected_agents = {"commander", "planner", "code", "qa_test", "red_team", "docs"}
            assert expected_agents.issubset(involved_agents), f"Missing agents: {expected_agents - involved_agents}"
            
            # Verify task coordination
            task_entries = [entry for entry in lifecycle_entries if entry.type == "task"]
            assert len(task_entries) >= 5, "Insufficient task coordination logged"
            
            print("✅ Complete development lifecycle workflow completed successfully")

    @pytest.mark.asyncio
    async def test_multi_repository_coordination_workflow(self, test_environment):
        """Test coordination workflow across multiple repositories."""
        env = test_environment
        
        # Create multiple test repositories
        repos = []
        for i in range(3):
            repo_path = env["tmp_path"] / f"repo-{i}"
            repo_path.mkdir()
            repo = Repo.init(repo_path)
            
            # Add interdependent files
            (repo_path / f"service_{i}.py").write_text(f'''
def service_{i}_function():
    """Service {i} main function."""
    return "service_{i}_result"
''')
            if i > 0:
                (repo_path / f"client_{i-1}.py").write_text(f'''
# Depends on repo-{i-1}
import service_{i-1}

def call_service_{i-1}():
    return service_{i-1}.service_{i-1}_function()
''')
            
            repo.index.add_items([f for f in repo_path.glob("*.py")])
            repo.index.commit(f"Initial commit for repo-{i}")
            repos.append((repo_path, repo))
        
        # Update config with all repositories
        env["config"].repositories.extend([
            {"name": f"repo-{i}", "path": str(path), "enabled": True, "branch": "main"}
            for i, (path, _) in enumerate(repos)
        ])
        
        with patch('dev_guard.core.swarm.OpenRouterClient', return_value=env["mock_llm"]):
            swarm = DevGuardSwarm(env["config"])
            await swarm.initialize()
            
            # Mock multi-repo coordination responses
            env["mock_llm"].chat_completion.side_effect = [
                # Commander recognizes multi-repo change
                AsyncMock(content='{"task_type": "multi_repo_coordination", "requires_impact_analysis": true, "affected_repos": 3}'),
                # Impact Mapper analyzes cross-repo dependencies
                AsyncMock(content='{"dependency_graph": {"repo-0": ["repo-1"], "repo-1": ["repo-2"]}, "impact_scope": "cascade", "coordination_required": true}'),
                # Planner creates coordination strategy
                AsyncMock(content='{"coordination_plan": {"update_order": ["repo-2", "repo-1", "repo-0"], "parallel_tasks": [], "validation_steps": 3}}'),
                # Sequential execution confirmations
                AsyncMock(content='{"success": true, "repo": "repo-2", "changes_applied": true}'),
                AsyncMock(content='{"success": true, "repo": "repo-1", "changes_applied": true, "dependencies_updated": true}'),
                AsyncMock(content='{"success": true, "repo": "repo-0", "changes_applied": true, "integration_tested": true}')
            ]
            
            coordination_request = {
                "type": "multi_repository_update",
                "description": "Update API interface across all dependent repositories",
                "changes": {
                    "repo-0": [{"file": "service_0.py", "change": "Add new parameter to service_0_function"}],
                    "repo-1": [{"file": "client_0.py", "change": "Update calls to service_0_function"}],
                    "repo-2": [{"file": "client_1.py", "change": "Handle new service interface"}]
                },
                "coordination_strategy": "sequential",
                "validation_required": True
            }
            
            result = await swarm.process_user_request(coordination_request)
            
            assert result["success"], f"Multi-repo coordination failed: {result.get('error')}"
            
            # Verify coordination was logged
            coord_entries = env["shared_memory"].search_entries(tags={"coordination", "multi_repository"})
            assert len(coord_entries) > 0, "No coordination activities logged"
            
            # Verify impact analysis was performed
            impact_entries = env["shared_memory"].search_entries(tags={"impact", "dependency"})
            assert len(impact_entries) > 0, "No impact analysis logged"
            
            print("✅ Multi-repository coordination workflow completed successfully")

    @pytest.mark.asyncio
    async def test_notification_integration_workflow(self, test_environment):
        """Test notification system integration in workflows."""
        env = test_environment
        
        # Enable notifications for this test
        env["config"].notifications.enabled = True
        
        with patch('dev_guard.core.swarm.OpenRouterClient', return_value=env["mock_llm"]):
            with patch('dev_guard.notifications.notification_manager.NotificationManager') as mock_notification_manager:
                mock_manager = AsyncMock()
                mock_notification_manager.return_value = mock_manager
                
                swarm = DevGuardSwarm(env["config"])
                await swarm.initialize()
                
                # Mock workflow with critical security finding
                env["mock_llm"].chat_completion.side_effect = [
                    # Commander detects critical security issue
                    AsyncMock(content='{"task_type": "security_incident", "severity": "critical", "immediate_action": true}'),
                    # Security analysis result
                    AsyncMock(content='{"critical_vulnerabilities": 1, "cve": "CVE-2024-12345", "exploit_available": true, "immediate_patching_required": true}')
                ]
                
                critical_request = {
                    "type": "security_incident",
                    "description": "Critical security vulnerability detected in production dependencies",
                    "severity": "critical",
                    "requires_immediate_action": True
                }
                
                result = await swarm.process_user_request(critical_request)
                
                assert result["success"], f"Critical security workflow failed: {result.get('error')}"
                
                # Verify notifications were triggered
                assert mock_manager.send_notification.called, "Notifications not triggered for critical issue"
                
                # Check notification content
                call_args = mock_manager.send_notification.call_args[0][0]
                assert call_args.level.value == "critical", "Notification level incorrect"
                assert "security" in call_args.content.lower(), "Notification missing security context"
                
                print("✅ Notification integration workflow completed successfully")
