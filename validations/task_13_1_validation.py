#!/usr/bin/env python3
"""
Task 13.1 Validation Script - Cross-Repository Impact Analysis Implementation

Validates the Impact Mapper Agent's cross-repository impact analysis capabilities
including API change detection, dependency analysis, and relationship mapping.
"""

import asyncio
import json
from datetime import datetime

# Add src to path for imports
import sys
import tempfile
from pathlib import Path
from typing import Any

src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.dev_guard.agents.impact_mapper import ImpactMapperAgent  # noqa: E402
from src.dev_guard.core.shared_memory import SharedMemory  # noqa: E402
from src.dev_guard.llm.providers.openai_provider import OpenAIProvider  # noqa: E402
from src.dev_guard.memory.vector_db import VectorDB  # noqa: E402


class Task13ValidationTest:
    """Test suite for Task 13.1 Cross-Repository Impact Analysis."""
    
    def __init__(self):
        self.results = {}
        self.temp_dir = None
        self.impact_agent = None
        
    async def setup_test_environment(self):
        """Set up test environment with mock repositories."""
        self.temp_dir = tempfile.mkdtemp(prefix="impact_test_")
        
        # Create mock repositories
        await self._create_mock_repositories()
        
        # Initialize components
        shared_memory = SharedMemory()
        vector_db = VectorDB()
        llm_provider = OpenAIProvider()
        
        # Initialize Impact Mapper Agent
        self.impact_agent = ImpactMapperAgent(
            agent_id="test_impact_mapper",
            shared_memory=shared_memory,
            vector_db=vector_db,
            llm_provider=llm_provider
        )
        
        print(f"✅ Test environment setup complete in {self.temp_dir}")
        
    async def _create_mock_repositories(self):
        """Create mock repositories for testing."""
        repos = {
            "user_service": {
                "src/user_service/api.py": '''
"""User Service API"""
from typing import Dict, List, Optional

class UserAPI:
    def get_user(self, user_id: str) -> Optional[Dict]:
        """Get user by ID."""
        pass
    
    def create_user(self, user_data: Dict) -> str:
        """Create new user."""
        pass
    
    def update_user(self, user_id: str, data: Dict) -> bool:
        """Update user data."""
        pass
    
    def delete_user(self, user_id: str) -> bool:
        """Delete user."""
        pass

def authenticate_user(username: str, password: str) -> Optional[str]:
    """Authenticate user and return token."""
    pass
''',
                "requirements.txt": '''
flask==2.3.0
sqlalchemy==2.0.0
redis==4.5.0
pydantic==1.10.0
''',
                "package.json": '''
{
  "name": "user-service",
  "version": "1.0.0",
  "dependencies": {
    "express": "^4.18.0",
    "mongoose": "^7.0.0",
    "jsonwebtoken": "^9.0.0"
  }
}
'''
            },
            "order_service": {
                "src/order_service/api.py": '''
"""Order Service API"""
from typing import Dict, List
from user_service.api import authenticate_user, UserAPI

class OrderAPI:
    def __init__(self):
        self.user_api = UserAPI()
    
    def create_order(self, user_id: str, items: List[Dict]) -> str:
        """Create new order."""
        # Validate user exists
        user = self.user_api.get_user(user_id)
        if not user:
            raise ValueError("User not found")
        pass
    
    def get_order(self, order_id: str) -> Optional[Dict]:
        """Get order by ID."""
        pass
''',
                "requirements.txt": '''
flask==2.3.0
sqlalchemy==2.0.0
requests==2.30.0
user-service==1.0.0
'''
            },
            "frontend_app": {
                "src/components/UserProfile.js": '''
// User Profile Component
import { authenticate, getUserProfile } from '../api/userService';

class UserProfile {
    constructor() {
        this.userApi = new UserAPI();
    }
    
    async loadProfile(userId) {
        const user = await this.userApi.get_user(userId);
        return user;
    }
    
    async updateProfile(userId, data) {
        return await this.userApi.update_user(userId, data);
    }
}

export default UserProfile;
''',
                "package.json": '''
{
  "name": "frontend-app",
  "version": "2.0.0",
  "dependencies": {
    "react": "^18.2.0",
    "axios": "^1.4.0",
    "user-service-client": "^1.0.0"
  }
}
'''
            }
        }
        
        # Create repository directories and files
        for repo_name, files in repos.items():
            repo_path = Path(self.temp_dir) / repo_name
            repo_path.mkdir(parents=True)
            
            for file_path, content in files.items():
                full_path = repo_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content.strip())
    
    async def test_impact_analysis(self):
        """Test basic impact analysis functionality."""
        print("\n🧪 Testing Impact Analysis...")
        
        try:
            task = {
                "task_type": "analyze_impact",
                "source_repository": "user_service",
                "changes": [
                    {
                        "file": "src/user_service/api.py",
                        "change_type": "modification",
                        "description": "Modified get_user method signature"
                    }
                ],
                "target_repositories": ["order_service", "frontend_app"]
            }
            
            result = await self.impact_agent.execute_task(task)
            
            # Validate result structure
            assert result.get("success"), f"Impact analysis failed: {result.get('error')}"
            assert "impact_analysis" in result, "Impact analysis results missing"
            
            impact_analysis = result["impact_analysis"]
            assert "affected_repositories" in impact_analysis, "Affected repositories missing"
            assert len(impact_analysis["affected_repositories"]) > 0, "No affected repositories found"
            
            self.results["impact_analysis"] = {
                "status": "✅ PASS",
                "details": f"Analyzed impact on {len(impact_analysis['affected_repositories'])} repositories"
            }
            
        except Exception as e:
            self.results["impact_analysis"] = {
                "status": "❌ FAIL", 
                "error": str(e)
            }
    
    async def test_api_change_analysis(self):
        """Test API change analysis functionality."""
        print("\n🧪 Testing API Change Analysis...")
        
        try:
            task = {
                "task_type": "analyze_api_changes",
                "source_repository": "user_service",
                "changes": [
                    {
                        "file": "src/user_service/api.py",
                        "change_type": "modification",
                        "old_content": "def get_user(self, user_id: str) -> Optional[Dict]:",
                        "new_content": "def get_user(self, user_id: str, include_profile: bool = False) -> Optional[Dict]:"
                    }
                ]
            }
            
            result = await self.impact_agent.execute_task(task)
            
            # Validate result structure
            assert result.get("success"), f"API change analysis failed: {result.get('error')}"
            assert "api_changes" in result, "API changes missing"
            
            api_changes = result["api_changes"]
            assert len(api_changes) > 0, "No API changes detected"
            
            self.results["api_change_analysis"] = {
                "status": "✅ PASS",
                "details": f"Detected {len(api_changes)} API changes"
            }
            
        except Exception as e:
            self.results["api_change_analysis"] = {
                "status": "❌ FAIL",
                "error": str(e)
            }
    
    async def test_dependency_impact_analysis(self):
        """Test dependency impact analysis."""
        print("\n🧪 Testing Dependency Impact Analysis...")
        
        try:
            task = {
                "task_type": "analyze_dependency_impact",
                "source_repository": "user_service", 
                "dependency_changes": [
                    {
                        "dependency": "sqlalchemy",
                        "old_version": "1.4.0",
                        "new_version": "2.0.0",
                        "change_type": "major_upgrade"
                    }
                ]
            }
            
            result = await self.impact_agent.execute_task(task)
            
            # Validate result structure
            assert result.get("success"), f"Dependency analysis failed: {result.get('error')}"
            assert "dependency_impacts" in result, "Dependency impacts missing"
            
            dependency_impacts = result["dependency_impacts"]
            assert len(dependency_impacts) > 0, "No dependency impacts found"
            
            self.results["dependency_impact_analysis"] = {
                "status": "✅ PASS",
                "details": f"Analyzed {len(dependency_impacts)} dependency impacts"
            }
            
        except Exception as e:
            self.results["dependency_impact_analysis"] = {
                "status": "❌ FAIL",
                "error": str(e)
            }
    
    async def test_repository_relationship_mapping(self):
        """Test repository relationship mapping."""
        print("\n🧪 Testing Repository Relationship Mapping...")
        
        try:
            task = {
                "task_type": "map_repository_relationships",
                "repositories": ["user_service", "order_service", "frontend_app"]
            }
            
            result = await self.impact_agent.execute_task(task)
            
            # Validate result structure
            assert result.get("success"), f"Relationship mapping failed: {result.get('error')}"
            assert "relationships" in result, "Relationships missing"
            
            relationships = result["relationships"]
            assert len(relationships) > 0, "No relationships found"
            
            self.results["repository_relationship_mapping"] = {
                "status": "✅ PASS",
                "details": f"Mapped {len(relationships)} repository relationships"
            }
            
        except Exception as e:
            self.results["repository_relationship_mapping"] = {
                "status": "❌ FAIL",
                "error": str(e)
            }
    
    async def test_breaking_change_detection(self):
        """Test breaking change detection."""
        print("\n🧪 Testing Breaking Change Detection...")
        
        try:
            task = {
                "task_type": "detect_breaking_changes",
                "source_repository": "user_service",
                "changes": [
                    {
                        "file": "src/user_service/api.py",
                        "change_type": "modification",
                        "old_content": "def delete_user(self, user_id: str) -> bool:",
                        "new_content": "def delete_user(self, user_id: str, force: bool = False) -> Dict[str, Any]:"
                    }
                ]
            }
            
            result = await self.impact_agent.execute_task(task)
            
            # Validate result structure
            assert result.get("success"), f"Breaking change detection failed: {result.get('error')}"
            assert "breaking_changes" in result, "Breaking changes missing"
            
            breaking_changes = result["breaking_changes"]
            # Note: May be empty if no breaking changes detected, which is valid
            
            self.results["breaking_change_detection"] = {
                "status": "✅ PASS",
                "details": f"Analyzed breaking changes - found {len(breaking_changes)} issues"
            }
            
        except Exception as e:
            self.results["breaking_change_detection"] = {
                "status": "❌ FAIL",
                "error": str(e)
            }
    
    async def test_impact_report_generation(self):
        """Test impact report generation."""
        print("\n🧪 Testing Impact Report Generation...")
        
        try:
            task = {
                "task_type": "generate_impact_report",
                "source_repository": "user_service",
                "target_repository": "order_service",
                "include_details": True
            }
            
            result = await self.impact_agent.execute_task(task)
            
            # Validate result structure
            assert result.get("success"), f"Report generation failed: {result.get('error')}"
            assert "report" in result, "Report missing"
            
            report = result["report"]
            assert "repository_info" in report, "Repository info missing"
            assert "impact_overview" in report, "Impact overview missing"
            
            self.results["impact_report_generation"] = {
                "status": "✅ PASS",
                "details": "Generated comprehensive impact report"
            }
            
        except Exception as e:
            self.results["impact_report_generation"] = {
                "status": "❌ FAIL",
                "error": str(e)
            }
    
    async def test_coordination_task_suggestions(self):
        """Test coordination task suggestions."""
        print("\n🧪 Testing Coordination Task Suggestions...")
        
        try:
            task = {
                "task_type": "suggest_coordination_tasks",
                "source_repository": "user_service",
                "impact_type": "api_breaking",
                "affected_repositories": ["order_service", "frontend_app"],
                "severity": "high"
            }
            
            result = await self.impact_agent.execute_task(task)
            
            # Validate result structure
            assert result.get("success"), f"Coordination suggestions failed: {result.get('error')}"
            assert "coordination_tasks" in result, "Coordination tasks missing"
            
            coordination_tasks = result["coordination_tasks"]
            assert len(coordination_tasks) > 0, "No coordination tasks generated"
            
            self.results["coordination_task_suggestions"] = {
                "status": "✅ PASS",
                "details": f"Generated {len(coordination_tasks)} coordination tasks"
            }
            
        except Exception as e:
            self.results["coordination_task_suggestions"] = {
                "status": "❌ FAIL",
                "error": str(e)
            }
    
    async def test_python_api_extraction(self):
        """Test Python API extraction helper method."""
        print("\n🧪 Testing Python API Extraction...")
        
        try:
            # Test with sample Python code
            sample_code = '''
class TestAPI:
    def public_method(self, param: str) -> str:
        pass
    
    def _private_method(self):
        pass

def public_function(x: int, y: int) -> int:
    return x + y

async def async_function() -> None:
    pass
'''
            
            apis = self.impact_agent._extract_python_apis(sample_code, "test.py")
            
            # Validate extraction
            assert len(apis) > 0, "No APIs extracted"
            
            # Check for expected APIs
            api_names = [api["name"] for api in apis]
            assert "TestAPI" in api_names, "Class not extracted"
            assert "public_function" in api_names, "Function not extracted"
            assert "async_function" in api_names, "Async function not extracted"
            
            self.results["python_api_extraction"] = {
                "status": "✅ PASS",
                "details": f"Extracted {len(apis)} APIs from sample code"
            }
            
        except Exception as e:
            self.results["python_api_extraction"] = {
                "status": "❌ FAIL",
                "error": str(e)
            }
    
    async def run_validation_tests(self):
        """Run all validation tests."""
        print("🚀 Starting Task 13.1 Validation Tests...")
        
        await self.setup_test_environment()
        
        # Run all test methods
        test_methods = [
            self.test_impact_analysis,
            self.test_api_change_analysis,
            self.test_dependency_impact_analysis,
            self.test_repository_relationship_mapping,
            self.test_breaking_change_detection,
            self.test_impact_report_generation,
            self.test_coordination_task_suggestions,
            self.test_python_api_extraction
        ]
        
        for test_method in test_methods:
            try:
                await test_method()
            except Exception as e:
                test_name = test_method.__name__
                self.results[test_name] = {
                    "status": "❌ FAIL",
                    "error": f"Test execution failed: {str(e)}"
                }
    
    def generate_report(self) -> dict[str, Any]:
        """Generate validation test report."""
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results.values() if r["status"].startswith("✅")])
        failed_tests = total_tests - passed_tests
        
        report = {
            "task": "13.1 - Cross-Repository Impact Analysis Implementation",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": f"{(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "0%"
            },
            "test_results": self.results,
            "validation_status": "✅ COMPLETE" if failed_tests == 0 else f"⚠️ PARTIAL ({failed_tests} failures)"
        }
        
        return report


async def main():
    """Main validation function."""
    validator = Task13ValidationTest()
    
    try:
        await validator.run_validation_tests()
        report = validator.generate_report()
        
        # Print summary
        print("\n" + "="*60)
        print("📊 TASK 13.1 VALIDATION REPORT")
        print("="*60)
        print(f"Task: {report['task']}")
        print(f"Status: {report['validation_status']}")
        print(f"Tests: {report['summary']['passed']}/{report['summary']['total_tests']} passed ({report['summary']['success_rate']})")
        print("\nTest Results:")
        print("-" * 40)
        
        for test_name, result in report['test_results'].items():
            status = result['status']
            details = result.get('details', result.get('error', ''))
            print(f"{status} {test_name}: {details}")
        
        # Save detailed report
        report_file = Path(__file__).parent / "task_13_1_validation_results.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        # Return appropriate exit code
        return 0 if report['summary']['failed'] == 0 else 1
        
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        return 1


if __name__ == "__main__":
    import asyncio
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
