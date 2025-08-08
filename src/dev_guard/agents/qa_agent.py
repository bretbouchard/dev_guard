"""QA Test agent for DevGuard - handles testing and quality assurance tasks."""

import json
import logging
import os
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..memory.shared_memory import AgentState
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class QATestAgent(BaseAgent):
    """
    QA Test agent responsible for:
    - Running automated tests
    - Code quality analysis 
    - Test generation and coverage analysis
    - Performance testing and benchmarking
    - Security vulnerability scanning
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_provider = kwargs.get('llm_provider')
        
        # QA-specific configuration
        self.test_frameworks = ["pytest", "unittest", "nose2"]
        self.coverage_threshold = 80
        
        # Goose CLI integration
        self.goose_path = self._find_goose_executable()
        self.session_id = None
        
        # TDD and test generation configuration
        self.tdd_enabled = True
        self.test_patterns = {
            "unit": "test_*.py",
            "integration": "test_integration_*.py",
            "e2e": "test_e2e_*.py",
            "performance": "test_perf_*.py"
        }
        
        # Test templates and patterns
        self.test_templates = {
            "pytest": self._get_pytest_template(),
            "unittest": self._get_unittest_template(),
            "bdd": self._get_bdd_template()
        }
        
        # TDD workflow state
        self.tdd_cycle_state = "red"  # red, green, refactor
        
    async def execute(self, state: Any) -> Any:
        """Execute the QA test agent's main logic."""
        # Extract task from state
        if isinstance(state, dict):
            task = state
        else:
            task = {"type": "run_tests", "description": str(state)}
        
        return await self.execute_task(task)
    
    def _find_goose_executable(self) -> str | None:
        """Find Goose CLI executable in the system."""
        return shutil.which("goose")
    
    async def _run_goose_command(self, args: list[str], input_text: str | None = None, 
                                 cwd: str | None = None) -> dict[str, Any]:
        """Run a Goose CLI command with enhanced logging."""
        if not self.goose_path:
            return {
                "success": False,
                "error": "Goose CLI not found. Please install Goose to use this feature.",
                "command": " ".join(args),
                "return_code": -1
            }
        
        try:
            start_time = datetime.now(UTC)
            cmd = [self.goose_path] + args
            working_dir = cwd or self.working_directory or os.getcwd()
            
            # Execute command
            process = subprocess.run(
                cmd,
                input=input_text,
                text=True,
                capture_output=True,
                cwd=working_dir,
                timeout=300  # 5 minute timeout
            )
            
            end_time = datetime.now(UTC)
            duration = (end_time - start_time).total_seconds()
            
            return {
                "success": process.returncode == 0,
                "command": " ".join(cmd),
                "output": process.stdout,
                "error": process.stderr if process.stderr else None,
                "return_code": process.returncode,
                "session_id": self.session_id,
                "execution_time": duration,
                "working_directory": working_dir
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Goose command timed out after 5 minutes",
                "command": " ".join(args),
                "return_code": -1
            }
        except Exception as e:
            self.logger.error(f"Error running Goose command: {e}")
            return {
                "success": False,
                "error": str(e),
                "command": " ".join(args),
                "return_code": -1
            }
    
    async def execute_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a QA/testing task.
        
        Args:
            task: Task dictionary with type, description, and parameters
            
        Returns:
            Execution result with test results and quality metrics
        """
        try:
            self.logger.info(f"QA Test agent executing task: {task.get('type', 'unknown')}")
            
            # Update agent state
            self._update_state("working", task.get("task_id"))
            
            task_type = task.get("type", "")
            
            if task_type == "run_tests":
                result = await self._run_tests(task)
            elif task_type == "generate_tests":
                result = await self._generate_tests(task)
            elif task_type == "analyze_coverage":
                result = await self._analyze_coverage(task)
            elif task_type == "quality_check":
                result = await self._quality_check(task)
            elif task_type == "security_scan":
                result = await self._security_scan(task)
            elif task_type == "performance_test":
                result = await self._performance_test(task)
            elif task_type == "tdd_cycle":
                result = await self._run_tdd_cycle(task)
            elif task_type == "tdd_red":
                result = await self._tdd_red_phase(
                    task.get("target_file", ""), 
                    task.get("requirements", ""), 
                    task.get("test_type", "unit")
                )
            elif task_type == "tdd_green":
                result = await self._tdd_green_phase(
                    task.get("target_file", ""), 
                    task.get("test_file", "")
                )
            elif task_type == "tdd_refactor":
                result = await self._tdd_refactor_phase(
                    task.get("target_file", ""), 
                    task.get("test_file", "")
                )
            elif task_type == "generate_bdd_tests":
                result = await self._generate_behavior_driven_tests(task)
            elif task_type == "goose_fix":
                result = await self._goose_fix_command(task)
            elif task_type == "goose_write_tests":
                result = await self._goose_write_tests_command(task)
            elif task_type == "automated_qa_pipeline":
                result = await self._run_automated_qa_pipeline(task)
            else:
                result = await self._generic_qa_task(task)
            
            self._update_state("idle")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in QA test task execution: {e}")
            self._update_state("error", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "test_results": {}
            }
    
    async def _run_tests(self, task: dict[str, Any]) -> dict[str, Any]:
        """Run automated tests with comprehensive analysis and reporting."""
        try:
            test_path = task.get("test_path", "tests/")
            test_framework = task.get("framework", "pytest")
            test_pattern = task.get("pattern", "test_*.py")
            include_coverage = task.get("coverage", True)
            
            # Enhanced test execution with multiple phases
            execution_results = {
                "test_execution": {},
                "coverage_analysis": {},
                "performance_metrics": {},
                "quality_assessment": {},
                "summary": {}
            }
            
            # Phase 1: Execute tests with coverage
            test_cmd_result = await self._execute_test_suite(
                test_path, test_framework, test_pattern, include_coverage
            )
            execution_results["test_execution"] = test_cmd_result
            
            if not test_cmd_result["success"]:
                return {
                    "success": False,
                    "error": "Test execution failed",
                    "execution_results": execution_results
                }
            
            # Phase 2: Analyze coverage if requested
            if include_coverage:
                coverage_result = await self._detailed_coverage_analysis(test_path)
                execution_results["coverage_analysis"] = coverage_result
            
            # Phase 3: Performance metrics
            performance_result = await self._analyze_test_performance(test_cmd_result)
            execution_results["performance_metrics"] = performance_result
            
            # Phase 4: Quality assessment
            quality_result = await self._assess_test_quality(test_path)
            execution_results["quality_assessment"] = quality_result
            
            # Phase 5: Generate comprehensive summary
            summary = self._generate_test_summary(execution_results)
            execution_results["summary"] = summary
            
            return {
                "success": True,
                "test_results": execution_results,
                "recommendations": summary.get("recommendations", []),
                "overall_score": summary.get("overall_score", 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error running tests: {e}")
            return {
                "success": False,
                "error": str(e),
                "test_results": {}
            }
    
    async def _execute_test_suite(self, test_path: str, framework: str, 
                                  pattern: str, include_coverage: bool) -> dict[str, Any]:
        """Execute test suite with the specified framework."""
        try:
            # Build test command based on framework
            if framework == "pytest":
                cmd = ["python", "-m", "pytest", test_path, "-v", "--tb=short", 
                       "--junitxml=test-results.xml"]
                if include_coverage:
                    cmd.extend(["--cov=src", "--cov-report=json", "--cov-report=term-missing"])
            elif framework == "unittest":
                cmd = ["python", "-m", "unittest", "discover", "-s", test_path, "-p", pattern, "-v"]
            else:
                cmd = ["python", "-m", "pytest", test_path, "-v"]
            
            # Execute tests with timeout
            start_time = datetime.now(UTC)
            result = await self._run_command(cmd, timeout=600)  # 10 minute timeout
            end_time = datetime.now(UTC)
            
            # Parse test results
            test_results = self._parse_test_output(result["output"], framework)
            
            return {
                "success": result["success"],
                "command": " ".join(cmd),
                "execution_time": (end_time - start_time).total_seconds(),
                "test_results": test_results,
                "output": result["output"][:2000],  # Truncate for storage
                "return_code": result.get("return_code", -1)
            }
            
        except Exception as e:
            self.logger.error(f"Error executing test suite: {e}")
            return {
                "success": False,
                "error": str(e),
                "test_results": {}
            }
    
    async def _detailed_coverage_analysis(self, test_path: str) -> dict[str, Any]:
        """Perform detailed coverage analysis with recommendations."""
        try:
            # Read coverage JSON report if available
            coverage_file = Path("coverage.json")
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.loads(f.read())
                
                analysis = {
                    "total_coverage": coverage_data.get("totals", {}).get("percent_covered", 0),
                    "files_coverage": {},
                    "uncovered_lines": {},
                    "missing_coverage": [],
                    "recommendations": []
                }
                
                # Analyze per-file coverage
                for file_path, file_data in coverage_data.get("files", {}).items():
                    file_coverage = (file_data["summary"]["covered_lines"] / 
                                   max(file_data["summary"]["num_statements"], 1)) * 100
                    analysis["files_coverage"][file_path] = file_coverage
                    
                    # Track uncovered lines
                    missing_lines = file_data.get("missing_lines", [])
                    if missing_lines:
                        analysis["uncovered_lines"][file_path] = missing_lines
                    
                    # Generate recommendations for low coverage files
                    if file_coverage < self.coverage_threshold:
                        analysis["missing_coverage"].append({
                            "file": file_path,
                            "coverage": file_coverage,
                            "missing_lines": len(missing_lines)
                        })
                
                # Generate overall recommendations
                if analysis["total_coverage"] < self.coverage_threshold:
                    analysis["recommendations"].append(
                        f"Overall coverage {analysis['total_coverage']:.1f}% below target {self.coverage_threshold}%"
                    )
                
                for low_cov_file in analysis["missing_coverage"][:5]:  # Top 5 files needing attention
                    analysis["recommendations"].append(
                        f"Improve coverage for {low_cov_file['file']}: {low_cov_file['coverage']:.1f}%"
                    )
                
                return {
                    "success": True,
                    "analysis": analysis
                }
            else:
                return {
                    "success": False,
                    "error": "Coverage report not found",
                    "analysis": {}
                }
                
        except Exception as e:
            self.logger.error(f"Error in coverage analysis: {e}")
            return {
                "success": False,
                "error": str(e),
                "analysis": {}
            }
    
    async def _analyze_test_performance(self, test_result: dict[str, Any]) -> dict[str, Any]:
        """Analyze test execution performance."""
        try:
            execution_time = test_result.get("execution_time", 0)
            test_results = test_result.get("test_results", {})
            
            performance_analysis = {
                "total_execution_time": execution_time,
                "average_test_time": 0,
                "slow_tests": [],
                "performance_score": 100,
                "recommendations": []
            }
            
            # Calculate average test time
            total_tests = test_results.get("tests_run", 0)
            if total_tests > 0:
                performance_analysis["average_test_time"] = execution_time / total_tests
            
            # Performance scoring
            if execution_time > 300:  # 5 minutes
                performance_analysis["performance_score"] = 60
                performance_analysis["recommendations"].append(
                    "Test suite takes over 5 minutes - consider optimization"
                )
            elif execution_time > 60:  # 1 minute
                performance_analysis["performance_score"] = 80
                performance_analysis["recommendations"].append(
                    "Consider optimizing slower tests for better developer experience"
                )
            
            # Identify slow tests from output (if available)
            output = test_result.get("output", "")
            slow_test_pattern = r'(\S+::\S+)\s+.*\[(\d+)s\]'
            slow_matches = re.findall(slow_test_pattern, output)
            
            for test_name, duration in slow_matches:
                if float(duration) > 5:  # Tests taking more than 5 seconds
                    performance_analysis["slow_tests"].append({
                        "test": test_name,
                        "duration": float(duration)
                    })
            
            return {
                "success": True,
                "performance": performance_analysis
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing test performance: {e}")
            return {
                "success": False,
                "error": str(e),
                "performance": {}
            }
    
    async def _assess_test_quality(self, test_path: str) -> dict[str, Any]:
        """Assess the quality of the test suite."""
        try:
            quality_assessment = {
                "test_organization": {"score": 0, "issues": []},
                "test_completeness": {"score": 0, "issues": []},
                "test_maintainability": {"score": 0, "issues": []},
                "best_practices": {"score": 0, "issues": []},
                "overall_score": 0
            }
            
            # Analyze test files
            test_files = list(Path(test_path).rglob("test_*.py")) + list(Path(test_path).rglob("*_test.py"))
            
            if not test_files:
                return {
                    "success": False,
                    "error": "No test files found",
                    "quality": {}
                }
            
            total_tests = 0
            fixture_count = 0
            mock_usage = 0
            docstring_count = 0
            
            for test_file in test_files:
                try:
                    with open(test_file, encoding='utf-8') as f:
                        content = f.read()
                    
                    # Count test functions
                    total_tests += len(re.findall(r'def test_\w+', content))
                    
                    # Check for fixtures
                    if 'pytest.fixture' in content or '@fixture' in content:
                        fixture_count += 1
                    
                    # Check for mock usage
                    if any(mock_term in content for mock_term in ['Mock', 'patch', 'mock']):
                        mock_usage += 1
                    
                    # Check for test docstrings
                    docstring_count += len(re.findall(r'def test_\w+.*?:\s*""".*?"""', content, re.DOTALL))
                    
                except Exception as e:
                    self.logger.warning(f"Error analyzing {test_file}: {e}")
            
            # Score test organization
            org_score = min(100, len(test_files) * 20)  # More files = better organization
            quality_assessment["test_organization"]["score"] = org_score
            
            # Score test completeness
            completeness_score = min(100, total_tests * 5)  # More tests = better completeness
            quality_assessment["test_completeness"]["score"] = completeness_score
            
            # Score maintainability (fixtures, mocks, docstrings)
            maintainability_factors = [
                fixture_count > 0,  # Has fixtures
                mock_usage > 0,     # Uses mocks
                docstring_count / max(total_tests, 1) > 0.5  # >50% tests have docstrings
            ]
            maintainability_score = sum(maintainability_factors) * 33
            quality_assessment["test_maintainability"]["score"] = maintainability_score
            
            # Score best practices
            practices_score = 100  # Start with perfect score
            if fixture_count == 0:
                quality_assessment["best_practices"]["issues"].append("No test fixtures found")
                practices_score -= 25
            if mock_usage == 0:
                quality_assessment["best_practices"]["issues"].append("No mocking detected")
                practices_score -= 25
            if docstring_count / max(total_tests, 1) < 0.3:
                quality_assessment["best_practices"]["issues"].append("Few test docstrings")
                practices_score -= 25
            
            quality_assessment["best_practices"]["score"] = max(0, practices_score)
            
            # Calculate overall score
            scores = [
                quality_assessment["test_organization"]["score"],
                quality_assessment["test_completeness"]["score"],
                quality_assessment["test_maintainability"]["score"],
                quality_assessment["best_practices"]["score"]
            ]
            quality_assessment["overall_score"] = sum(scores) / len(scores)
            
            return {
                "success": True,
                "quality": quality_assessment,
                "statistics": {
                    "total_test_files": len(test_files),
                    "total_tests": total_tests,
                    "fixture_files": fixture_count,
                    "mock_usage_files": mock_usage,
                    "documented_tests": docstring_count
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing test quality: {e}")
            return {
                "success": False,
                "error": str(e),
                "quality": {}
            }
    
    def _generate_test_summary(self, execution_results: dict[str, Any]) -> dict[str, Any]:
        """Generate comprehensive test execution summary."""
        try:
            test_exec = execution_results.get("test_execution", {})
            coverage_analysis = execution_results.get("coverage_analysis", {})
            performance = execution_results.get("performance_metrics", {})
            quality = execution_results.get("quality_assessment", {})
            
            # Extract key metrics
            tests_passed = test_exec.get("test_results", {}).get("passed", 0)
            tests_failed = test_exec.get("test_results", {}).get("failed", 0)
            tests_total = tests_passed + tests_failed
            
            coverage_percent = coverage_analysis.get("analysis", {}).get("total_coverage", 0)
            execution_time = performance.get("performance", {}).get("total_execution_time", 0)
            quality_score = quality.get("quality", {}).get("overall_score", 0)
            
            # Calculate overall score
            test_success_rate = (tests_passed / max(tests_total, 1)) * 100
            coverage_score = min(coverage_percent, 100)
            performance_score = performance.get("performance", {}).get("performance_score", 100)
            
            overall_score = (test_success_rate * 0.4 + coverage_score * 0.3 + 
                           performance_score * 0.15 + quality_score * 0.15)
            
            # Generate recommendations
            recommendations = []
            
            if test_success_rate < 100:
                recommendations.append(f"Fix {tests_failed} failing tests")
            
            if coverage_percent < self.coverage_threshold:
                recommendations.append(f"Improve test coverage from {coverage_percent:.1f}% to {self.coverage_threshold}%")
            
            if execution_time > 120:
                recommendations.append("Optimize test execution time")
            
            if quality_score < 80:
                recommendations.append("Improve test quality and maintainability")
            
            # Add specific recommendations from sub-analyses
            recommendations.extend(coverage_analysis.get("analysis", {}).get("recommendations", [])[:3])
            recommendations.extend(performance.get("performance", {}).get("recommendations", [])[:2])
            
            return {
                "overall_score": round(overall_score, 1),
                "test_success_rate": round(test_success_rate, 1),
                "coverage_percentage": round(coverage_percent, 1),
                "execution_time_seconds": round(execution_time, 2),
                "quality_score": round(quality_score, 1),
                "recommendations": recommendations[:8],  # Top 8 recommendations
                "status": "EXCELLENT" if overall_score >= 90 else 
                         "GOOD" if overall_score >= 75 else 
                         "NEEDS_IMPROVEMENT" if overall_score >= 60 else "POOR"
            }
            
        except Exception as e:
            self.logger.error(f"Error generating test summary: {e}")
            return {
                "overall_score": 0,
                "status": "ERROR",
                "recommendations": ["Fix test execution issues"]
            }
    
    def get_capabilities(self) -> list[str]:
        """Get QA Test agent capabilities."""
        return [
            "automated_testing",
            "test_generation",
            "coverage_analysis",
            "quality_assessment",
            "performance_testing",
            "security_scanning",
            "test_execution",
            "test_validation",
            "goose_test_generation",
            "comprehensive_reporting",
            "test_optimization",
            "framework_support_pytest",
            "framework_support_unittest",
            "ci_cd_integration",
            "test_maintenance",
            "tdd_support",
            "test_driven_development",
            "red_green_refactor",
            "behavior_driven_development",
            "test_templates",
            "advanced_test_patterns",
            "goose_fix_command",
            "goose_write_tests",
            "automated_qa_pipeline",
            "code_repair_automation",
            "intelligent_bug_fixing"
        ]
    
    async def _generate_tests(self, task: dict[str, Any]) -> dict[str, Any]:
        """Generate tests for specified code using Goose CLI."""
        try:
            target_file = task.get("target_file", "")
            if not target_file:
                return {
                    "success": False,
                    "error": "No target file specified for test generation"
                }
            
            # Check if target file exists
            target_path = Path(target_file)
            if not target_path.exists():
                return {
                    "success": False,
                    "error": f"Target file does not exist: {target_file}"
                }

            # Use Goose CLI for test generation if available
            if self.goose_path:
                result = await self._generate_tests_with_goose(target_file, task)
                if result["success"]:
                    return result
                else:
                    self.logger.warning(f"Goose test generation failed: {result.get('error')}")
            
            # Fallback to LLM or template generation
            with open(target_path, encoding='utf-8') as f:
                code_content = f.read()
            
            if self.llm_provider:
                generated_tests = await self._llm_generate_tests(code_content, target_file)
            else:
                generated_tests = self._template_generate_tests(target_file)
            
            # Save generated tests
            test_file = self._get_test_file_path(target_file)
            test_dir = test_file.parent
            test_dir.mkdir(parents=True, exist_ok=True)
            
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(generated_tests)
            
            return {
                "success": True,
                "test_file": str(test_file),
                "tests_generated": True,
                "method": "fallback_generation",
                "message": f"Tests generated for {target_file}"
            }
            
        except Exception as e:
            self.logger.error(f"Error generating tests: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _generate_tests_with_goose(self, target_file: str, task: dict[str, Any]) -> dict[str, Any]:
        """Generate comprehensive tests using Goose CLI."""
        try:
            # Start a Goose session focused on test generation
            test_type = task.get("test_type", "comprehensive")
            coverage_target = task.get("coverage_target", 90)
            
            # Create test prompt based on requirements
            test_prompt = self._create_test_generation_prompt(target_file, test_type, coverage_target)
            
            # Execute Goose command for test generation
            goose_result = await self._run_goose_command(
                ["session", "start", "--file", target_file],
                input_text=test_prompt
            )
            
            if not goose_result["success"]:
                return {
                    "success": False,
                    "error": f"Goose test generation failed: {goose_result.get('error')}",
                    "goose_output": goose_result
                }
            
            # Check if test files were created
            test_file = self._get_test_file_path(target_file)
            if test_file.exists():
                # Verify test quality and completeness
                test_quality = await self._validate_generated_tests(str(test_file), target_file)
                
                return {
                    "success": True,
                    "test_file": str(test_file),
                    "method": "goose_cli",
                    "test_quality": test_quality,
                    "goose_session_id": goose_result.get("session_id"),
                    "message": f"Comprehensive tests generated using Goose CLI for {target_file}"
                }
            else:
                # Try to extract generated test code from Goose output
                test_code = self._extract_test_code_from_output(goose_result["output"])
                if test_code:
                    test_dir = test_file.parent
                    test_dir.mkdir(parents=True, exist_ok=True)
                    
                    with open(test_file, 'w', encoding='utf-8') as f:
                        f.write(test_code)
                    
                    return {
                        "success": True,
                        "test_file": str(test_file),
                        "method": "goose_cli_extracted",
                        "message": f"Tests extracted from Goose output for {target_file}"
                    }
                else:
                    return {
                        "success": False,
                        "error": "Goose CLI did not generate recognizable test files",
                        "goose_output": goose_result["output"][:500]
                    }
            
        except Exception as e:
            self.logger.error(f"Error in Goose test generation: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_test_generation_prompt(self, target_file: str, test_type: str, coverage_target: int) -> str:
        """Create a comprehensive prompt for Goose test generation."""
        base_prompt = f"""Please generate comprehensive {test_type} tests for the code in {target_file}.

Requirements:
- Target test coverage: {coverage_target}%
- Include unit tests for all public methods and functions
- Add edge case testing (boundary conditions, error cases)
- Include integration tests where appropriate
- Use pytest framework with proper fixtures
- Add docstrings explaining test purpose
- Mock external dependencies appropriately
- Include parameterized tests for multiple input scenarios

Test Categories to Include:
1. Happy path tests (normal operation)
2. Edge case tests (boundary values, empty inputs)
3. Error handling tests (exceptions, invalid inputs)
4. Performance tests (if applicable)
5. Security tests (if applicable)

Please create well-structured, maintainable test code that follows Python testing best practices."""

        return base_prompt
    
    async def _validate_generated_tests(self, test_file: str, target_file: str) -> dict[str, Any]:
        """Validate the quality and completeness of generated tests."""
        try:
            validation_results = {
                "syntax_valid": False,
                "test_count": 0,
                "has_fixtures": False,
                "has_mocks": False,
                "has_edge_cases": False,
                "estimated_coverage": 0,
                "quality_score": 0
            }
            
            # Check syntax validity
            with open(test_file, encoding='utf-8') as f:
                test_content = f.read()
            
            try:
                compile(test_content, test_file, 'exec')
                validation_results["syntax_valid"] = True
            except SyntaxError:
                validation_results["syntax_valid"] = False
                return validation_results
            
            # Analyze test content
            import ast
            try:
                tree = ast.parse(test_content)
                
                # Count test functions
                test_count = len([node for node in ast.walk(tree) 
                                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_')])
                validation_results["test_count"] = test_count
                
                # Check for fixtures
                validation_results["has_fixtures"] = 'pytest.fixture' in test_content or '@fixture' in test_content
                
                # Check for mocks
                validation_results["has_mocks"] = any(mock_lib in test_content 
                                                   for mock_lib in ['mock', 'Mock', 'patch', 'MagicMock'])
                
                # Check for edge case patterns
                edge_case_patterns = ['edge', 'boundary', 'empty', 'none', 'null', 'exception', 'error']
                validation_results["has_edge_cases"] = any(pattern in test_content.lower() 
                                                         for pattern in edge_case_patterns)
                
                # Estimate coverage based on heuristics
                if test_count > 0:
                    base_coverage = min(test_count * 10, 70)  # Base coverage
                    if validation_results["has_fixtures"]: base_coverage += 10
                    if validation_results["has_mocks"]: base_coverage += 10
                    if validation_results["has_edge_cases"]: base_coverage += 10
                    validation_results["estimated_coverage"] = min(base_coverage, 95)
                
                # Calculate quality score
                quality_factors = [
                    validation_results["syntax_valid"],
                    test_count >= 3,
                    validation_results["has_fixtures"],
                    validation_results["has_mocks"],
                    validation_results["has_edge_cases"]
                ]
                validation_results["quality_score"] = sum(quality_factors) * 20
                
            except Exception as e:
                self.logger.warning(f"Error analyzing test content: {e}")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error validating tests: {e}")
            return {"syntax_valid": False, "quality_score": 0}
    
    def _extract_test_code_from_output(self, goose_output: str) -> str | None:
        """Extract Python test code from Goose CLI output."""
        try:
            # Look for Python code blocks in the output
            code_pattern = r'```python\n(.*?)\n```'
            matches = re.findall(code_pattern, goose_output, re.DOTALL)
            
            if matches:
                # Find the longest code block (likely the main test file)
                test_code = max(matches, key=len)
                
                # Validate it looks like test code
                if 'def test_' in test_code or 'import pytest' in test_code:
                    return test_code
            
            # Alternative: look for test functions directly
            test_function_pattern = r'(def test_.*?(?=\ndef|\Z))'
            test_functions = re.findall(test_function_pattern, goose_output, re.DOTALL)
            
            if test_functions:
                # Combine test functions with necessary imports
                imports = "import pytest\nfrom unittest.mock import Mock, patch\n\n"
                return imports + '\n\n'.join(test_functions)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting test code: {e}")
            return None
    
    async def _analyze_coverage(self, task: dict[str, Any]) -> dict[str, Any]:
        """Analyze test coverage."""
        try:
            source_path = task.get("source_path", "src/")
            test_path = task.get("test_path", "tests/")
            
            # Run coverage analysis
            cmd = [
                "python", "-m", "pytest",
                test_path,
                f"--cov={source_path}",
                "--cov-report=json",
                "--cov-report=term-missing",
                "-q"
            ]
            
            result = await self._run_command(cmd)
            
            # Parse coverage results
            coverage_data = self._parse_coverage_output(result["output"])
            
            # Generate coverage report
            coverage_report = {
                "total_coverage": coverage_data.get("total_coverage", 0),
                "files_coverage": coverage_data.get("files", {}),
                "threshold_met": coverage_data.get("total_coverage", 0) >= self.coverage_threshold,
                "missing_lines": coverage_data.get("missing_lines", {}),
                "recommendations": []
            }
            
            # Add recommendations
            if coverage_report["total_coverage"] < self.coverage_threshold:
                coverage_report["recommendations"].append(
                    f"Coverage {coverage_report['total_coverage']}% is below threshold {self.coverage_threshold}%"
                )
            
            # Find files with low coverage
            for file_path, file_coverage in coverage_data.get("files", {}).items():
                if file_coverage < 70:
                    coverage_report["recommendations"].append(
                        f"Consider adding tests for {file_path} (current: {file_coverage}%)"
                    )
            
            return {
                "success": result["success"],
                "coverage_report": coverage_report,
                "raw_output": result["output"][:500]
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing coverage: {e}")
            return {
                "success": False,
                "error": str(e),
                "coverage_report": {}
            }
    
    async def _quality_check(self, task: dict[str, Any]) -> dict[str, Any]:
        """Perform code quality checks."""
        try:
            target_path = task.get("target_path", "src/")
            checks = task.get("checks", ["lint", "style", "complexity"])
            
            quality_results = {
                "overall_score": 0,
                "checks_performed": [],
                "issues_found": [],
                "recommendations": []
            }
            
            # Run linting if requested
            if "lint" in checks:
                lint_result = await self._run_lint_check(target_path)
                quality_results["checks_performed"].append("lint")
                if lint_result.get("issues"):
                    quality_results["issues_found"].extend(lint_result["issues"])
            
            # Run style checks if requested
            if "style" in checks:
                style_result = await self._run_style_check(target_path)
                quality_results["checks_performed"].append("style")
                if style_result.get("issues"):
                    quality_results["issues_found"].extend(style_result["issues"])
            
            # Calculate overall score
            total_issues = len(quality_results["issues_found"])
            if total_issues == 0:
                quality_results["overall_score"] = 100
            elif total_issues < 5:
                quality_results["overall_score"] = 80
            elif total_issues < 10:
                quality_results["overall_score"] = 60
            else:
                quality_results["overall_score"] = 40
            
            # Add recommendations
            if total_issues > 0:
                quality_results["recommendations"].append(
                    f"Found {total_issues} code quality issues that should be addressed"
                )
            
            return {
                "success": True,
                "quality_results": quality_results
            }
            
        except Exception as e:
            self.logger.error(f"Error in quality check: {e}")
            return {
                "success": False,
                "error": str(e),
                "quality_results": {}
            }
    
    async def _security_scan(self, task: dict[str, Any]) -> dict[str, Any]:
        """Perform security vulnerability scanning."""
        try:
            target_path = task.get("target_path", ".")
            
            security_results = {
                "vulnerabilities_found": 0,
                "security_score": 100,
                "issues": [],
                "recommendations": []
            }
            
            # Simple security checks
            security_issues = []
            
            # Check for common security issues in Python files
            python_files = list(Path(target_path).rglob("*.py"))
            for file_path in python_files:
                try:
                    with open(file_path, encoding='utf-8') as f:
                        content = f.read()
                        issues = self._check_security_patterns(content, str(file_path))
                        security_issues.extend(issues)
                except Exception as e:
                    self.logger.warning(f"Could not scan {file_path}: {e}")
            
            security_results["issues"] = security_issues
            security_results["vulnerabilities_found"] = len(security_issues)
            
            # Calculate security score
            if len(security_issues) == 0:
                security_results["security_score"] = 100
            elif len(security_issues) < 3:
                security_results["security_score"] = 80
            else:
                security_results["security_score"] = 60
            
            # Add recommendations
            if security_issues:
                security_results["recommendations"].append(
                    "Review and address identified security issues"
                )
            
            return {
                "success": True,
                "security_results": security_results
            }
            
        except Exception as e:
            self.logger.error(f"Error in security scan: {e}")
            return {
                "success": False,
                "error": str(e),
                "security_results": {}
            }
    
    async def _performance_test(self, task: dict[str, Any]) -> dict[str, Any]:
        """Run performance tests."""
        try:
            # Simple performance testing placeholder
            performance_results = {
                "execution_time": 0.0,
                "memory_usage": 0,
                "performance_score": 100,
                "bottlenecks": [],
                "recommendations": []
            }
            
            # This would be expanded with actual performance testing
            # For now, return a placeholder result
            
            return {
                "success": True,
                "performance_results": performance_results,
                "message": "Performance testing not yet fully implemented"
            }
            
        except Exception as e:
            self.logger.error(f"Error in performance test: {e}")
            return {
                "success": False,
                "error": str(e),
                "performance_results": {}
            }
    
    async def _generic_qa_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Handle generic QA tasks."""
        try:
            description = task.get("description", "")
            
            # Basic QA task execution
            result = {
                "success": True,
                "message": f"Processed QA task: {description}",
                "actions_taken": ["analyzed_task", "provided_feedback"],
                "recommendations": ["Consider adding specific test cases"]
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in generic QA task: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _llm_generate_tests(self, code_content: str, target_file: str) -> str:
        """Use LLM to generate comprehensive tests."""
        try:
            prompt = f"""
            Generate comprehensive pytest tests for the following Python code from {target_file}:
            
            ```python
            {code_content}
            ```
            
            Please generate:
            1. Unit tests for all functions and methods
            2. Edge case testing
            3. Error handling tests
            4. Integration tests where applicable
            5. Mock usage for external dependencies
            
            Follow pytest conventions and best practices.
            Include docstrings explaining what each test does.
            """
            
            response = await self.llm_provider.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            
            return response.content
            
        except Exception as e:
            self.logger.error(f"Error generating tests with LLM: {e}")
            return self._template_generate_tests(target_file)
    
    def _template_generate_tests(self, target_file: str) -> str:
        """Generate basic test template."""
        module_name = Path(target_file).stem
        
        return f'''"""Tests for {target_file}."""

import pytest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


class Test{module_name.title().replace('_', '')}:
    """Test class for {module_name} module."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        # TODO: Implement actual tests
        assert True
        
    def test_edge_cases(self):
        """Test edge cases."""
        # TODO: Implement edge case tests
        assert True
        
    def test_error_handling(self):
        """Test error handling."""
        # TODO: Implement error handling tests  
        assert True


if __name__ == "__main__":
    pytest.main([__file__])
'''
    
    def _get_test_file_path(self, target_file: str) -> Path:
        """Get the corresponding test file path."""
        target_path = Path(target_file)
        
        # If target is in src/, put test in tests/
        if "src/" in str(target_path):
            relative_path = target_path.relative_to(target_path.parts[0])  # Remove first part
            test_file = Path("tests") / relative_path.with_name(f"test_{relative_path.name}")
        else:
            test_file = target_path.parent / f"test_{target_path.name}"
        
        return test_file
    
    async def _run_command(self, cmd: list[str]) -> dict[str, Any]:
        """Run a command and return results."""
        try:
            start_time = datetime.now()
            
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return {
                "success": process.returncode == 0,
                "output": process.stdout + process.stderr,
                "return_code": process.returncode,
                "execution_time": execution_time
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "Command timed out after 300 seconds",
                "return_code": -1,
                "execution_time": 300
            }
        except Exception as e:
            return {
                "success": False,
                "output": f"Command execution error: {str(e)}",
                "return_code": -1,
                "execution_time": 0
            }
    
    def _parse_test_output(self, output: str, framework: str) -> dict[str, Any]:
        """Parse test framework output."""
        results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "test_details": []
        }
        
        if framework == "pytest":
            lines = output.split('\n')
            for line in lines:
                if " PASSED " in line:
                    results["passed"] += 1
                elif " FAILED " in line:
                    results["failed"] += 1
                elif " SKIPPED " in line:
                    results["skipped"] += 1
                elif " ERROR " in line:
                    results["errors"] += 1
                    
                # Look for summary line
                if "failed" in line and "passed" in line:
                    # Try to extract numbers from pytest summary
                    words = line.split()
                    for i, word in enumerate(words):
                        if word == "failed" and i > 0:
                            try:
                                results["failed"] = int(words[i-1])
                            except ValueError:
                                pass
                        elif word == "passed" and i > 0:
                            try:
                                results["passed"] = int(words[i-1])
                            except ValueError:
                                pass
        
        results["total_tests"] = results["passed"] + results["failed"] + results["skipped"] + results["errors"]
        
        return results
    
    def _parse_coverage_output(self, output: str) -> dict[str, Any]:
        """Parse coverage output."""
        coverage_data = {
            "total_coverage": 0,
            "files": {},
            "missing_lines": {}
        }
        
        lines = output.split('\n')
        for line in lines:
            # Look for coverage percentage
            if "%" in line and "TOTAL" in line:
                parts = line.split()
                for part in parts:
                    if part.endswith('%'):
                        try:
                            coverage_data["total_coverage"] = int(part[:-1])
                        except ValueError:
                            pass
        
        return coverage_data
    
    async def _run_lint_check(self, target_path: str) -> dict[str, Any]:
        """Run linting check."""
        try:
            # Try to run flake8 or pylint
            cmd = ["python", "-m", "flake8", target_path, "--count", "--statistics"]
            result = await self._run_command(cmd)
            
            issues = []
            if not result["success"] and result["output"]:
                # Parse lint issues
                lines = result["output"].split('\n')
                for line in lines:
                    if line.strip() and ':' in line:
                        issues.append({
                            "type": "lint",
                            "message": line.strip(),
                            "severity": "warning"
                        })
            
            return {
                "success": True,
                "issues": issues
            }
            
        except Exception as e:
            return {
                "success": False,
                "issues": [{"type": "lint", "message": f"Lint check error: {e}", "severity": "error"}]
            }
    
    async def _run_style_check(self, target_path: str) -> dict[str, Any]:
        """Run style check."""
        try:
            # Simple style check - could be expanded
            issues = []
            
            # Check Python files for basic style issues
            python_files = list(Path(target_path).rglob("*.py"))
            for file_path in python_files[:10]:  # Limit to first 10 files
                try:
                    with open(file_path, encoding='utf-8') as f:
                        content = f.read()
                        file_issues = self._check_style_patterns(content, str(file_path))
                        issues.extend(file_issues)
                except Exception as e:
                    self.logger.warning(f"Could not check style in {file_path}: {e}")
            
            return {
                "success": True,
                "issues": issues
            }
            
        except Exception as e:
            return {
                "success": False,
                "issues": [{"type": "style", "message": f"Style check error: {e}", "severity": "error"}]
            }
    
    def _check_style_patterns(self, content: str, file_path: str) -> list[dict[str, Any]]:
        """Check for basic style issues."""
        issues = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for long lines (> 120 characters)
            if len(line) > 120:
                issues.append({
                    "type": "style",
                    "message": f"{file_path}:{i} - Line too long ({len(line)} > 120 characters)",
                    "severity": "warning"
                })
            
            # Check for trailing whitespace
            if line.rstrip() != line:
                issues.append({
                    "type": "style", 
                    "message": f"{file_path}:{i} - Trailing whitespace",
                    "severity": "info"
                })
        
        return issues
    
    def _check_security_patterns(self, content: str, file_path: str) -> list[dict[str, Any]]:
        """Check for basic security issues."""
        issues = []
        
        # Check for potential security issues
        security_patterns = [
            ("eval(", "Use of eval() can be dangerous"),
            ("exec(", "Use of exec() can be dangerous"), 
            ("shell=True", "shell=True in subprocess can be risky"),
            ("pickle.load", "Pickle deserialization can be unsafe"),
            ("yaml.load", "Use yaml.safe_load instead of yaml.load"),
        ]
        
        for pattern, message in security_patterns:
            if pattern in content:
                issues.append({
                    "type": "security",
                    "message": f"{file_path} - {message}",
                    "severity": "high"
                })
        
        return issues
    
    def _update_state(self, status: str, task_id: str | None = None, error: str | None = None) -> None:
        """Update agent state in shared memory."""
        state = AgentState(
            agent_id=self.agent_id,
            status=status,
            current_task=task_id,
            last_heartbeat=datetime.now(UTC),
            metadata={
                "error": error if error else None,
                "capabilities": ["testing", "quality_assurance", "coverage_analysis"]
            }
        )
        self.shared_memory.update_agent_state(state)
    def get_status(self) -> dict[str, Any]:
        """Get current agent status."""
        return {
            "agent_id": self.agent_id,
            "type": "qa_test",
            "capabilities": self.get_capabilities(),
            "config": {
                "test_frameworks": self.test_frameworks,
                "coverage_threshold": self.coverage_threshold,
                "tdd_enabled": self.tdd_enabled,
                "tdd_cycle_state": self.tdd_cycle_state
            }
        }
    
    # TDD Support Methods
    
    def _get_pytest_template(self) -> str:
        """Get pytest test template."""
        return '''import pytest
from unittest.mock import Mock, patch, MagicMock
from {module_name} import {class_or_function_name}


class Test{ClassName}:
    """Test cases for {ClassName}."""
    
    @pytest.fixture
    def setup_data(self):
        """Setup test data."""
        return {{"test_key": "test_value"}}
    
    def test_{function_name}_basic_functionality(self, setup_data):
        """Test basic functionality of {function_name}."""
        # Arrange
        expected = "expected_value"
        
        # Act
        result = {function_name}(setup_data)
        
        # Assert
        assert result == expected
    
    def test_{function_name}_edge_cases(self):
        """Test edge cases for {function_name}."""
        # Test with None
        with pytest.raises(ValueError):
            {function_name}(None)
            
        # Test with empty input
        result = {function_name}("")
        assert result is not None
    
    @pytest.mark.parametrize("input_value,expected", [
        ("test1", "result1"),
        ("test2", "result2"),
        ("test3", "result3"),
    ])
    def test_{function_name}_parametrized(self, input_value, expected):
        """Parametrized test for {function_name}."""
        result = {function_name}(input_value)
        assert result == expected
    
    def test_{function_name}_with_mocks(self):
        """Test {function_name} with mocks."""
        with patch("{module_name}.dependency") as mock_dep:
            mock_dep.return_value = "mocked_result"
            
            result = {function_name}("test_input")
            
            mock_dep.assert_called_once_with("test_input")
            assert result == "expected_result"
'''

    def _get_unittest_template(self) -> str:
        """Get unittest test template."""
        return '''import unittest
from unittest.mock import Mock, patch, MagicMock
from {module_name} import {class_or_function_name}


class Test{ClassName}(unittest.TestCase):
    """Test cases for {ClassName}."""
    
    def setUp(self):
        """Setup test data."""
        self.test_data = {{"test_key": "test_value"}}
        
    def tearDown(self):
        """Cleanup after tests."""
        pass
    
    def test_{function_name}_basic_functionality(self):
        """Test basic functionality of {function_name}."""
        # Arrange
        expected = "expected_value"
        
        # Act
        result = {function_name}(self.test_data)
        
        # Assert
        self.assertEqual(result, expected)
    
    def test_{function_name}_edge_cases(self):
        """Test edge cases for {function_name}."""
        # Test with None
        with self.assertRaises(ValueError):
            {function_name}(None)
            
        # Test with empty input
        result = {function_name}("")
        self.assertIsNotNone(result)
    
    def test_{function_name}_with_mocks(self):
        """Test {function_name} with mocks."""
        with patch("{module_name}.dependency") as mock_dep:
            mock_dep.return_value = "mocked_result"
            
            result = {function_name}("test_input")
            
            mock_dep.assert_called_once_with("test_input")
            self.assertEqual(result, "expected_result")


if __name__ == "__main__":
    unittest.main()
'''

    def _get_bdd_template(self) -> str:
        """Get BDD (Behavior-Driven Development) test template."""
        return '''import pytest
from pytest_bdd import scenarios, given, when, then, parsers
from {{module_name}} import {{class_or_function_name}}


# Load scenarios from feature file
scenarios("features/{{feature_name}}.feature")


@given("I have a component")
def setup_component():
    """Setup the component for testing."""
    return {{class_or_function_name}}()


@given(parsers.parse("the component has {{value}} set to {{input_value}}"))
def set_component_value(component, value, input_value):
    """Set a value on the component."""
    setattr(component, value, input_value)


@when("I call the function_name method")
def call_function(component):
    """Call the function under test."""
    return component.function_name()


@when(parsers.parse("I call function_name with {{input_value}}"))
def call_function_with_input(component, input_value):
    """Call the function with specific input."""
    return component.function_name(input_value)


@then("the result should be valid")
def validate_result(result):
    """Validate the result is valid."""
    assert result is not None


@then(parsers.parse("the result should equal {{expected_value}}"))
def validate_specific_result(result, expected_value):
    """Validate specific result value."""
    assert str(result) == expected_value


@then("no exceptions should be raised")
def no_exceptions():
    """Ensure no exceptions were raised."""
    # This step passes if we reach here without exceptions
    pass
'''

    async def _run_tdd_cycle(self, task: dict[str, Any]) -> dict[str, Any]:
        """Run a complete TDD (Test-Driven Development) cycle."""
        try:
            target_file = task.get("target_file", "")
            test_type = task.get("test_type", "unit")
            requirements = task.get("requirements", "")
            
            tdd_results = {
                "success": True,
                "cycle_completed": False,
                "phases": {},
                "final_state": "red",  # red, green, refactor
                "recommendations": []
            }
            
            # Phase 1: RED - Write failing test
            red_phase = await self._tdd_red_phase(target_file, requirements, test_type)
            tdd_results["phases"]["red"] = red_phase
            
            if not red_phase["success"]:
                tdd_results["success"] = False
                tdd_results["recommendations"].append("Failed to create failing test. Review requirements and try again.")
                return tdd_results
                
            # Phase 2: GREEN - Write minimal code to pass
            green_phase = await self._tdd_green_phase(target_file, red_phase["test_file"])
            tdd_results["phases"]["green"] = green_phase
            
            if not green_phase["success"]:
                tdd_results["final_state"] = "red"
                tdd_results["recommendations"].append("Failed to make tests pass. Review implementation approach.")
                return tdd_results
                
            # Phase 3: REFACTOR - Improve code quality
            refactor_phase = await self._tdd_refactor_phase(target_file, red_phase["test_file"])
            tdd_results["phases"]["refactor"] = refactor_phase
            
            if refactor_phase["success"]:
                tdd_results["final_state"] = "refactor"
                tdd_results["cycle_completed"] = True
                tdd_results["recommendations"].append("TDD cycle completed successfully. Consider additional test cases.")
            else:
                tdd_results["final_state"] = "green"
                tdd_results["recommendations"].append("Refactoring phase had issues. Code works but could be improved.")
            
            # Update TDD state
            self.tdd_cycle_state = tdd_results["final_state"]
            
            return tdd_results
            
        except Exception as e:
            logger.error(f"Error in TDD cycle: {e}")
            return {
                "success": False,
                "error": str(e),
                "cycle_completed": False,
                "final_state": "error"
            }

    async def _tdd_red_phase(self, target_file: str, requirements: str, test_type: str) -> dict[str, Any]:
        """TDD Red Phase: Write a failing test."""
        try:
            # Generate test based on requirements (should fail initially)
            test_generation_task = {
                "target_file": target_file,
                "test_type": test_type,
                "coverage_target": 100,
                "requirements": requirements,
                "tdd_phase": "red",
                "expect_failure": True
            }
            
            # Use Goose to generate failing test
            test_result = await self._generate_tests_with_goose(target_file, test_generation_task)
            
            if not test_result["success"]:
                return {
                    "success": False,
                    "error": "Failed to generate failing test",
                    "phase": "red"
                }
            
            test_file = test_result["test_file"]
            
            # Run the test to confirm it fails
            test_execution = await self._execute_test_suite(
                test_file, "pytest", "test_*.py", False
            )
            
            # In RED phase, we expect tests to fail
            expected_failure = (
                test_execution["test_results"]["failed"] > 0 or
                test_execution["test_results"]["errors"] > 0
            )
            
            return {
                "success": expected_failure,
                "test_file": test_file,
                "test_results": test_execution["test_results"],
                "phase": "red",
                "message": "Failing test created successfully" if expected_failure else "Test should fail but passed"
            }
            
        except Exception as e:
            logger.error(f"Error in TDD red phase: {e}")
            return {
                "success": False,
                "error": str(e),
                "phase": "red"
            }

    async def _tdd_green_phase(self, target_file: str, test_file: str) -> dict[str, Any]:
        """TDD Green Phase: Write minimal code to make tests pass."""
        try:
            # Use Goose to generate minimal implementation
            if self.goose_path:
                implementation_prompt = f"""
Generate minimal code implementation for {target_file} to make the tests in {test_file} pass.

Requirements:
1. Write only the minimum code needed to pass the tests
2. Don't add extra features or optimizations
3. Focus on making tests green, not on perfect code quality
4. Use simple, direct implementations

Follow TDD GREEN phase principles: make it work, don't make it perfect yet.
"""
                
                goose_result = await self._run_goose_command([
                    "session", "start"
                ], input_text=implementation_prompt)
                
                if not goose_result["success"]:
                    # Fallback: suggest minimal implementation structure
                    return await self._generate_minimal_implementation(target_file, test_file)
            else:
                # Fallback without Goose
                return await self._generate_minimal_implementation(target_file, test_file)
            
            # Run tests to verify they pass
            test_execution = await self._execute_test_suite(
                test_file, "pytest", "test_*.py", False
            )
            
            tests_passing = test_execution["test_results"]["failed"] == 0
            
            return {
                "success": tests_passing,
                "test_results": test_execution["test_results"],
                "phase": "green",
                "implementation": "Generated via Goose CLI",
                "message": "Tests passing" if tests_passing else "Tests still failing"
            }
            
        except Exception as e:
            logger.error(f"Error in TDD green phase: {e}")
            return {
                "success": False,
                "error": str(e),
                "phase": "green"
            }

    async def _tdd_refactor_phase(self, target_file: str, test_file: str) -> dict[str, Any]:
        """TDD Refactor Phase: Improve code quality while keeping tests green."""
        try:
            # Analyze current code quality
            quality_before = await self._assess_test_quality(Path(target_file).parent)
            
            # Use Goose for refactoring suggestions
            if self.goose_path:
                refactor_prompt = f"""
Refactor the code in {target_file} to improve quality while maintaining all test functionality.

Current implementation passes all tests in {test_file}.

Refactoring goals:
1. Improve code readability and maintainability
2. Apply SOLID principles where appropriate
3. Remove code duplication
4. Optimize performance if needed
5. Add proper error handling
6. Ensure all tests continue to pass

IMPORTANT: Do not break existing functionality. All tests must continue to pass.
"""
                
                refactor_result = await self._run_goose_command([
                    "session", "start"  
                ], input_text=refactor_prompt)
                
                # Run tests again to ensure refactoring didn't break anything
                test_execution = await self._execute_test_suite(
                    test_file, "pytest", "test_*.py", False
                )
                
                tests_still_passing = test_execution["test_results"]["failed"] == 0
                
                if not tests_still_passing:
                    return {
                        "success": False,
                        "phase": "refactor",
                        "error": "Refactoring broke existing tests",
                        "test_results": test_execution["test_results"]
                    }
                
                # Assess quality after refactoring
                quality_after = await self._assess_test_quality(Path(target_file).parent)
                
                quality_improved = (
                    quality_after["quality"]["overall_score"] > 
                    quality_before["quality"]["overall_score"]
                )
                
                return {
                    "success": True,
                    "phase": "refactor",
                    "quality_before": quality_before["quality"]["overall_score"],
                    "quality_after": quality_after["quality"]["overall_score"],
                    "quality_improved": quality_improved,
                    "test_results": test_execution["test_results"],
                    "message": "Refactoring completed successfully"
                }
            else:
                # Without Goose, provide refactoring recommendations
                return await self._suggest_refactoring_improvements(target_file)
                
        except Exception as e:
            logger.error(f"Error in TDD refactor phase: {e}")
            return {
                "success": False,
                "error": str(e),
                "phase": "refactor"
            }

    async def _generate_minimal_implementation(self, target_file: str, test_file: str) -> dict[str, Any]:
        """Generate minimal implementation to pass tests (fallback without Goose)."""
        try:
            # Read test file to understand requirements
            test_path = Path(test_file)
            if test_path.exists():
                test_content = test_path.read_text()
                
                # Extract function/class names from tests
                import re
                functions = re.findall(r'def test_(\w+)', test_content)
                classes = re.findall(r'class Test(\w+)', test_content)
                
                # Generate minimal stub implementation
                implementation = '''"""Minimal implementation generated by TDD Green phase."""

'''
                
                for cls in classes:
                    implementation += f'''
class {cls}:
    """Minimal {cls} implementation."""
    
    def __init__(self):
        pass
'''
                
                for func in functions:
                    clean_func = func.replace('_basic_functionality', '').replace('_edge_cases', '').replace('_with_mocks', '')
                    if clean_func not in implementation:
                        implementation += f'''
    
def {clean_func}(*args, **kwargs):
    """Minimal {clean_func} implementation."""
    if not args and not kwargs:
        return None
    return "placeholder_result"
'''
                
                # Write minimal implementation
                target_path = Path(target_file)
                if not target_path.exists():
                    target_path.write_text(implementation)
                
                return {
                    "success": True,
                    "phase": "green",
                    "implementation": "Minimal stub generated",
                    "message": "Basic implementation structure created"
                }
            else:
                return {
                    "success": False,
                    "error": f"Test file {test_file} not found",
                    "phase": "green"
                }
                
        except Exception as e:
            logger.error(f"Error generating minimal implementation: {e}")
            return {
                "success": False,
                "error": str(e),
                "phase": "green"
            }

    async def _suggest_refactoring_improvements(self, target_file: str) -> dict[str, Any]:
        """Suggest refactoring improvements (fallback without Goose)."""
        try:
            suggestions = [
                "Extract common functionality into helper methods",
                "Add proper error handling and input validation",
                "Consider using design patterns (Strategy, Factory, etc.)",
                "Add type hints for better code documentation",
                "Remove any code duplication",
                "Optimize performance-critical sections",
                "Add comprehensive docstrings",
                "Consider breaking large functions into smaller ones"
            ]
            
            return {
                "success": True,
                "phase": "refactor",
                "suggestions": suggestions,
                "message": "Refactoring suggestions provided (manual implementation needed)"
            }
            
        except Exception as e:
            logger.error(f"Error suggesting refactoring improvements: {e}")
            return {
                "success": False,
                "error": str(e),
                "phase": "refactor"
            }

    async def _goose_fix_command(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute Goose's 'fix' command for automated bug fixing and code repair."""
        try:
            target_file = task.get("target_file", "")
            error_description = task.get("error_description", "")
            fix_prompt = task.get("fix_prompt", "")
            
            if not target_file:
                return {
                    "success": False,
                    "error": "No target file specified for Goose fix command"
                }
            
            # Log the fix attempt
            self.log_observation(
                f"Starting Goose fix command for {target_file}",
                data={"target_file": target_file, "error_description": error_description}
            )

            # Prepare Goose fix command
            fix_args = ["session", "start"]
            if self.session_id:
                fix_args = ["session", "resume", self.session_id]

            # Create fix prompt
            if fix_prompt:
                prompt_text = fix_prompt
            elif error_description:
                prompt_text = f"Fix the following error in {target_file}:\n\n{error_description}\n\nPlease analyze the code and provide a fix."
            else:
                prompt_text = f"Analyze {target_file} for potential issues and fix any bugs or problems found."

            # Run Goose session with fix prompt
            session_result = await self._run_goose_command(fix_args)

            if not session_result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to start Goose session: {session_result.get('error', 'Unknown error')}",
                    "goose_output": session_result.get("output", "")
                }

            # Extract session ID if available
            if "session" in session_result["output"] and not self.session_id:
                # Parse session ID from output
                import re
                session_match = re.search(r'session[:\s]+([a-f0-9\-]+)', session_result["output"])
                if session_match:
                    self.session_id = session_match.group(1)

            # Now run the fix command
            fix_command_result = await self._run_goose_command(
                ["run", "--message", prompt_text],
                cwd=str(Path(target_file).parent)
            )

            # Parse the fix results
            fix_success = fix_command_result["success"]
            fix_output = fix_command_result.get("output", "")
            fix_error = fix_command_result.get("error", "")

            # Log the results
            self.log_decision(
                f"Goose fix command completed for {target_file}",
                f"Fix {'successful' if fix_success else 'failed'}: {fix_output[:200]}...",
                data={
                    "success": fix_success,
                    "output": fix_output[:500] + "..." if len(fix_output) > 500 else fix_output,
                    "session_id": self.session_id
                }
            )            # Extract any code changes made
            code_changes = self._extract_code_changes_from_output(fix_output)
            
            # Verify the fix by running tests if available
            verification_results = {}
            if fix_success and target_file.endswith('.py'):
                test_file = self._find_test_file_for_target(target_file)
                if test_file and Path(test_file).exists():
                    test_result = await self._run_tests({
                        "test_path": test_file,
                        "framework": "pytest"
                    })
                    verification_results = test_result
            
            return {
                "success": fix_success,
                "target_file": target_file,
                "fix_applied": fix_success,
                "goose_output": fix_output,
                "goose_error": fix_error,
                "session_id": self.session_id,
                "code_changes": code_changes,
                "verification_results": verification_results,
                "execution_time": fix_command_result.get("execution_time", 0),
                "working_directory": fix_command_result.get("working_directory", "")
            }
            
        except Exception as e:
            self.logger.error(f"Error in Goose fix command: {e}")
            return {
                "success": False,
                "error": str(e),
                "target_file": task.get("target_file", ""),
                "method": "goose_fix_failed"
            }

    async def _goose_write_tests_command(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute Goose's 'write-tests' command for automated test generation."""
        try:
            target_file = task.get("target_file", "")
            test_type = task.get("test_type", "unit")
            coverage_target = task.get("coverage_target", "comprehensive")
            test_framework = task.get("test_framework", "pytest")
            
            if not target_file:
                return {
                    "success": False,
                    "error": "No target file specified for Goose write-tests command"
                }
            
            target_path = Path(target_file)
            if not target_path.exists():
                return {
                    "success": False,
                    "error": f"Target file {target_file} does not exist"
                }
            
            # Log the test generation attempt
            self.log_observation(
                f"Starting Goose write-tests command for {target_file}",
                data={
                    "target_file": target_file,
                    "test_type": test_type,
                    "framework": test_framework,
                    "coverage_target": coverage_target
                }
            )
            
            # Prepare Goose session
            session_args = ["session", "start"]
            if self.session_id:
                session_args = ["session", "resume", self.session_id]
            
            session_result = await self._run_goose_command(session_args)
            
            if not session_result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to start Goose session: {session_result.get('error', 'Unknown error')}",
                    "goose_output": session_result.get("output", "")
                }
            
            # Create comprehensive test generation prompt
            test_prompt = self._create_goose_test_prompt(
                target_file, test_type, coverage_target, test_framework
            )
            
            # Run the write-tests command
            test_command_result = await self._run_goose_command(
                ["run", "--message", test_prompt],
                cwd=str(target_path.parent)
            )
            
            # Parse the test generation results
            test_success = test_command_result["success"]
            test_output = test_command_result.get("output", "")
            test_error = test_command_result.get("error", "")
            
            # Extract generated test code
            generated_tests = self._extract_test_code_from_output(test_output)
            
            # Determine test file location
            test_file_path = self._determine_test_file_path(target_file, test_framework)
            
            # Save generated tests if successful
            tests_saved = False
            if generated_tests and test_success:
                try:
                    test_path = Path(test_file_path)
                    test_path.parent.mkdir(parents=True, exist_ok=True)
                    test_path.write_text(generated_tests)
                    tests_saved = True
                    
                    # Log the test file creation
                    self.log_result(
                        f"Generated tests saved to {test_file_path}",
                        data={
                            "test_file": test_file_path,
                            "lines_of_code": len(generated_tests.split('\n')),
                            "framework": test_framework
                        }
                    )
                except Exception as e:
                    self.logger.error(f"Failed to save generated tests: {e}")
            
            # Run the generated tests to verify they work
            test_execution_results = {}
            if tests_saved:
                test_run_result = await self._run_tests({
                    "test_path": test_file_path,
                    "framework": test_framework
                })
                test_execution_results = test_run_result
            
            # Calculate coverage if possible
            coverage_results = {}
            if tests_saved and test_success:
                coverage_result = await self._analyze_coverage({
                    "target_files": [target_file],
                    "test_files": [test_file_path]
                })
                coverage_results = coverage_result
            
            return {
                "success": test_success,
                "target_file": target_file,
                "test_file": test_file_path,
                "tests_generated": bool(generated_tests),
                "tests_saved": tests_saved,
                "goose_output": test_output,
                "goose_error": test_error,
                "session_id": self.session_id,
                "generated_test_code": generated_tests,
                "test_execution_results": test_execution_results,
                "coverage_results": coverage_results,
                "test_framework": test_framework,
                "execution_time": test_command_result.get("execution_time", 0),
                "working_directory": test_command_result.get("working_directory", "")
            }
            
        except Exception as e:
            self.logger.error(f"Error in Goose write-tests command: {e}")
            return {
                "success": False,
                "error": str(e),
                "target_file": task.get("target_file", ""),
                "method": "goose_write_tests_failed"
            }

    async def _run_automated_qa_pipeline(self, task: dict[str, Any]) -> dict[str, Any]:
        """Run a complete automated QA pipeline using Goose fix and write-tests."""
        try:
            target_files = task.get("target_files", [])
            pipeline_type = task.get("pipeline_type", "comprehensive")  # comprehensive, fix_only, test_only
            fix_issues = task.get("fix_issues", True)
            generate_tests = task.get("generate_tests", True)
            
            if not target_files:
                return {
                    "success": False,
                    "error": "No target files specified for QA pipeline"
                }
            
            # Log pipeline start
            self.log_observation(
                f"Starting automated QA pipeline for {len(target_files)} files",
                data={
                    "target_files": target_files,
                    "pipeline_type": pipeline_type,
                    "fix_issues": fix_issues,
                    "generate_tests": generate_tests
                }
            )
            
            pipeline_results = {
                "success": True,
                "files_processed": 0,
                "fixes_applied": 0,
                "tests_generated": 0,
                "file_results": {},
                "overall_coverage": {},
                "issues_found": [],
                "recommendations": []
            }
            
            # Process each target file
            for target_file in target_files:
                file_result = {
                    "file": target_file,
                    "fix_result": None,
                    "test_result": None,
                    "issues": [],
                    "success": True
                }
                
                try:
                    # Phase 1: Fix issues if requested
                    if fix_issues and pipeline_type in ["comprehensive", "fix_only"]:
                        fix_task = {
                            "target_file": target_file,
                            "fix_prompt": f"Analyze {target_file} for code quality issues, bugs, and improvements. Apply fixes as needed."
                        }
                        
                        fix_result = await self._goose_fix_command(fix_task)
                        file_result["fix_result"] = fix_result
                        
                        if fix_result["success"] and fix_result.get("fix_applied"):
                            pipeline_results["fixes_applied"] += 1
                    
                    # Phase 2: Generate tests if requested
                    if generate_tests and pipeline_type in ["comprehensive", "test_only"]:
                        test_task = {
                            "target_file": target_file,
                            "test_type": "comprehensive",
                            "coverage_target": "high",
                            "test_framework": "pytest"
                        }
                        
                        test_result = await self._goose_write_tests_command(test_task)
                        file_result["test_result"] = test_result
                        
                        if test_result["success"] and test_result.get("tests_generated"):
                            pipeline_results["tests_generated"] += 1
                    
                    pipeline_results["files_processed"] += 1
                    
                except Exception as e:
                    file_result["success"] = False
                    file_result["error"] = str(e)
                    file_result["issues"].append(f"Processing error: {str(e)}")
                    pipeline_results["success"] = False
                
                pipeline_results["file_results"][target_file] = file_result
            
            # Phase 3: Run comprehensive analysis on all generated tests
            if generate_tests:
                all_test_files = []
                for file_result in pipeline_results["file_results"].values():
                    if file_result.get("test_result") and file_result["test_result"].get("test_file"):
                        all_test_files.append(file_result["test_result"]["test_file"])
                
                if all_test_files:
                    # Run all tests together
                    combined_test_result = await self._run_tests({
                        "test_paths": all_test_files,
                        "framework": "pytest"
                    })
                    
                    # Analyze overall coverage
                    coverage_result = await self._analyze_coverage({
                        "target_files": target_files,
                        "test_files": all_test_files
                    })
                    
                    pipeline_results["overall_coverage"] = coverage_result
                    pipeline_results["combined_test_results"] = combined_test_result
            
            # Phase 4: Generate recommendations
            recommendations = self._generate_qa_pipeline_recommendations(pipeline_results)
            pipeline_results["recommendations"] = recommendations
            
            # Log pipeline completion
            self.log_result(
                f"Automated QA pipeline completed: {pipeline_results['files_processed']} files processed, "
                f"{pipeline_results['fixes_applied']} fixes applied, {pipeline_results['tests_generated']} test suites generated",
                data={
                    "pipeline_summary": {
                        k: v for k, v in pipeline_results.items() 
                        if k not in ["file_results"]  # Exclude detailed results from memory log
                    }
                }
            )
            
            return pipeline_results
            
        except Exception as e:
            self.logger.error(f"Error in automated QA pipeline: {e}")
            return {
                "success": False,
                "error": str(e),
                "method": "qa_pipeline_failed"
            }

    def _extract_code_changes_from_output(self, output: str) -> list[dict[str, Any]]:
        """Extract code changes from Goose command output."""
        changes = []
        
        # Look for common patterns in Goose output
        patterns = [
            r'(?:Modified|Changed|Updated|Fixed).*?(\w+\.py)',
            r'```python\n(.*?)\n```',
            r'File: (.*?)\n.*?Changes:.*?\n(.*?)(?=\n\n|\n$)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, output, re.DOTALL | re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    file_path, change_content = match[0], match[1] if len(match) > 1 else ""
                else:
                    file_path, change_content = match, ""
                
                changes.append({
                    "file": file_path,
                    "change": change_content.strip(),
                    "type": "modification"
                })
        
        return changes

    def _create_goose_test_prompt(self, target_file: str, test_type: str, 
                                  coverage_target: str, framework: str) -> str:
        """Create a comprehensive prompt for Goose test generation."""
        target_path = Path(target_file)
        
        prompt = f"""Generate comprehensive {framework} tests for the file {target_file}.

Requirements:
- Test type: {test_type}
- Coverage target: {coverage_target}
- Framework: {framework}
- Follow testing best practices

Please create tests that cover:
1. All public methods and functions
2. Edge cases and error conditions
3. Input validation
4. Expected behavior verification
5. Integration points if applicable

For {framework} tests, include:
- Proper fixtures and setup/teardown
- Parametrized tests where appropriate
- Mocking for external dependencies
- Clear test descriptions and assertions
- Error case testing with appropriate assertions

The tests should be saved in an appropriate test file location following standard conventions.
Please ensure the tests are runnable and follow {framework} best practices.
"""

        return prompt

    def _find_test_file_for_target(self, target_file: str) -> str | None:
        """Find the corresponding test file for a target file."""
        target_path = Path(target_file)
        
        # Common test file patterns
        test_patterns = [
            f"test_{target_path.stem}.py",
            f"{target_path.stem}_test.py",
            f"tests/test_{target_path.stem}.py",
            f"test/test_{target_path.stem}.py"
        ]
        
        for pattern in test_patterns:
            test_path = target_path.parent / pattern
            if test_path.exists():
                return str(test_path)
        
        return None

    def _determine_test_file_path(self, target_file: str, framework: str) -> str:
        """Determine where to save generated tests."""
        target_path = Path(target_file)
        
        # Standard test directory patterns
        if (target_path.parent / "tests").exists():
            test_dir = target_path.parent / "tests"
        elif (target_path.parent / "test").exists():
            test_dir = target_path.parent / "test"
        else:
            # Create tests directory
            test_dir = target_path.parent / "tests"
            test_dir.mkdir(exist_ok=True)
        
        test_filename = f"test_{target_path.stem}.py"
        return str(test_dir / test_filename)

    def _generate_qa_pipeline_recommendations(self, pipeline_results: dict[str, Any]) -> list[str]:
        """Generate recommendations based on QA pipeline results."""
        recommendations = []
        
        files_processed = pipeline_results.get("files_processed", 0)
        fixes_applied = pipeline_results.get("fixes_applied", 0)
        tests_generated = pipeline_results.get("tests_generated", 0)
        
        if fixes_applied > 0:
            recommendations.append(f"Applied {fixes_applied} automated fixes. Review changes before committing.")
        
        if tests_generated > 0:
            recommendations.append(f"Generated {tests_generated} test suites. Run tests to ensure they pass.")
        
        # Coverage recommendations
        coverage = pipeline_results.get("overall_coverage", {})
        if coverage and "coverage_percentage" in coverage:
            cov_pct = coverage["coverage_percentage"]
            if cov_pct < 70:
                recommendations.append("Consider adding more tests to improve coverage above 70%")
            elif cov_pct >= 90:
                recommendations.append("Excellent test coverage! Maintain this level in future changes.")
        
        # File-specific recommendations
        file_results = pipeline_results.get("file_results", {})
        failed_files = [f for f, r in file_results.items() if not r.get("success", True)]
        
        if failed_files:
            recommendations.append(f"Review failed files: {', '.join(failed_files)}")
        
        if not recommendations:
            recommendations.append("QA pipeline completed successfully. No immediate actions required.")
        
        return recommendations

    async def _generate_behavior_driven_tests(self, task: dict[str, Any]) -> dict[str, Any]:
        """Generate Behavior-Driven Development (BDD) style tests."""
        try:
            target_file = task.get("target_file", "")
            feature_description = task.get("feature_description", "")
            user_stories = task.get("user_stories", [])
            
            # Generate feature file
            feature_content = self._create_feature_file(target_file, feature_description, user_stories)
            
            # Generate step definitions
            step_definitions = self._create_step_definitions(target_file, user_stories)
            
            # Create feature and test files
            target_path = Path(target_file)
            feature_dir = target_path.parent / "features"
            feature_dir.mkdir(exist_ok=True)
            
            feature_file = feature_dir / f"{target_path.stem}.feature"
            feature_file.write_text(feature_content)
            
            test_file = target_path.parent / f"test_bdd_{target_path.stem}.py"
            test_file.write_text(step_definitions)
            
            return {
                "success": True,
                "method": "bdd_generated",
                "feature_file": str(feature_file),
                "test_file": str(test_file),
                "message": "BDD tests generated successfully"
            }
            
        except Exception as e:
            logger.error(f"Error generating BDD tests: {e}")
            return {
                "success": False,
                "error": str(e),
                "method": "bdd_generation_failed"
            }

    def _create_feature_file(self, target_file: str, description: str, user_stories: list[str]) -> str:
        """Create a Gherkin feature file."""
        target_path = Path(target_file)
        feature_name = target_path.stem.replace("_", " ").title()
        
        feature_content = f'''Feature: {feature_name}
  {description}
  
  As a user
  I want to use {feature_name} functionality
  So that I can accomplish my tasks efficiently

'''
        
        for i, story in enumerate(user_stories, 1):
            scenario_name = f"User Story {i}"
            feature_content += f'''  Scenario: {scenario_name}
    Given I have access to the {feature_name} functionality
    When I {story.lower()}
    Then the operation should complete successfully
    And the result should be valid

'''
        
        return feature_content

    def _create_step_definitions(self, target_file: str, user_stories: list[str]) -> str:
        """Create BDD step definitions."""
        target_path = Path(target_file)
        module_name = target_path.stem
        
        bdd_template = '''import pytest
from pytest_bdd import scenarios, given, when, then, parsers
from {module_name} import {class_name}


# Load scenarios from feature file
scenarios("features/{feature_name}.feature")


@given("I have a component")
def setup_component():
    """Setup the component for testing."""
    return {class_name}()


@given(parsers.parse("the component has {{value}} set to {{input_value}}"))
def set_component_value(component, value, input_value):
    """Set a value on the component."""
    setattr(component, value, input_value)


@when("I call the process method")
def call_function(component):
    """Call the function under test."""
    return component.process()


@when(parsers.parse("I call process with {{input_value}}"))
def call_function_with_input(component, input_value):
    """Call the function with specific input."""
    return component.process(input_value)


@then("the result should be valid")
def validate_result(result):
    """Validate the result is valid."""
    assert result is not None


@then(parsers.parse("the result should equal {{expected_value}}"))
def validate_specific_result(result, expected_value):
    """Validate specific result value."""
    assert str(result) == expected_value


@then("no exceptions should be raised")
def no_exceptions():
    """Ensure no exceptions were raised."""
    # This step passes if we reach here without exceptions
    pass
'''
        
        return bdd_template.format(
            module_name=module_name,
            class_name=module_name.title(),
            feature_name=module_name
        )
