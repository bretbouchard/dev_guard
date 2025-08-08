"""
Impact Mapper Agent for DevGuard - Advanced Cross-Repository Impact Analysis

This agent analyzes code changes across repositories to identify potential impacts,
dependencies, breaking changes, and coordination needs. It performs deep analysis
of code structure, API changes, dependency relationships, and cross-repository
coordination requirements.
"""

import logging
import ast
import re
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from .base_agent import BaseAgent
from ..core.config import Config
from ..memory.shared_memory import SharedMemory, AgentState
from ..memory.vector_db import VectorDatabase
from ..llm.provider import LLMProvider

logger = logging.getLogger(__name__)


class ImpactType(Enum):
    """Types of cross-repository impacts."""
    API_BREAKING = "api_breaking"
    API_NON_BREAKING = "api_non_breaking"
    DEPENDENCY_CHANGE = "dependency_change"
    SCHEMA_CHANGE = "schema_change"
    PERFORMANCE_IMPACT = "performance_impact"
    SECURITY_IMPACT = "security_impact"
    CONFIGURATION_CHANGE = "configuration_change"
    WORKFLOW_CHANGE = "workflow_change"


class ImpactSeverity(Enum):
    """Severity levels for cross-repository impacts."""
    CRITICAL = "critical"  # Breaking changes requiring immediate attention
    HIGH = "high"         # Significant changes affecting functionality
    MEDIUM = "medium"     # Changes requiring coordination but not breaking
    LOW = "low"          # Minor changes with minimal impact
    INFO = "info"        # Informational changes


@dataclass
class ImpactAnalysis:
    """Detailed impact analysis result."""
    impact_id: str
    source_repository: str
    target_repository: str
    impact_type: ImpactType
    severity: ImpactSeverity
    description: str
    affected_files: List[str]
    affected_functions: List[str]
    affected_classes: List[str]
    breaking_changes: List[Dict[str, Any]]
    recommendations: List[str]
    estimated_effort: str  # hours, days, weeks
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class APIChange:
    """Represents an API change with impact analysis."""
    function_name: str
    class_name: Optional[str]
    file_path: str
    change_type: str  # added, removed, modified, deprecated
    old_signature: Optional[str]
    new_signature: Optional[str]
    breaking: bool
    deprecation_info: Optional[Dict[str, Any]] = None


@dataclass
class DependencyImpact:
    """Represents dependency relationship impact."""
    dependency_name: str
    old_version: Optional[str]
    new_version: Optional[str]
    affected_repositories: List[str]
    compatibility_issues: List[str]
    upgrade_path: List[str]


class ImpactMapperAgent(BaseAgent):
    """
    Advanced Impact Mapper Agent for cross-repository impact analysis.
    
    Capabilities:
    - Cross-repository change impact analysis
    - API breaking change detection
    - Dependency relationship mapping
    - Code structure analysis
    - Inter-service compatibility checking
    - Impact severity assessment
    - Automated recommendations generation
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_provider = kwargs.get('llm_provider')
        
        # Impact analysis state
        self.repository_mappings: Dict[str, Dict[str, Any]] = {}
        self.api_definitions: Dict[str, Dict[str, Any]] = {}
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.impact_history: List[ImpactAnalysis] = []
        
        # Analysis patterns and rules
        self.api_patterns = {
            'function_def': r'def\s+(\w+)\s*\([^)]*\):',
            'class_def': r'class\s+(\w+)(?:\([^)]*\))?:',
            'import_from': r'from\s+(\w+(?:\.\w+)*)\s+import\s+([^#\n]+)',
            'import_direct': r'import\s+([^#\n]+)',
            'endpoint_pattern': r'@app\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']',
            'decorator_pattern': r'@(\w+(?:\.\w+)*)'
        }
        
        self.breaking_change_patterns = [
            r'def\s+(\w+)\s*\([^)]*\)\s*->',  # Function signature changes
            r'class\s+(\w+).*:',              # Class definition changes
            r'raise\s+(\w+Error)',            # New exceptions
            r'@deprecated',                   # Deprecation markers
        ]
        
        logger.info(f"Impact Mapper Agent {self.agent_id} initialized")
        
    async def execute(self, state: Any) -> Any:
        """Execute the impact mapper agent's main logic."""
        if isinstance(state, dict):
            task = state
        else:
            task = {"type": "analyze_impact", "description": str(state)}
        
        return await self.execute_task(task)
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a comprehensive impact analysis task."""
        try:
            self._update_state("busy", task.get("task_id"))
            
            task_type = task.get("type", "analyze_impact")
            
            if task_type == "analyze_impact":
                result = await self._analyze_cross_repository_impact(task)
            elif task_type == "analyze_api_changes":
                result = await self._analyze_api_changes(task)
            elif task_type == "analyze_dependency_impact":
                result = await self._analyze_dependency_impact(task)
            elif task_type == "map_repository_relationships":
                result = await self._map_repository_relationships(task)
            elif task_type == "detect_breaking_changes":
                result = await self._detect_breaking_changes(task)
            elif task_type == "generate_impact_report":
                result = await self._generate_impact_report(task)
            elif task_type == "validate_compatibility":
                result = await self._validate_compatibility(task)
            elif task_type == "suggest_coordination":
                result = await self._suggest_coordination_tasks(task)
            else:
                result = {"success": False, "error": f"Unknown task type: {task_type}"}
            
            self._update_state("idle")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in impact mapper task execution: {e}")
            self._update_state("error", error=str(e))
            return {"success": False, "error": str(e)}
    
    # ================================
    # CORE IMPACT ANALYSIS METHODS
    # ================================
    
    async def _analyze_cross_repository_impact(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive cross-repository impact analysis."""
        try:
            source_repo = task.get("source_repository")
            changed_files = task.get("changed_files", [])
            change_type = task.get("change_type", "commit")
            target_repos = task.get("target_repositories", [])
            
            if not source_repo:
                return {"success": False, "error": "Source repository required"}
            
            # Analyze changes in source repository
            change_analysis = await self._analyze_repository_changes(source_repo, changed_files)
            
            # Find potentially affected repositories
            if not target_repos:
                target_repos = await self._discover_related_repositories(source_repo)
            
            # Perform impact analysis for each target repository
            impact_results = []
            for target_repo in target_repos:
                impact = await self._analyze_repository_pair_impact(
                    source_repo, target_repo, change_analysis
                )
                if impact:
                    impact_results.append(impact)
            
            # Consolidate and prioritize results
            consolidated_impact = self._consolidate_impact_analysis(impact_results)
            
            # Generate recommendations
            recommendations = await self._generate_impact_recommendations(
                source_repo, consolidated_impact
            )
            
            # Store results
            self._store_impact_analysis(source_repo, consolidated_impact)
            
            return {
                "success": True,
                "source_repository": source_repo,
                "change_type": change_type,
                "affected_repositories": len(impact_results),
                "critical_impacts": len([i for i in impact_results if i.severity == ImpactSeverity.CRITICAL]),
                "high_impacts": len([i for i in impact_results if i.severity == ImpactSeverity.HIGH]),
                "medium_impacts": len([i for i in impact_results if i.severity == ImpactSeverity.MEDIUM]),
                "low_impacts": len([i for i in impact_results if i.severity == ImpactSeverity.LOW]),
                "impact_details": [asdict(impact) for impact in impact_results],
                "recommendations": recommendations,
                "analysis_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _analyze_repository_changes(self, repository: str, changed_files: List[str]) -> Dict[str, Any]:
        """Analyze changes within a repository to understand their nature."""
        analysis = {
            "api_changes": [],
            "dependency_changes": [],
            "config_changes": [],
            "schema_changes": [],
            "breaking_changes": []
        }
        
        try:
            # Get repository path from config or vector DB
            repo_path = await self._get_repository_path(repository)
            if not repo_path:
                return analysis
                
            for file_path in changed_files:
                full_path = Path(repo_path) / file_path
                if not full_path.exists():
                    continue
                
                # Analyze different file types
                if file_path.endswith(('.py', '.js', '.ts', '.java', '.go', '.rs')):
                    code_analysis = await self._analyze_code_file(full_path, file_path)
                    analysis["api_changes"].extend(code_analysis.get("api_changes", []))
                    analysis["breaking_changes"].extend(code_analysis.get("breaking_changes", []))
                
                elif file_path.endswith(('.json', '.yaml', '.yml', '.toml', '.ini')):
                    config_analysis = await self._analyze_config_file(full_path, file_path)
                    analysis["config_changes"].extend(config_analysis.get("config_changes", []))
                
                elif file_path in ['requirements.txt', 'package.json', 'Cargo.toml', 'go.mod', 'pom.xml']:
                    dep_analysis = await self._analyze_dependency_file(full_path, file_path)
                    analysis["dependency_changes"].extend(dep_analysis.get("dependency_changes", []))
                
                elif file_path.endswith(('.sql', '.json', '.proto', '.avsc')):
                    schema_analysis = await self._analyze_schema_file(full_path, file_path)
                    analysis["schema_changes"].extend(schema_analysis.get("schema_changes", []))
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing repository changes: {e}")
            return analysis
    
    async def _analyze_code_file(self, file_path: Path, relative_path: str) -> Dict[str, Any]:
        """Analyze a code file for API and breaking changes."""
        analysis = {"api_changes": [], "breaking_changes": []}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract API definitions
            if file_path.suffix == '.py':
                api_changes = self._extract_python_apis(content, relative_path)
            elif file_path.suffix in ['.js', '.ts']:
                api_changes = self._extract_javascript_apis(content, relative_path)
            else:
                api_changes = self._extract_generic_apis(content, relative_path)
            
            analysis["api_changes"] = api_changes
            
            # Detect breaking changes
            breaking_changes = self._detect_breaking_changes_in_content(content, relative_path)
            analysis["breaking_changes"] = breaking_changes
            
        except Exception as e:
            self.logger.error(f"Error analyzing code file {file_path}: {e}")
        
        return analysis
    
    async def _analyze_repository_pair_impact(self, source_repo: str, target_repo: str, 
                                            change_analysis: Dict[str, Any]) -> Optional[ImpactAnalysis]:
        """Analyze impact between two specific repositories."""
        try:
            # Check if repositories are related
            relationship = await self._get_repository_relationship(source_repo, target_repo)
            if not relationship:
                return None
            
            # Analyze specific impacts based on relationship type
            impacts = []
            severity = ImpactSeverity.LOW
            
            # API impact analysis
            if change_analysis["api_changes"]:
                api_impact = await self._analyze_api_impact(
                    source_repo, target_repo, change_analysis["api_changes"]
                )
                if api_impact:
                    impacts.extend(api_impact)
                    severity = max(severity, ImpactSeverity.HIGH)
            
            # Dependency impact analysis
            if change_analysis["dependency_changes"]:
                dep_impact = await self._analyze_dependency_relationship_impact(
                    source_repo, target_repo, change_analysis["dependency_changes"]
                )
                if dep_impact:
                    impacts.extend(dep_impact)
                    severity = max(severity, ImpactSeverity.MEDIUM)
            
            # Breaking changes analysis
            if change_analysis["breaking_changes"]:
                breaking_impact = await self._analyze_breaking_change_impact(
                    source_repo, target_repo, change_analysis["breaking_changes"]
                )
                if breaking_impact:
                    impacts.extend(breaking_impact)
                    severity = ImpactSeverity.CRITICAL
            
            if not impacts:
                return None
            
            # Create impact analysis
            impact_analysis = ImpactAnalysis(
                impact_id=f"{source_repo}_{target_repo}_{int(datetime.now().timestamp())}",
                source_repository=source_repo,
                target_repository=target_repo,
                impact_type=self._determine_primary_impact_type(impacts),
                severity=severity,
                description=self._generate_impact_description(impacts),
                affected_files=self._extract_affected_files(impacts),
                affected_functions=self._extract_affected_functions(impacts),
                affected_classes=self._extract_affected_classes(impacts),
                breaking_changes=change_analysis["breaking_changes"],
                recommendations=await self._generate_mitigation_recommendations(impacts),
                estimated_effort=self._estimate_coordination_effort(impacts, severity),
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "relationship_type": relationship,
                    "change_count": len(impacts),
                    "analysis_version": "1.0"
                }
            )
            
            return impact_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing repository pair impact: {e}")
            return None
            
    
    # ================================
    # API ANALYSIS METHODS
    # ================================
    
    async def _analyze_api_changes(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze API changes and their cross-repository impact."""
        try:
            source_repo = task.get("source_repository")
            api_changes = task.get("api_changes", [])
            target_repos = task.get("target_repositories", [])
            
            if not source_repo or not api_changes:
                return {"success": False, "error": "Source repository and API changes required"}
            
            analysis_results = []
            
            for change in api_changes:
                # Analyze each API change
                impact_details = await self._analyze_single_api_change(source_repo, change)
                
                # Find affected repositories
                for target_repo in target_repos:
                    usage_impact = await self._find_api_usage_impact(
                        target_repo, change, impact_details
                    )
                    if usage_impact:
                        analysis_results.append(usage_impact)
            
            return {
                "success": True,
                "source_repository": source_repo,
                "api_changes_analyzed": len(api_changes),
                "affected_repositories": len(set(r["repository"] for r in analysis_results)),
                "impact_details": analysis_results,
                "analysis_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _analyze_dependency_impact(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dependency changes and their cross-repository impact."""
        try:
            source_repo = task.get("source_repository")
            dependency_changes = task.get("dependency_changes", [])
            
            if not source_repo or not dependency_changes:
                return {"success": False, "error": "Source repository and dependency changes required"}
            
            impact_analysis = []
            
            for dep_change in dependency_changes:
                # Analyze dependency impact
                dep_impact = DependencyImpact(
                    dependency_name=dep_change["name"],
                    old_version=dep_change.get("old_version"),
                    new_version=dep_change.get("new_version"),
                    affected_repositories=[],
                    compatibility_issues=[],
                    upgrade_path=[]
                )
                
                # Find repositories using this dependency
                affected_repos = await self._find_repositories_using_dependency(
                    dep_change["name"]
                )
                dep_impact.affected_repositories = affected_repos
                
                # Analyze compatibility
                compatibility = await self._analyze_dependency_compatibility(dep_change)
                dep_impact.compatibility_issues = compatibility.get("issues", [])
                dep_impact.upgrade_path = compatibility.get("upgrade_path", [])
                
                impact_analysis.append(asdict(dep_impact))
            
            return {
                "success": True,
                "source_repository": source_repo,
                "dependency_changes_analyzed": len(dependency_changes),
                "total_affected_repositories": len(set(
                    repo for dep in impact_analysis 
                    for repo in dep["affected_repositories"]
                )),
                "dependency_impacts": impact_analysis,
                "analysis_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _map_repository_relationships(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Map and analyze relationships between repositories."""
        try:
            repositories = task.get("repositories", [])
            if not repositories:
                # Get all repositories from vector DB
                repositories = await self._get_all_repositories()
            
            relationship_map = {}
            dependency_graph = {}
            
            for repo in repositories:
                # Analyze repository metadata
                repo_info = await self._analyze_repository_metadata(repo)
                relationship_map[repo] = repo_info
                
                # Build dependency graph
                dependencies = await self._extract_repository_dependencies(repo)
                dependency_graph[repo] = dependencies
            
            # Find inter-repository relationships
            relationships = self._analyze_repository_relationships(
                relationship_map, dependency_graph
            )
            
            # Store relationship mappings
            self.repository_mappings = relationship_map
            self.dependency_graph = dependency_graph
            
            return {
                "success": True,
                "repositories_analyzed": len(repositories),
                "relationships_found": len(relationships),
                "relationship_map": relationship_map,
                "dependency_graph": dependency_graph,
                "inter_repo_relationships": relationships,
                "analysis_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _detect_breaking_changes(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Detect breaking changes in code modifications."""
        try:
            repository = task.get("repository")
            changed_files = task.get("changed_files", [])
            old_content = task.get("old_content", {})
            new_content = task.get("new_content", {})
            
            if not repository or not changed_files:
                return {"success": False, "error": "Repository and changed files required"}
            
            breaking_changes = []
            
            for file_path in changed_files:
                old_file_content = old_content.get(file_path, "")
                new_file_content = new_content.get(file_path, "")
                
                if old_file_content and new_file_content:
                    file_breaking_changes = await self._detect_file_breaking_changes(
                        file_path, old_file_content, new_file_content
                    )
                    breaking_changes.extend(file_breaking_changes)
            
            # Categorize breaking changes
            categorized_changes = self._categorize_breaking_changes(breaking_changes)
            
            return {
                "success": True,
                "repository": repository,
                "files_analyzed": len(changed_files),
                "breaking_changes_detected": len(breaking_changes),
                "breaking_changes": breaking_changes,
                "categorized_changes": categorized_changes,
                "analysis_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _generate_impact_report(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive impact analysis report."""
        try:
            analysis_id = task.get("analysis_id")
            include_details = task.get("include_details", True)
            
            # Retrieve impact analysis from history
            target_analysis = None
            if analysis_id:
                target_analysis = next(
                    (a for a in self.impact_history if a.impact_id == analysis_id), None
                )
            else:
                # Use most recent analysis
                target_analysis = self.impact_history[-1] if self.impact_history else None
            
            if not target_analysis:
                return {"success": False, "error": "No impact analysis found"}
            
            # Generate comprehensive report
            report = {
                "analysis_summary": {
                    "impact_id": target_analysis.impact_id,
                    "source_repository": target_analysis.source_repository,
                    "target_repository": target_analysis.target_repository,
                    "impact_type": target_analysis.impact_type.value,
                    "severity": target_analysis.severity.value,
                    "timestamp": target_analysis.timestamp.isoformat()
                },
                "impact_overview": {
                    "description": target_analysis.description,
                    "affected_files_count": len(target_analysis.affected_files),
                    "affected_functions_count": len(target_analysis.affected_functions),
                    "affected_classes_count": len(target_analysis.affected_classes),
                    "breaking_changes_count": len(target_analysis.breaking_changes)
                },
                "recommendations": target_analysis.recommendations,
                "estimated_effort": target_analysis.estimated_effort
            }
            
            if include_details:
                report["detailed_analysis"] = {
                    "affected_files": target_analysis.affected_files,
                    "affected_functions": target_analysis.affected_functions,
                    "affected_classes": target_analysis.affected_classes,
                    "breaking_changes": target_analysis.breaking_changes,
                    "metadata": target_analysis.metadata
                }
            
            return {
                "success": True,
                "report": report,
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _validate_compatibility(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Validate compatibility between repository versions."""
        try:
            source_repo = task.get("source_repository")
            target_repo = task.get("target_repository")
            source_version = task.get("source_version", "HEAD")
            target_version = task.get("target_version", "HEAD")
            
            if not source_repo or not target_repo:
                return {"success": False, "error": "Source and target repositories required"}
            
            # Get API definitions for both repositories
            source_apis = await self._extract_repository_apis(source_repo, source_version)
            target_apis = await self._extract_repository_apis(target_repo, target_version)
            
            # Compare API compatibility
            compatibility_issues = []
            missing_apis = []
            incompatible_apis = []
            
            # Check for missing or incompatible APIs
            for api_name, api_def in source_apis.items():
                if api_name not in target_apis:
                    missing_apis.append({
                        "api_name": api_name,
                        "type": api_def.get("type"),
                        "signature": api_def.get("signature")
                    })
                else:
                    # Check signature compatibility
                    target_def = target_apis[api_name]
                    if not self._check_api_compatibility(api_def, target_def):
                        incompatible_apis.append({
                            "api_name": api_name,
                            "source_signature": api_def.get("signature"),
                            "target_signature": target_def.get("signature"),
                            "compatibility_issues": self._analyze_signature_differences(
                                api_def, target_def
                            )
                        })
            
            compatibility_score = self._calculate_compatibility_score(
                len(source_apis), len(missing_apis), len(incompatible_apis)
            )
            
            return {
                "success": True,
                "source_repository": source_repo,
                "target_repository": target_repo,
                "source_version": source_version,
                "target_version": target_version,
                "compatibility_score": compatibility_score,
                "total_apis_checked": len(source_apis),
                "missing_apis": missing_apis,
                "incompatible_apis": incompatible_apis,
                "compatibility_status": "compatible" if compatibility_score >= 0.9 else 
                                     "mostly_compatible" if compatibility_score >= 0.7 else
                                     "incompatible",
                "analysis_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _suggest_coordination_tasks(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest coordination tasks based on impact analysis."""
        try:
            impact_analysis = task.get("impact_analysis")
            if not impact_analysis:
                return {"success": False, "error": "Impact analysis required"}
            
            coordination_tasks = []
            
            # Analyze impact severity and generate appropriate tasks
            severity = impact_analysis.get("severity", "low")
            impact_type = impact_analysis.get("impact_type", "unknown")
            
            if severity in ["critical", "high"]:
                # High-priority coordination tasks
                coordination_tasks.extend([
                    {
                        "task_type": "immediate_notification",
                        "priority": "critical",
                        "description": "Notify affected teams immediately",
                        "assignee": "commander",
                        "deadline": "immediate"
                    },
                    {
                        "task_type": "impact_review",
                        "priority": "high",
                        "description": "Review breaking changes and coordination plan",
                        "assignee": "qa_agent",
                        "deadline": "within_24_hours"
                    }
                ])
            
            if impact_type == "api_breaking":
                coordination_tasks.extend([
                    {
                        "task_type": "api_documentation_update",
                        "priority": "high",
                        "description": "Update API documentation with breaking changes",
                        "assignee": "docs_agent",
                        "deadline": "before_release"
                    },
                    {
                        "task_type": "backward_compatibility_check",
                        "priority": "medium",
                        "description": "Implement backward compatibility if possible",
                        "assignee": "code_agent",
                        "deadline": "before_release"
                    }
                ])
            
            if impact_type == "dependency_change":
                coordination_tasks.extend([
                    {
                        "task_type": "dependency_update_coordination",
                        "priority": "medium", 
                        "description": "Coordinate dependency updates across repositories",
                        "assignee": "dependency_manager",
                        "deadline": "within_week"
                    }
                ])
            
            return {
                "success": True,
                "coordination_tasks": coordination_tasks,
                "total_tasks": len(coordination_tasks),
                "critical_tasks": len([t for t in coordination_tasks if t["priority"] == "critical"]),
                "high_priority_tasks": len([t for t in coordination_tasks if t["priority"] == "high"]),
                "suggested_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ================================
    # HELPER AND UTILITY METHODS
    # ================================
    
    async def _discover_related_repositories(self, source_repo: str) -> List[str]:
        """Discover repositories potentially related to the source repository."""
        related_repos = set()
        
        try:
            # Check dependency graph
            if source_repo in self.dependency_graph:
                related_repos.update(self.dependency_graph[source_repo])
            
            # Check reverse dependencies
            for repo, deps in self.dependency_graph.items():
                if source_repo in deps:
                    related_repos.add(repo)
            
            # Search vector database for related repositories
            search_results = await self.vector_db.search(
                query=f"repository {source_repo} dependencies imports",
                collection_name="code_knowledge",
                n_results=50
            )
            
            for result in search_results.get("documents", []):
                metadata = result.get("metadata", {})
                repo_path = metadata.get("source_file", "")
                if repo_path and repo_path != source_repo:
                    # Extract repository name from path
                    repo_name = self._extract_repo_name_from_path(repo_path)
                    if repo_name:
                        related_repos.add(repo_name)
            
            return list(related_repos)
            
        except Exception as e:
            self.logger.error(f"Error discovering related repositories: {e}")
            return []
    
    def _extract_repo_name_from_path(self, path: str) -> Optional[str]:
        """Extract repository name from file path."""
        try:
            parts = path.split('/')
            # Look for common repository path patterns
            for i, part in enumerate(parts):
                if part in ['repos', 'repositories', 'projects', 'src']:
                    if i + 1 < len(parts):
                        return parts[i + 1]
            # Fallback: return the first directory-like part
            for part in parts:
                if part and '.' not in part and len(part) > 2:
                    return part
            return None
        except:
            return None
    
    async def _get_repository_path(self, repository: str) -> Optional[str]:
        """Get the file system path for a repository."""
        try:
            # Try to get from vector database metadata
            search_results = await self.vector_db.search(
                query=f"repository {repository}",
                collection_name="code_knowledge",
                n_results=1
            )
            
            if search_results.get("documents"):
                metadata = search_results["documents"][0].get("metadata", {})
                source_file = metadata.get("source_file", "")
                if source_file:
                    # Extract repository root from source file path
                    path_parts = source_file.split("/")
                    for i, part in enumerate(path_parts):
                        if part == repository:
                            return "/".join(path_parts[:i+1])
            
            # Fallback: try common repository locations
            common_paths = [
                f"/Users/{repository}",
                f"/home/{repository}",
                f"./repos/{repository}",
                f"./{repository}"
            ]
            
            for path in common_paths:
                if Path(path).exists():
                    return path
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting repository path: {e}")
            return None
    
    def _extract_python_apis(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Extract Python API definitions from code content."""
        apis = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    api = {
                        "name": node.name,
                        "type": "function",
                        "file_path": file_path,
                        "line_number": node.lineno,
                        "signature": self._get_function_signature(node),
                        "is_public": not node.name.startswith("_"),
                        "decorators": [d.id for d in node.decorator_list if hasattr(d, 'id')],
                        "async": isinstance(node, ast.AsyncFunctionDef)
                    }
                    apis.append(api)
                
                elif isinstance(node, ast.ClassDef):
                    api = {
                        "name": node.name,
                        "type": "class",
                        "file_path": file_path,
                        "line_number": node.lineno,
                        "methods": [],
                        "is_public": not node.name.startswith("_"),
                        "base_classes": [base.id for base in node.bases if hasattr(base, 'id')]
                    }
                    
                    # Extract class methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method = {
                                "name": item.name,
                                "signature": self._get_function_signature(item),
                                "is_public": not item.name.startswith("_"),
                                "line_number": item.lineno
                            }
                            api["methods"].append(method)
                    
                    apis.append(api)
        
        except Exception as e:
            self.logger.error(f"Error extracting Python APIs from {file_path}: {e}")
        
        return apis
    
    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """Extract function signature from AST node."""
        try:
            args = []
            for arg in node.args.args:
                args.append(arg.arg)
            
            signature = f"{node.name}({', '.join(args)})"
            
            if node.returns:
                signature += f" -> {ast.unparse(node.returns)}"
            
            return signature
        
        except Exception:
            return f"{node.name}(...)"
    
    def _extract_javascript_apis(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Extract JavaScript API definitions from code content."""
        apis = []
        
        try:
            # Simple regex-based extraction for JavaScript
            # Function declarations
            function_pattern = r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)'
            for match in re.finditer(function_pattern, content):
                api = {
                    "name": match.group(1),
                    "type": "function",
                    "file_path": file_path,
                    "is_export": "export" in match.group(0),
                    "is_async": "async" in match.group(0)
                }
                apis.append(api)
            
            # Class declarations
            class_pattern = r'(?:export\s+)?class\s+(\w+)'
            for match in re.finditer(class_pattern, content):
                api = {
                    "name": match.group(1),
                    "type": "class",
                    "file_path": file_path,
                    "is_export": "export" in match.group(0)
                }
                apis.append(api)
        
        except Exception as e:
            self.logger.error(f"Error extracting JavaScript APIs from {file_path}: {e}")
        
        return apis
    
    def _parse_requirements_txt(self, content: str) -> Dict[str, str]:
        """Parse requirements.txt content and extract dependencies."""
        dependencies = {}
        
        try:
            for line in content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Handle different requirement formats
                    if '==' in line:
                        name, version = line.split('==', 1)
                        dependencies[name.strip()] = version.strip()
                    elif '>=' in line:
                        name, version = line.split('>=', 1)
                        dependencies[name.strip()] = f">={version.strip()}"
                    elif '<=' in line:
                        name, version = line.split('<=', 1)
                        dependencies[name.strip()] = f"<={version.strip()}"
                    else:
                        # No version specified
                        dependencies[line] = "latest"
        
        except Exception as e:
            self.logger.error(f"Error parsing requirements.txt: {e}")
        
        return dependencies
    
    def _parse_package_json(self, content: str) -> Dict[str, Dict[str, str]]:
        """Parse package.json content and extract dependencies."""
        try:
            package_data = json.loads(content)
            return {
                "dependencies": package_data.get("dependencies", {}),
                "devDependencies": package_data.get("devDependencies", {}),
                "peerDependencies": package_data.get("peerDependencies", {})
            }
        except Exception as e:
            self.logger.error(f"Error parsing package.json: {e}")
            return {"dependencies": {}, "devDependencies": {}, "peerDependencies": {}}
    
    def _detect_breaking_changes_in_content(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Detect breaking changes in file content."""
        breaking_changes = []
        
        try:
            lines = content.split('\n')
            
            for i, line in enumerate(lines):
                for pattern in self.breaking_change_patterns:
                    matches = re.finditer(pattern, line)
                    for match in matches:
                        breaking_change = {
                            "type": "potential_breaking_change",
                            "file_path": file_path,
                            "line_number": i + 1,
                            "line_content": line.strip(),
                            "pattern_matched": pattern,
                            "description": f"Potential breaking change detected in {match.group(0)}"
                        }
                        breaking_changes.append(breaking_change)
        
        except Exception as e:
            self.logger.error(f"Error detecting breaking changes: {e}")
        
        return breaking_changes
    
    # ================================
    # STATUS AND UTILITY METHODS
    # ================================
    
    def _update_state(self, status: str, task_id: Optional[str] = None, error: Optional[str] = None) -> None:
        """Update agent state."""
        state = AgentState(
            agent_id=self.agent_id,
            status=status,
            current_task=task_id,
            last_heartbeat=datetime.now(timezone.utc),
            metadata={"error": error if error else None}
        )
        self.shared_memory.update_agent_state(state)
    
    def get_capabilities(self) -> List[str]:
        return ["impact_analysis"]
    
    def get_status(self) -> Dict[str, Any]:
        return {"agent_id": self.agent_id, "type": "impact_mapper"}
