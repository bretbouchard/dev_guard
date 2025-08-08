"""
Dependency Manager Agent for DevGuard - Advanced Dependency Tracking and 
Security Management

This agent provides comprehensive dependency management across repositories including
version tracking, automated updates, security vulnerability scanning, compatibility
analysis, and update justification logging.
"""

import logging
import json
import re
import subprocess
import hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum

from .base_agent import BaseAgent
from ..core.config import Config
from ..memory.shared_memory import SharedMemory, AgentState
from ..memory.vector_db import VectorDatabase
from ..llm.provider import LLMProvider

logger = logging.getLogger(__name__)


class DependencyType(Enum):
    """Types of dependencies."""
    PRODUCTION = "production"
    DEVELOPMENT = "development"
    OPTIONAL = "optional"
    PEER = "peer"
    BUILD = "build"


class SecuritySeverity(Enum):
    """Security vulnerability severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class UpdateStrategy(Enum):
    """Dependency update strategies."""
    AUTOMATIC = "automatic"        # Auto-update within semver constraints
    MANUAL = "manual"             # Require manual approval
    SECURITY_ONLY = "security_only"  # Only security updates
    PATCH_ONLY = "patch_only"     # Only patch version updates
    FROZEN = "frozen"             # No updates allowed


@dataclass
class DependencyInfo:
    """Information about a specific dependency."""
    name: str
    current_version: Optional[str]
    latest_version: Optional[str]
    dependency_type: DependencyType
    file_path: str
    ecosystem: str  # python, nodejs, etc.
    constraint: Optional[str] = None  # Version constraint from file
    license: Optional[str] = None
    security_vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    update_available: bool = False
    is_outdated: bool = False
    days_behind: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityVulnerability:
    """Security vulnerability information."""
    vulnerability_id: str  # CVE, GHSA, etc.
    severity: SecuritySeverity
    title: str
    description: str
    affected_versions: List[str]
    fixed_versions: List[str]
    published_date: datetime
    cvss_score: Optional[float] = None
    cwe_ids: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)


@dataclass
class DependencyAuditReport:
    """Comprehensive dependency audit report."""
    audit_id: str
    repository_path: str
    start_time: datetime
    end_time: Optional[datetime] = None
    dependencies: List[DependencyInfo] = field(default_factory=list)
    vulnerabilities: List[SecurityVulnerability] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    update_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DepManagerAgent(BaseAgent):
    """
    Advanced Dependency Manager Agent for comprehensive dependency management.
    
    Handles dependency tracking, version management, security vulnerability scanning,
    automated updates, and cross-repository compatibility analysis.
    """

    def __init__(
        self,
        agent_id: str,
        config: Config,
        shared_memory: SharedMemory,
        vector_db: VectorDatabase,
        llm_provider: Optional[LLMProvider] = None
    ):
        super().__init__(agent_id, config, shared_memory, vector_db)
        self.llm_provider = llm_provider
        
        # Initialize dependency cache
        self.dependency_cache: Dict[str, DependencyAuditReport] = {}
        self.vulnerability_database: Dict[str, List[SecurityVulnerability]] = {}
        
        # Common dependency file patterns
        self.dependency_files = {
            "python": ["requirements.txt", "pyproject.toml", "setup.py", "Pipfile"],
            "nodejs": ["package.json", "yarn.lock", "package-lock.json"],
            "java": ["pom.xml", "build.gradle", "gradle.properties"],
            "ruby": ["Gemfile", "Gemfile.lock"],
            "php": ["composer.json", "composer.lock"],
            "go": ["go.mod", "go.sum"],
            "rust": ["Cargo.toml", "Cargo.lock"],
            "docker": ["Dockerfile", "docker-compose.yml"]
        }

    async def execute(self, state: Any) -> Any:
        """Execute the dependency manager agent's main logic."""
        if isinstance(state, dict):
            task = state
        else:
            task = {"type": "dependency_audit", "description": str(state)}
        
        return await self.execute_task(task)

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a dependency management task."""
        try:
            task_type = task.get("type", "dependency_audit")
            self._update_state("working", task.get("task_id"))
            
            if task_type == "dependency_audit":
                result = await self._perform_dependency_audit(task)
            elif task_type == "security_scan":
                result = await self._perform_security_scan(task)
            elif task_type == "version_check":
                result = await self._check_version_updates(task)
            elif task_type == "dependency_update":
                result = await self._update_dependencies(task)
            elif task_type == "compatibility_analysis":
                result = await self._analyze_compatibility(task)
            else:
                result = {
                    "success": False,
                    "error": f"Unknown task type: {task_type}"
                }
            
            self._update_state("idle")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in dependency manager: {e}")
            self._update_state("error", error=str(e))
            return {"success": False, "error": str(e)}

    async def _perform_dependency_audit(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive dependency audit for a repository."""
        try:
            repository_path = task.get("repository_path")
            if not repository_path:
                return {"success": False, "error": "Repository path required"}
            
            repo_path = Path(repository_path)
            if not repo_path.exists():
                return {"success": False, "error": f"Repository path does not exist: {repository_path}"}
            
            audit_id = f"dep_audit_{hashlib.md5(str(repo_path).encode()).hexdigest()}_{int(datetime.now().timestamp())}"
            
            # Create audit report
            report = DependencyAuditReport(
                audit_id=audit_id,
                repository_path=str(repo_path),
                start_time=datetime.now(timezone.utc)
            )
            
            # Discover dependency files
            dependency_files = await self._discover_dependency_files(repo_path)
            
            # Parse dependencies from each file
            all_dependencies = []
            for file_path, ecosystem in dependency_files.items():
                try:
                    file_dependencies = await self._parse_dependency_file(file_path, ecosystem)
                    all_dependencies.extend(file_dependencies)
                except Exception as e:
                    self.logger.warning(f"Error parsing {file_path}: {e}")
            
            # Get latest version information
            for dep in all_dependencies:
                try:
                    await self._enrich_dependency_info(dep)
                except Exception as e:
                    self.logger.warning(f"Error enriching {dep.name}: {e}")
            
            report.dependencies = all_dependencies
            report.end_time = datetime.now(timezone.utc)
            
            # Generate statistics
            report.statistics = self._generate_dependency_statistics(all_dependencies)
            
            # Generate update recommendations
            report.update_recommendations = self._generate_update_recommendations(all_dependencies)
            
            # Cache the report
            self.dependency_cache[str(repo_path)] = report
            
            # Log results
            self.log_observation(
                f"Completed dependency audit: {repository_path}",
                data={
                    "audit_id": audit_id,
                    "dependencies_found": len(all_dependencies),
                    "update_recommendations": len(report.update_recommendations),
                    "statistics": report.statistics
                }
            )
            
            return {
                "success": True,
                "audit_id": audit_id,
                "repository_path": str(repo_path),
                "audit_report": asdict(report),
                "summary": self._generate_audit_summary(report)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _discover_dependency_files(self, repo_path: Path) -> Dict[Path, str]:
        """Discover dependency files in the repository."""
        discovered_files = {}
        
        for ecosystem, file_patterns in self.dependency_files.items():
            for pattern in file_patterns:
                # Check root directory
                file_path = repo_path / pattern
                if file_path.exists():
                    discovered_files[file_path] = ecosystem
                
                # Search recursively for certain files
                if pattern in ["requirements.txt", "package.json", "Cargo.toml"]:
                    for found_file in repo_path.rglob(pattern):
                        if found_file not in discovered_files:
                            discovered_files[found_file] = ecosystem
        
        return discovered_files

    async def _parse_dependency_file(self, file_path: Path, ecosystem: str) -> List[DependencyInfo]:
        """Parse dependencies from a specific file."""
        dependencies = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if ecosystem == "python":
                dependencies = self._parse_python_dependencies(file_path, content)
            elif ecosystem == "nodejs":
                dependencies = self._parse_nodejs_dependencies(file_path, content)
            elif ecosystem == "java":
                dependencies = self._parse_java_dependencies(file_path, content)
            else:
                self.logger.warning(f"Unsupported ecosystem: {ecosystem}")
        
        except Exception as e:
            self.logger.error(f"Error reading {file_path}: {e}")
        
        return dependencies

    def _parse_python_dependencies(self, file_path: Path, content: str) -> List[DependencyInfo]:
        """Parse Python dependency files."""
        dependencies = []
        
        if file_path.name == "requirements.txt":
            for line_num, line in enumerate(content.split('\n'), 1):
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('-'):
                    dep_info = self._parse_python_requirement_line(line, str(file_path))
                    if dep_info:
                        dependencies.append(dep_info)
        
        elif file_path.name == "pyproject.toml":
            dependencies = self._parse_pyproject_toml(content, str(file_path))
        
        return dependencies

    def _parse_python_requirement_line(self, line: str, file_path: str) -> Optional[DependencyInfo]:
        """Parse a single Python requirement line."""
        # Handle different requirement formats: package==1.0, package>=1.0, package
        match = re.match(r'^([a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]?)([<>=!]+.+)?$', line)
        if not match:
            return None
        
        name = match.group(1)
        constraint = match.group(2) if match.group(2) else None
        
        # Extract version if exact constraint
        current_version = None
        if constraint and '==' in constraint:
            current_version = constraint.replace('==', '').strip()
        
        return DependencyInfo(
            name=name,
            current_version=current_version,
            latest_version=None,  # Will be filled later
            dependency_type=DependencyType.PRODUCTION,
            file_path=file_path,
            ecosystem="python",
            constraint=constraint
        )

    def _parse_pyproject_toml(self, content: str, file_path: str) -> List[DependencyInfo]:
        """Parse pyproject.toml file."""
        dependencies = []
        
        try:
            # Basic TOML parsing without external dependency
            lines = content.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                
                # Track current section
                if line.startswith('['):
                    current_section = line.strip('[]')
                    continue
                
                # Parse dependencies in relevant sections
                if current_section in ['project.dependencies', 'tool.poetry.dependencies']:
                    if '=' in line and not line.startswith('#'):
                        parts = line.split('=', 1)
                        if len(parts) == 2:
                            name = parts[0].strip().strip('"\'')
                            version_spec = parts[1].strip().strip('"\'')
                            
                            if name and name != 'python':
                                dep_info = DependencyInfo(
                                    name=name,
                                    current_version=None,
                                    latest_version=None,
                                    dependency_type=DependencyType.PRODUCTION,
                                    file_path=file_path,
                                    ecosystem="python",
                                    constraint=version_spec
                                )
                                dependencies.append(dep_info)
        
        except Exception as e:
            self.logger.error(f"Error parsing pyproject.toml: {e}")
        
        return dependencies

    def _parse_nodejs_dependencies(self, file_path: Path, content: str) -> List[DependencyInfo]:
        """Parse Node.js package.json dependencies."""
        dependencies = []
        
        try:
            if file_path.name == "package.json":
                data = json.loads(content)
                
                # Parse production dependencies
                if "dependencies" in data:
                    for name, version in data["dependencies"].items():
                        dependencies.append(DependencyInfo(
                            name=name,
                            current_version=version.lstrip('^~>=<'),
                            latest_version=None,
                            dependency_type=DependencyType.PRODUCTION,
                            file_path=str(file_path),
                            ecosystem="nodejs",
                            constraint=version
                        ))
                
                # Parse development dependencies
                if "devDependencies" in data:
                    for name, version in data["devDependencies"].items():
                        dependencies.append(DependencyInfo(
                            name=name,
                            current_version=version.lstrip('^~>=<'),
                            latest_version=None,
                            dependency_type=DependencyType.DEVELOPMENT,
                            file_path=str(file_path),
                            ecosystem="nodejs",
                            constraint=version
                        ))
        
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing package.json: {e}")
        
        return dependencies

    def _parse_java_dependencies(self, file_path: Path, content: str) -> List[DependencyInfo]:
        """Parse Java dependencies from pom.xml or build.gradle."""
        dependencies = []
        
        if file_path.name == "pom.xml":
            # Basic XML parsing for Maven dependencies
            dependency_pattern = r'<dependency>.*?<groupId>(.*?)</groupId>.*?<artifactId>(.*?)</artifactId>.*?<version>(.*?)</version>.*?</dependency>'
            matches = re.findall(dependency_pattern, content, re.DOTALL)
            
            for group_id, artifact_id, version in matches:
                name = f"{group_id.strip()}:{artifact_id.strip()}"
                dependencies.append(DependencyInfo(
                    name=name,
                    current_version=version.strip(),
                    latest_version=None,
                    dependency_type=DependencyType.PRODUCTION,
                    file_path=str(file_path),
                    ecosystem="java",
                    constraint=version.strip()
                ))
        
        elif file_path.name in ["build.gradle", "build.gradle.kts"]:
            # Basic Gradle parsing
            dependency_patterns = [
                r"implementation\s+['\"]([^:]+):([^:]+):([^'\"]+)['\"]",
                r"compile\s+['\"]([^:]+):([^:]+):([^'\"]+)['\"]",
                r"api\s+['\"]([^:]+):([^:]+):([^'\"]+)['\"]"
            ]
            
            for pattern in dependency_patterns:
                matches = re.findall(pattern, content)
                for group_id, artifact_id, version in matches:
                    name = f"{group_id.strip()}:{artifact_id.strip()}"
                    dependencies.append(DependencyInfo(
                        name=name,
                        current_version=version.strip(),
                        latest_version=None,
                        dependency_type=DependencyType.PRODUCTION,
                        file_path=str(file_path),
                        ecosystem="java",
                        constraint=version.strip()
                    ))
        
        return dependencies

    async def _enrich_dependency_info(self, dep: DependencyInfo) -> None:
        """Enrich dependency with latest version and security information."""
        try:
            # Get latest version information
            latest_version = await self._get_latest_version(dep.name, dep.ecosystem)
            if latest_version:
                dep.latest_version = latest_version
                dep.update_available = self._is_update_available(dep.current_version, latest_version)
                dep.is_outdated = dep.update_available
        
        except Exception as e:
            self.logger.warning(f"Error enriching {dep.name}: {e}")

    async def _get_latest_version(self, package_name: str, ecosystem: str) -> Optional[str]:
        """Get the latest version of a package."""
        try:
            if ecosystem == "python":
                return await self._get_pypi_latest_version(package_name)
            elif ecosystem == "nodejs":
                return await self._get_npm_latest_version(package_name)
            elif ecosystem == "java":
                return await self._get_maven_latest_version(package_name)
        
        except Exception as e:
            self.logger.warning(f"Error getting latest version for {package_name}: {e}")
        
        return None

    async def _get_pypi_latest_version(self, package_name: str) -> Optional[str]:
        """Get latest version from PyPI."""
        try:
            result = subprocess.run(
                ["python", "-m", "pip", "index", "versions", package_name],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Parse output to extract latest version
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if "Available versions:" in line:
                        # Extract first version (latest)
                        versions_part = line.split("Available versions:")[1].strip()
                        if versions_part:
                            latest = versions_part.split(',')[0].strip()
                            return latest
        
        except Exception as e:
            self.logger.debug(f"Error checking PyPI for {package_name}: {e}")
        
        return None

    async def _get_npm_latest_version(self, package_name: str) -> Optional[str]:
        """Get latest version from NPM."""
        try:
            result = subprocess.run(
                ["npm", "view", package_name, "version"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
        
        except Exception as e:
            self.logger.debug(f"Error checking NPM for {package_name}: {e}")
        
        return None

    async def _get_maven_latest_version(self, package_name: str) -> Optional[str]:
        """Get latest version from Maven Central (simplified)."""
        # This would require more complex implementation
        # For now, return None (could integrate with Maven API)
        return None

    def _is_update_available(self, current: Optional[str], latest: Optional[str]) -> bool:
        """Check if an update is available."""
        if not current or not latest:
            return False
        
        try:
            # Simple version comparison (could use packaging library for better comparison)
            current_parts = [int(x) for x in current.split('.')]
            latest_parts = [int(x) for x in latest.split('.')]
            
            # Pad shorter version with zeros
            max_len = max(len(current_parts), len(latest_parts))
            current_parts.extend([0] * (max_len - len(current_parts)))
            latest_parts.extend([0] * (max_len - len(latest_parts)))
            
            return latest_parts > current_parts
        
        except (ValueError, AttributeError):
            return False

    def _generate_dependency_statistics(self, dependencies: List[DependencyInfo]) -> Dict[str, Any]:
        """Generate statistics from dependency analysis."""
        stats = {
            "total_dependencies": len(dependencies),
            "by_ecosystem": {},
            "by_type": {},
            "updates_available": 0,
            "outdated_dependencies": 0,
            "security_vulnerabilities": 0
        }
        
        for dep in dependencies:
            # Count by ecosystem
            ecosystem = dep.ecosystem
            stats["by_ecosystem"][ecosystem] = stats["by_ecosystem"].get(ecosystem, 0) + 1
            
            # Count by type
            dep_type = dep.dependency_type.value
            stats["by_type"][dep_type] = stats["by_type"].get(dep_type, 0) + 1
            
            # Count updates and issues
            if dep.update_available:
                stats["updates_available"] += 1
            
            if dep.is_outdated:
                stats["outdated_dependencies"] += 1
            
            if dep.security_vulnerabilities:
                stats["security_vulnerabilities"] += len(dep.security_vulnerabilities)
        
        return stats

    def _generate_update_recommendations(self, dependencies: List[DependencyInfo]) -> List[Dict[str, Any]]:
        """Generate update recommendations based on dependency analysis."""
        recommendations = []
        
        for dep in dependencies:
            if dep.update_available and dep.latest_version:
                recommendation = {
                    "package_name": dep.name,
                    "current_version": dep.current_version,
                    "latest_version": dep.latest_version,
                    "ecosystem": dep.ecosystem,
                    "file_path": dep.file_path,
                    "priority": "medium",
                    "reason": "Update available"
                }
                
                # Determine priority based on security vulnerabilities
                if dep.security_vulnerabilities:
                    critical_vulns = [v for v in dep.security_vulnerabilities 
                                    if v.get("severity") in ["critical", "high"]]
                    if critical_vulns:
                        recommendation["priority"] = "high"
                        recommendation["reason"] = f"Security vulnerabilities found: {len(critical_vulns)}"
                
                recommendations.append(recommendation)
        
        # Sort by priority (high first)
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))
        
        return recommendations

    def _generate_audit_summary(self, report: DependencyAuditReport) -> str:
        """Generate human-readable audit summary."""
        stats = report.statistics
        summary_parts = [
            f"Dependency Audit Summary for {report.repository_path}:",
            f"• Found {stats.get('total_dependencies', 0)} dependencies",
            f"• {stats.get('updates_available', 0)} updates available",
            f"• {stats.get('outdated_dependencies', 0)} outdated dependencies",
        ]
        
        if stats.get('security_vulnerabilities', 0) > 0:
            summary_parts.append(f"• ⚠️  {stats['security_vulnerabilities']} security vulnerabilities found")
        
        if report.update_recommendations:
            high_priority = len([r for r in report.update_recommendations if r["priority"] == "high"])
            if high_priority > 0:
                summary_parts.append(f"• 🚨 {high_priority} high-priority updates recommended")
        
        return "\n".join(summary_parts)

    async def _perform_security_scan(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive security vulnerability scanning."""
        try:
            repository_path = task.get("repository_path")
            scan_type = task.get("scan_type", "comprehensive")  # comprehensive, quick, critical_only
            include_dev_dependencies = task.get("include_dev_dependencies", True)
            
            if not repository_path:
                return {"success": False, "error": "Repository path required"}
            
            repo_path = Path(repository_path)
            if not repo_path.exists():
                return {"success": False, "error": f"Repository path does not exist: {repository_path}"}
            
            scan_id = f"security_scan_{hashlib.md5(str(repo_path).encode()).hexdigest()}_{int(datetime.now().timestamp())}"
            
            vulnerabilities = []
            scan_stats = {
                "dependencies_scanned": 0,
                "vulnerabilities_found": 0,
                "critical_vulns": 0,
                "high_vulns": 0,
                "medium_vulns": 0,
                "low_vulns": 0,
                "scan_duration": 0,
                "tools_used": []
            }
            
            scan_start = datetime.now()
            
            # Get or create dependency audit
            if str(repo_path) in self.dependency_cache:
                report = self.dependency_cache[str(repo_path)]
                dependencies = report.dependencies
            else:
                # Perform quick dependency discovery
                dependency_files = await self._discover_dependency_files(repo_path)
                dependencies = []
                for file_path, ecosystem in dependency_files.items():
                    file_dependencies = await self._parse_dependency_file(file_path, ecosystem)
                    dependencies.extend(file_dependencies)
            
            # Filter dependencies based on scan parameters
            filtered_dependencies = []
            for dep in dependencies:
                if not include_dev_dependencies and dep.dependency_type == DependencyType.DEVELOPMENT:
                    continue
                filtered_dependencies.append(dep)
            
            scan_stats["dependencies_scanned"] = len(filtered_dependencies)
            
            # Enhanced security scanning
            for dep in filtered_dependencies:
                try:
                    dep_vulnerabilities = await self._enhanced_vulnerability_scan(dep, scan_type)
                    vulnerabilities.extend(dep_vulnerabilities)
                    
                    # Update dependency with vulnerability info
                    dep.security_vulnerabilities = [asdict(v) for v in dep_vulnerabilities]
                    
                except Exception as e:
                    self.logger.warning(f"Error scanning {dep.name}: {e}")
            
            # Additional security checks
            additional_vulns = await self._perform_additional_security_checks(repo_path, scan_type)
            vulnerabilities.extend(additional_vulns)
            
            # Calculate statistics
            scan_stats["vulnerabilities_found"] = len(vulnerabilities)
            for vuln in vulnerabilities:
                if vuln.severity == SecuritySeverity.CRITICAL:
                    scan_stats["critical_vulns"] += 1
                elif vuln.severity == SecuritySeverity.HIGH:
                    scan_stats["high_vulns"] += 1
                elif vuln.severity == SecuritySeverity.MEDIUM:
                    scan_stats["medium_vulns"] += 1
                elif vuln.severity == SecuritySeverity.LOW:
                    scan_stats["low_vulns"] += 1
            
            scan_end = datetime.now()
            scan_stats["scan_duration"] = (scan_end - scan_start).total_seconds()
            
            # Generate comprehensive security report
            security_report = {
                "scan_id": scan_id,
                "repository_path": str(repo_path),
                "scan_type": scan_type,
                "scan_timestamp": datetime.now(timezone.utc).isoformat(),
                "scan_statistics": scan_stats,
                "vulnerabilities": [asdict(v) for v in vulnerabilities],
                "risk_summary": self._calculate_risk_summary(vulnerabilities),
                "compliance_status": self._assess_security_compliance(vulnerabilities),
                "remediation_plan": self._generate_remediation_plan(vulnerabilities, filtered_dependencies)
            }
            
            # Log security scan results
            self.log_observation(
                f"Completed security scan: {repository_path}",
                data={
                    "scan_id": scan_id,
                    "scan_type": scan_type,
                    "vulnerabilities_found": len(vulnerabilities),
                    "critical_vulns": scan_stats["critical_vulns"],
                    "scan_duration": scan_stats["scan_duration"]
                }
            )
            
            # Store scan results in cache for future reference
            self.vulnerability_database[scan_id] = vulnerabilities
            
            return {
                "success": True,
                "security_report": security_report,
                "summary": self._generate_security_summary(security_report)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _enhanced_vulnerability_scan(self, dependency: DependencyInfo, scan_type: str) -> List[SecurityVulnerability]:
        """Enhanced vulnerability scanning with multiple tools and databases."""
        vulnerabilities = []
        
        try:
            if dependency.ecosystem == "python":
                vulnerabilities.extend(await self._scan_python_vulnerabilities_enhanced(dependency, scan_type))
            elif dependency.ecosystem == "nodejs":
                vulnerabilities.extend(await self._scan_nodejs_vulnerabilities_enhanced(dependency, scan_type))
            elif dependency.ecosystem == "java":
                vulnerabilities.extend(await self._scan_java_vulnerabilities(dependency, scan_type))
            
            # Add OWASP dependency check if available
            owasp_vulns = await self._scan_owasp_dependency_check(dependency)
            vulnerabilities.extend(owasp_vulns)
            
        except Exception as e:
            self.logger.warning(f"Error in enhanced scan for {dependency.name}: {e}")
        
        return vulnerabilities

    async def _scan_python_vulnerabilities_enhanced(self, dependency: DependencyInfo, scan_type: str) -> List[SecurityVulnerability]:
        """Enhanced Python vulnerability scanning with multiple sources."""
        vulnerabilities = []
        
        # Try safety first (existing implementation)
        safety_vulns = await self._scan_python_vulnerabilities(dependency)
        vulnerabilities.extend(safety_vulns)
        
        # Try pip-audit if available
        try:
            result = subprocess.run(
                ["pip-audit", "--desc", "--format=json", f"{dependency.name}=={dependency.current_version or 'latest'}"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0 and result.stdout:
                audit_data = json.loads(result.stdout)
                for vuln_data in audit_data.get("vulnerabilities", []):
                    vulnerability = SecurityVulnerability(
                        vulnerability_id=vuln_data.get("id", "unknown"),
                        severity=self._map_severity(vuln_data.get("fix_versions", [])),
                        title=vuln_data.get("description", "Security vulnerability"),
                        description=vuln_data.get("description", ""),
                        affected_versions=[dependency.current_version or "unknown"],
                        fixed_versions=vuln_data.get("fix_versions", []),
                        published_date=datetime.now(timezone.utc)
                    )
                    vulnerabilities.append(vulnerability)
        
        except Exception as e:
            self.logger.debug(f"pip-audit not available or error: {e}")
        
        return vulnerabilities

    async def _scan_nodejs_vulnerabilities_enhanced(self, dependency: DependencyInfo, scan_type: str) -> List[SecurityVulnerability]:
        """Enhanced Node.js vulnerability scanning."""
        vulnerabilities = []
        
        # Use existing npm audit
        npm_vulns = await self._scan_nodejs_vulnerabilities(dependency)
        vulnerabilities.extend(npm_vulns)
        
        # Try yarn audit if available
        try:
            result = subprocess.run(
                ["yarn", "audit", "--json"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=Path(dependency.file_path).parent
            )
            
            if result.returncode != 0:
                for line in result.stdout.strip().split('\n'):
                    try:
                        audit_data = json.loads(line)
                        if audit_data.get("type") == "auditAdvisory":
                            advisory = audit_data.get("data", {})
                            if advisory.get("module_name") == dependency.name:
                                vulnerability = SecurityVulnerability(
                                    vulnerability_id=str(advisory.get("id", "unknown")),
                                    severity=self._map_npm_severity(advisory.get("severity", "medium")),
                                    title=advisory.get("title", "Security vulnerability"),
                                    description=advisory.get("overview", ""),
                                    affected_versions=advisory.get("vulnerable_versions", []),
                                    fixed_versions=advisory.get("patched_versions", []),
                                    published_date=datetime.now(timezone.utc),
                                    cvss_score=advisory.get("cvss_score"),
                                    cwe_ids=advisory.get("cwe", [])
                                )
                                vulnerabilities.append(vulnerability)
                    except json.JSONDecodeError:
                        continue
        
        except Exception as e:
            self.logger.debug(f"Yarn audit not available or error: {e}")
        
        return vulnerabilities

    async def _scan_java_vulnerabilities(self, dependency: DependencyInfo, scan_type: str) -> List[SecurityVulnerability]:
        """Scan Java dependencies for vulnerabilities."""
        vulnerabilities = []
        
        try:
            # Try OWASP Dependency Check for Java
            if scan_type in ["comprehensive", "critical_only"]:
                result = subprocess.run(
                    ["dependency-check", "--project", "security-scan", 
                     "--scan", Path(dependency.file_path).parent,
                     "--format", "JSON", "--prettyPrint"],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                if result.returncode == 0 and result.stdout:
                    # Parse OWASP dependency check results
                    # This would require more complex JSON parsing
                    pass
        
        except Exception as e:
            self.logger.debug(f"OWASP Dependency Check not available: {e}")
        
        return vulnerabilities

    async def _scan_owasp_dependency_check(self, dependency: DependencyInfo) -> List[SecurityVulnerability]:
        """Use OWASP Dependency Check for cross-ecosystem vulnerability scanning."""
        vulnerabilities = []
        
        try:
            # This would implement OWASP Dependency Check integration
            # For now, return empty list as it requires external tool setup
            pass
        
        except Exception as e:
            self.logger.debug(f"OWASP scan error for {dependency.name}: {e}")
        
        return vulnerabilities

    async def _perform_additional_security_checks(self, repo_path: Path, scan_type: str) -> List[SecurityVulnerability]:
        """Perform additional repository-level security checks."""
        vulnerabilities = []
        
        try:
            # Check for security configuration files
            security_files = [
                ".github/dependabot.yml",
                ".dependabot/config.yml", 
                "security.md",
                ".security.yml"
            ]
            
            missing_security_config = []
            for security_file in security_files:
                if not (repo_path / security_file).exists():
                    missing_security_config.append(security_file)
            
            if len(missing_security_config) >= 3:  # Missing most security configs
                vulnerability = SecurityVulnerability(
                    vulnerability_id="DEVGUARD-SEC-001",
                    severity=SecuritySeverity.LOW,
                    title="Missing security configuration",
                    description=f"Repository missing security configuration files: {', '.join(missing_security_config)}",
                    affected_versions=["current"],
                    fixed_versions=["with security config"],
                    published_date=datetime.now(timezone.utc)
                )
                vulnerabilities.append(vulnerability)
            
            # Check for outdated dependencies (more than 1 year old)
            if scan_type == "comprehensive":
                old_deps = await self._check_for_ancient_dependencies(repo_path)
                for dep_name, age_days in old_deps.items():
                    if age_days > 365:  # More than 1 year old
                        vulnerability = SecurityVulnerability(
                            vulnerability_id=f"DEVGUARD-AGE-{hashlib.md5(dep_name.encode()).hexdigest()[:8]}",
                            severity=SecuritySeverity.MEDIUM,
                            title=f"Ancient dependency: {dep_name}",
                            description=f"Dependency {dep_name} is {age_days} days old and may have unpatched vulnerabilities",
                            affected_versions=["current"],
                            fixed_versions=["latest"],
                            published_date=datetime.now(timezone.utc)
                        )
                        vulnerabilities.append(vulnerability)
        
        except Exception as e:
            self.logger.debug(f"Additional security checks error: {e}")
        
        return vulnerabilities

    async def _check_for_ancient_dependencies(self, repo_path: Path) -> Dict[str, int]:
        """Check for very old dependencies that may pose security risks."""
        # This would implement age checking - placeholder for now
        return {}

    def _map_severity(self, fix_versions: List[str]) -> SecuritySeverity:
        """Map vulnerability data to severity level."""
        if not fix_versions:
            return SecuritySeverity.HIGH  # No fix available is serious
        return SecuritySeverity.MEDIUM

    def _map_npm_severity(self, npm_severity: str) -> SecuritySeverity:
        """Map NPM severity to our severity enum."""
        severity_map = {
            "critical": SecuritySeverity.CRITICAL,
            "high": SecuritySeverity.HIGH,
            "moderate": SecuritySeverity.MEDIUM,
            "medium": SecuritySeverity.MEDIUM,
            "low": SecuritySeverity.LOW,
            "info": SecuritySeverity.INFORMATIONAL
        }
        return severity_map.get(npm_severity.lower(), SecuritySeverity.MEDIUM)

    def _assess_security_compliance(self, vulnerabilities: List[SecurityVulnerability]) -> Dict[str, Any]:
        """Assess security compliance status based on vulnerabilities."""
        critical_count = len([v for v in vulnerabilities if v.severity == SecuritySeverity.CRITICAL])
        high_count = len([v for v in vulnerabilities if v.severity == SecuritySeverity.HIGH])
        
        compliance_status = "compliant"
        compliance_score = 100
        
        if critical_count > 0:
            compliance_status = "non_compliant"
            compliance_score = max(0, 100 - (critical_count * 30))
        elif high_count > 2:
            compliance_status = "at_risk"
            compliance_score = max(50, 100 - (high_count * 10))
        elif high_count > 0:
            compliance_status = "needs_attention"
            compliance_score = max(70, 100 - (high_count * 5))
        
        return {
            "status": compliance_status,
            "score": compliance_score,
            "critical_vulnerabilities": critical_count,
            "high_vulnerabilities": high_count,
            "requires_immediate_action": critical_count > 0 or high_count > 3,
            "compliance_notes": self._generate_compliance_notes(critical_count, high_count)
        }

    def _generate_compliance_notes(self, critical_count: int, high_count: int) -> List[str]:
        """Generate compliance notes based on vulnerability counts."""
        notes = []
        
        if critical_count > 0:
            notes.append(f"🚨 {critical_count} critical vulnerabilities require immediate remediation")
        
        if high_count > 3:
            notes.append(f"⚠️ {high_count} high-severity vulnerabilities exceed acceptable risk threshold")
        elif high_count > 0:
            notes.append(f"⚠️ {high_count} high-severity vulnerabilities should be addressed soon")
        
        if critical_count == 0 and high_count <= 2:
            notes.append("✅ Security posture within acceptable parameters")
        
        return notes

    def _generate_remediation_plan(self, vulnerabilities: List[SecurityVulnerability], dependencies: List[DependencyInfo]) -> Dict[str, Any]:
        """Generate a comprehensive remediation plan."""
        remediation_plan = {
            "immediate_actions": [],
            "short_term_actions": [],
            "long_term_actions": [],
            "estimated_effort": "medium",
            "priority_order": []
        }
        
        # Group vulnerabilities by affected dependency
        vuln_by_dep = {}
        for vuln in vulnerabilities:
            # Find affected dependency
            affected_deps = [d for d in dependencies if d.name in vuln.description or d.name == vuln.title.split()[-1]]
            for dep in affected_deps:
                if dep.name not in vuln_by_dep:
                    vuln_by_dep[dep.name] = []
                vuln_by_dep[dep.name].append(vuln)
        
        # Generate actions based on severity
        for dep_name, dep_vulns in vuln_by_dep.items():
            critical_vulns = [v for v in dep_vulns if v.severity == SecuritySeverity.CRITICAL]
            high_vulns = [v for v in dep_vulns if v.severity == SecuritySeverity.HIGH]
            
            if critical_vulns:
                remediation_plan["immediate_actions"].append({
                    "action": f"Update {dep_name} immediately",
                    "reason": f"{len(critical_vulns)} critical vulnerabilities",
                    "vulnerability_ids": [v.vulnerability_id for v in critical_vulns]
                })
            
            elif high_vulns:
                remediation_plan["short_term_actions"].append({
                    "action": f"Update {dep_name} within 48 hours",
                    "reason": f"{len(high_vulns)} high-severity vulnerabilities",
                    "vulnerability_ids": [v.vulnerability_id for v in high_vulns]
                })
        
        # Add long-term recommendations
        if len(vulnerabilities) > 10:
            remediation_plan["long_term_actions"].append({
                "action": "Implement automated dependency scanning in CI/CD",
                "reason": "High vulnerability count indicates need for continuous monitoring"
            })
        
        # Estimate effort
        total_critical = len([v for v in vulnerabilities if v.severity == SecuritySeverity.CRITICAL])
        total_high = len([v for v in vulnerabilities if v.severity == SecuritySeverity.HIGH])
        
        if total_critical > 3 or total_high > 10:
            remediation_plan["estimated_effort"] = "high"
        elif total_critical > 0 or total_high > 5:
            remediation_plan["estimated_effort"] = "medium"
        else:
            remediation_plan["estimated_effort"] = "low"
        
        return remediation_plan

    def _generate_security_summary(self, security_report: Dict[str, Any]) -> str:
        """Generate human-readable security scan summary."""
        stats = security_report["scan_statistics"]
        compliance = security_report["compliance_status"]
        
        summary_parts = [
            f"Security Scan Summary for {security_report['repository_path']}:",
            f"• Scanned {stats['dependencies_scanned']} dependencies",
            f"• Found {stats['vulnerabilities_found']} vulnerabilities",
        ]
        
        if stats["critical_vulns"] > 0:
            summary_parts.append(f"• 🚨 {stats['critical_vulns']} CRITICAL vulnerabilities")
        
        if stats["high_vulns"] > 0:
            summary_parts.append(f"• ⚠️ {stats['high_vulns']} high-severity vulnerabilities")
        
        summary_parts.append(f"• Compliance Status: {compliance['status'].upper()}")
        summary_parts.append(f"• Security Score: {compliance['score']}/100")
        
        if compliance["requires_immediate_action"]:
            summary_parts.append("• 🚨 IMMEDIATE ACTION REQUIRED")
        
        return "\n".join(summary_parts)

    async def _scan_dependency_vulnerabilities(self, dependency: DependencyInfo) -> List[SecurityVulnerability]:
        """Scan a specific dependency for vulnerabilities."""
        vulnerabilities = []
        
        try:
            if dependency.ecosystem == "python":
                vulnerabilities = await self._scan_python_vulnerabilities(dependency)
            elif dependency.ecosystem == "nodejs":
                vulnerabilities = await self._scan_nodejs_vulnerabilities(dependency)
        
        except Exception as e:
            self.logger.warning(f"Error scanning {dependency.name}: {e}")
        
        return vulnerabilities

    async def _scan_python_vulnerabilities(self, dependency: DependencyInfo) -> List[SecurityVulnerability]:
        """Scan Python package vulnerabilities using safety or similar."""
        vulnerabilities = []
        
        try:
            # Try to use safety if available
            result = subprocess.run(
                ["safety", "check", "--json", "--key", ""],
                input=f"{dependency.name}=={dependency.current_version or 'latest'}",
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and result.stdout:
                safety_data = json.loads(result.stdout)
                for vuln in safety_data:
                    vulnerability = SecurityVulnerability(
                        vulnerability_id=vuln.get("id", "unknown"),
                        severity=SecuritySeverity.MEDIUM,  # Default, could parse from data
                        title=vuln.get("advisory", "Security vulnerability"),
                        description=vuln.get("advisory", ""),
                        affected_versions=vuln.get("vulnerable_versions", []),
                        fixed_versions=[],  # Would need to parse
                        published_date=datetime.now(timezone.utc)  # Would need actual date
                    )
                    vulnerabilities.append(vulnerability)
        
        except subprocess.TimeoutExpired:
            self.logger.warning(f"Timeout scanning {dependency.name}")
        except Exception as e:
            self.logger.debug(f"Safety not available or error: {e}")
        
        return vulnerabilities

    async def _scan_nodejs_vulnerabilities(self, dependency: DependencyInfo) -> List[SecurityVulnerability]:
        """Scan Node.js package vulnerabilities."""
        vulnerabilities = []
        
        try:
            # Use npm audit if available
            result = subprocess.run(
                ["npm", "audit", "--json"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=Path(dependency.file_path).parent
            )
            
            if result.returncode != 0 and result.stdout:
                audit_data = json.loads(result.stdout)
                if "vulnerabilities" in audit_data:
                    for vuln_name, vuln_data in audit_data["vulnerabilities"].items():
                        if vuln_name == dependency.name:
                            vulnerability = SecurityVulnerability(
                                vulnerability_id=vuln_data.get("id", "unknown"),
                                severity=SecuritySeverity.MEDIUM,
                                title=vuln_data.get("title", "Security vulnerability"),
                                description=vuln_data.get("overview", ""),
                                affected_versions=vuln_data.get("vulnerable_versions", []),
                                fixed_versions=vuln_data.get("patched_versions", []),
                                published_date=datetime.now(timezone.utc)
                            )
                            vulnerabilities.append(vulnerability)
        
        except Exception as e:
            self.logger.debug(f"NPM audit not available or error: {e}")
        
        return vulnerabilities

    def _calculate_risk_summary(self, vulnerabilities: List[SecurityVulnerability]) -> Dict[str, Any]:
        """Calculate overall risk summary from vulnerabilities."""
        risk_counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "informational": 0
        }
        
        for vuln in vulnerabilities:
            severity = vuln.severity.value.lower()
            risk_counts[severity] = risk_counts.get(severity, 0) + 1
        
        # Calculate overall risk score (0-100)
        risk_score = (
            risk_counts["critical"] * 25 +
            risk_counts["high"] * 15 +
            risk_counts["medium"] * 8 +
            risk_counts["low"] * 3 +
            risk_counts["informational"] * 1
        )
        
        risk_level = "low"
        if risk_score > 50:
            risk_level = "critical"
        elif risk_score > 25:
            risk_level = "high"
        elif risk_score > 10:
            risk_level = "medium"
        
        return {
            "risk_score": min(risk_score, 100),
            "risk_level": risk_level,
            "severity_breakdown": risk_counts,
            "recommendations": self._generate_security_recommendations(vulnerabilities)
        }

    def _generate_security_recommendations(self, vulnerabilities: List[SecurityVulnerability]) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        
        critical_count = len([v for v in vulnerabilities if v.severity == SecuritySeverity.CRITICAL])
        high_count = len([v for v in vulnerabilities if v.severity == SecuritySeverity.HIGH])
        
        if critical_count > 0:
            recommendations.append(f"🚨 Immediately update {critical_count} critical vulnerabilities")
        
        if high_count > 0:
            recommendations.append(f"⚠️ Update {high_count} high-severity vulnerabilities within 24 hours")
        
        if len(vulnerabilities) > 5:
            recommendations.append("Consider implementing automated security scanning in CI/CD")
        
        return recommendations

    async def _check_version_updates(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Check for available version updates."""
        try:
            repository_path = task.get("repository_path")
            if not repository_path:
                return {"success": False, "error": "Repository path required"}
            
            # Leverage existing audit functionality
            audit_result = await self._perform_dependency_audit({"repository_path": repository_path})
            
            if audit_result["success"]:
                audit_report = audit_result["audit_report"]
                return {
                    "success": True,
                    "updates_available": audit_report["statistics"]["updates_available"],
                    "recommendations": audit_report["update_recommendations"],
                    "summary": f"Found {audit_report['statistics']['updates_available']} available updates"
                }
            else:
                return audit_result
                
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _update_dependencies(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Update dependencies based on strategy."""
        try:
            repository_path = task.get("repository_path")
            strategy = task.get("strategy", UpdateStrategy.MANUAL)
            packages = task.get("packages", [])  # Specific packages to update
            
            if not repository_path:
                return {"success": False, "error": "Repository path required"}
            
            # This would implement actual updates - for now, return planning
            return {
                "success": True,
                "message": "Dependency update functionality planned",
                "strategy": strategy.value if hasattr(strategy, 'value') else str(strategy),
                "packages_to_update": packages,
                "note": "Actual updates would require careful testing and validation"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _analyze_compatibility(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cross-repository compatibility."""
        try:
            source_repo = task.get("source_repository")
            target_repos = task.get("target_repositories", [])
            
            if not source_repo:
                return {"success": False, "error": "Source repository required"}
            
            # This would implement compatibility analysis - placeholder for now
            return {
                "success": True,
                "source_repository": source_repo,
                "target_repositories": target_repos,
                "compatibility_analysis": "Compatibility analysis functionality planned",
                "note": "Would analyze shared dependencies and version constraints"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

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
        """Get agent capabilities."""
        return [
            "dependency_tracking",
            "version_management", 
            "security_vulnerability_scanning",
            "dependency_auditing",
            "update_recommendations",
            "compatibility_analysis",
            "automated_updates",
            "risk_assessment"
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            "agent_id": self.agent_id, 
            "type": "dependency_manager",
            "capabilities": self.get_capabilities(),
            "cached_audits": len(self.dependency_cache),
            "supported_ecosystems": list(self.dependency_files.keys())
        }
    
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
        return ["dependency_management"]
    
    def get_status(self) -> Dict[str, Any]:
        return {"agent_id": self.agent_id, "type": "dep_manager"}
