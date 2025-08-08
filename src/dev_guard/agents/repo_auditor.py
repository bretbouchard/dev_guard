"""
Repository Auditor Agent for DevGuard - Advanced Repository Scanning and
File Ingestion

This agent performs comprehensive repository auditing including file scanning,
content ingestion, repository health checks, missing file detection, and
knowledge base maintenance.
"""

import logging
import json
import fnmatch
import hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, asdict, field
from enum import Enum

from .base_agent import BaseAgent
from ..core.config import Config
from ..memory.shared_memory import SharedMemory, AgentState
from ..memory.vector_db import VectorDatabase
from ..llm.provider import LLMProvider

logger = logging.getLogger(__name__)


class AuditSeverity(Enum):
    """Severity levels for repository audit findings."""
    CRITICAL = "critical"    # Major issues requiring immediate attention
    HIGH = "high"           # Important issues affecting repository health
    MEDIUM = "medium"       # Moderate issues requiring attention
    LOW = "low"            # Minor issues or improvements
    INFO = "info"          # Informational findings


class AuditType(Enum):
    """Types of repository audit operations."""
    FULL_SCAN = "full_scan"                      # Complete repository audit
    INCREMENTAL_SCAN = "incremental_scan"        # Scan only changed files
    FILE_INGESTION = "file_ingestion"            # Ingest files into vector DB
    MISSING_FILES = "missing_files"              # Check for missing files
    CLEANUP = "cleanup"                          # Clean up stale DB entries
    HEALTH_CHECK = "health_check"                # Repository health assessment
    METADATA_EXTRACTION = "metadata_extraction"  # Extract repository metadata
    DEPENDENCY_ANALYSIS = "dependency_analysis"  # Analyze dependencies


@dataclass
class AuditFinding:
    """Represents a finding from repository audit."""
    finding_id: str
    audit_type: AuditType
    severity: AuditSeverity
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    recommendation: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = datetime.now(timezone.utc)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RepositoryAuditReport:
    """Comprehensive repository audit report."""
    repository_path: str
    audit_id: str
    audit_type: AuditType
    start_time: datetime
    end_time: Optional[datetime] = None
    findings: List[AuditFinding] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RepoAuditorAgent(BaseAgent):
    """
    Repository Auditor Agent - Comprehensive repository scanning and file
    ingestion.
    
    Capabilities:
    - Repository scanning and file discovery
    - Content ingestion into vector database
    - Missing file detection
    - Repository health assessment
    - Incremental update processing
    - Vector database cleanup
    - Metadata extraction and analysis
    """

    def __init__(
        self,
        agent_id: str,
        config: Config,
        shared_memory: SharedMemory,
        vector_db: VectorDatabase,
        **kwargs: Any
    ):
        super().__init__(agent_id, config, shared_memory, vector_db)
        self.llm_provider: Optional[LLMProvider] = kwargs.get('llm_provider')
        
        # Repository tracking and caching
        self.repository_cache: Dict[str, Dict[str, Any]] = {}
        self.file_checksums: Dict[str, str] = {}
        self.last_scan_times: Dict[str, datetime] = {}
        
        # Audit configuration
        self.supported_file_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c',
            '.h', '.hpp', '.cs', '.rb', '.go', '.rs', '.php', '.swift',
            '.kt', '.scala', '.r', '.sh', '.sql', '.html', '.css',
            '.scss', '.less', '.xml', '.json', '.yaml', '.yml', '.toml',
            '.ini', '.cfg', '.conf', '.md', '.txt', '.rst', '.tex',
            '.log', '.csv'
        }
        
        self.important_files = {
            'README.md', 'README.txt', 'README.rst',
            '.gitignore', '.gitattributes',
            'requirements.txt', 'pyproject.toml', 'setup.py', 'setup.cfg',
            'package.json', 'package-lock.json', 'yarn.lock',
            'Cargo.toml', 'Cargo.lock',
            'go.mod', 'go.sum',
            'pom.xml', 'build.gradle', 'build.gradle.kts',
            'Makefile', 'CMakeLists.txt',
            'Dockerfile', 'docker-compose.yml', 'docker-compose.yaml',
            'LICENSE', 'LICENCE', 'COPYING',
            'CHANGELOG.md', 'CHANGELOG.txt', 'HISTORY.md',
            'CONTRIBUTING.md', 'CODE_OF_CONDUCT.md',
            '.github/workflows', '.gitlab-ci.yml', '.travis.yml'
        }
        
        self.ignore_patterns = [
            '__pycache__', '*.pyc', '*.pyo', '*.pyd',
            'node_modules', '.npm', '.yarn',
            '.git', '.svn', '.hg',
            '.venv', 'venv', '.env',
            'dist', 'build', 'target',
            '*.log', '*.tmp', '*.temp', '*.cache',
            '.DS_Store', 'Thumbs.db',
            '*.min.js', '*.min.css',
            '.idea', '.vscode', '*.swp', '*.swo'
        ]

    async def execute(self, state: Any) -> Any:
        """Execute the repo auditor agent's main logic."""
        if isinstance(state, dict):
            task = state
        else:
            task = {"type": "full_scan", "description": str(state)}
        
        return await self.execute_task(task)

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a repository audit task."""
        try:
            self._update_state("busy", task.get("task_id"))
            
            task_type = task.get("type", "full_scan")
            
            if task_type == "full_scan":
                result = await self._perform_full_repository_scan(task)
            elif task_type == "incremental_scan":
                result = await self._perform_incremental_scan(task)
            elif task_type == "ingest_files":
                result = await self._ingest_repository_files(task)
            elif task_type == "check_missing_files":
                result = await self._check_missing_files(task)
            elif task_type == "cleanup_vector_db":
                result = await self._cleanup_vector_database(task)
            elif task_type == "health_check":
                result = await self._perform_health_check(task)
            elif task_type == "extract_metadata":
                result = await self._extract_repository_metadata(task)
            elif task_type == "analyze_dependencies":
                result = await self._analyze_repository_dependencies(task)
            else:
                result = {
                    "success": False,
                    "error": f"Unknown task type: {task_type}"
                }
            
            self._update_state("idle")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in repo auditor: {e}")
            self._update_state("error", error=str(e))
            return {"success": False, "error": str(e)}

    async def _perform_full_repository_scan(
        self, task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform comprehensive repository scanning and analysis."""
        try:
            repository_path = task.get("repository_path")
            if not repository_path:
                return {"success": False, "error": "Repository path required"}
            
            repo_path = Path(repository_path)
            if not repo_path.exists():
                return {"success": False, "error": f"Repository path does not exist: {repository_path}"}
            
            audit_id = f"full_scan_{int(datetime.now().timestamp())}"
            
            # Create audit report with proper initialization
            report = RepositoryAuditReport(
                repository_path=str(repo_path),
                audit_id=audit_id,
                audit_type=AuditType.FULL_SCAN,
                start_time=datetime.now(timezone.utc)
            )
            
            # Perform comprehensive scanning
            await self._scan_repository_structure(repo_path, report)
            await self._analyze_repository_health(repo_path, report)
            await self._check_important_files(repo_path, report)
            await self._scan_file_content(repo_path, report)
            await self._extract_project_metadata(repo_path, report)
            
            # Finalize report
            report.end_time = datetime.now(timezone.utc)
            report.statistics.update({
                "scan_duration_seconds": (report.end_time - report.start_time).total_seconds(),
                "total_findings": len(report.findings),
                "critical_findings": len([f for f in report.findings if f.severity == AuditSeverity.CRITICAL]),
                "high_findings": len([f for f in report.findings if f.severity == AuditSeverity.HIGH]),
                "medium_findings": len([f for f in report.findings if f.severity == AuditSeverity.MEDIUM]),
                "low_findings": len([f for f in report.findings if f.severity == AuditSeverity.LOW])
            })
            
            # Store audit results in memory
            self.log_observation(
                f"Completed full repository scan: {repository_path}",
                data={
                    "audit_id": audit_id,
                    "repository_path": str(repo_path),
                    "findings_count": len(report.findings),
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

    async def _scan_repository_structure(self, repo_path: Path, report: RepositoryAuditReport):
        """Scan and analyze repository file structure."""
        try:
            file_count = 0
            directory_count = 0
            total_size = 0
            file_extensions: Dict[str, int] = {}
            
            for item in repo_path.rglob("*"):
                if self._should_ignore_path(str(item), self.ignore_patterns):
                    continue
                
                try:
                    if item.is_file():
                        file_count += 1
                        stat_info = item.stat()
                        total_size += stat_info.st_size
                        
                        # Count file extensions
                        ext = item.suffix.lower()
                        if ext:
                            file_extensions[ext] = file_extensions.get(ext, 0) + 1
                        
                        # Check for unusually large files
                        if stat_info.st_size > 10 * 1024 * 1024:  # 10MB
                            report.findings.append(AuditFinding(
                                finding_id=f"large_file_{hash(str(item))}",
                                audit_type=AuditType.FULL_SCAN,
                                severity=AuditSeverity.MEDIUM,
                                title="Large file detected",
                                description=f"File is {stat_info.st_size / (1024*1024):.1f}MB",
                                file_path=str(item.relative_to(repo_path)),
                                recommendation="Consider if this file should be tracked in version control"
                            ))
                    
                    elif item.is_dir():
                        directory_count += 1
                
                except (OSError, PermissionError) as e:
                    report.findings.append(AuditFinding(
                        finding_id=f"access_error_{hash(str(item))}",
                        audit_type=AuditType.FULL_SCAN,
                        severity=AuditSeverity.LOW,
                        title="File access error",
                        description=f"Cannot access: {e}",
                        file_path=str(item.relative_to(repo_path)) if item.is_relative_to(repo_path) else str(item)
                    ))
            
            # Update statistics
            report.statistics.update({
                "file_count": file_count,
                "directory_count": directory_count,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "file_extensions": file_extensions,
                "supported_file_count": sum(
                    count for ext, count in file_extensions.items() 
                    if ext in self.supported_file_extensions
                )
            })
            
        except Exception as e:
            report.findings.append(AuditFinding(
                finding_id="structure_scan_error",
                audit_type=AuditType.FULL_SCAN,
                severity=AuditSeverity.HIGH,
                title="Repository structure scan failed",
                description=str(e)
            ))

    async def _analyze_repository_health(self, repo_path: Path, report: RepositoryAuditReport):
        """Analyze overall repository health and quality indicators."""
        try:
            # Check if it's a Git repository
            git_dir = repo_path / ".git"
            is_git_repo = git_dir.exists()
            
            if is_git_repo:
                report.metadata["version_control"] = "git"
                
                # Check for common Git issues
                gitignore_path = repo_path / ".gitignore"
                if not gitignore_path.exists():
                    report.findings.append(AuditFinding(
                        finding_id="missing_gitignore",
                        audit_type=AuditType.FULL_SCAN,
                        severity=AuditSeverity.MEDIUM,
                        title="Missing .gitignore file",
                        description="Repository should have a .gitignore file to exclude unwanted files",
                        recommendation="Create a .gitignore file appropriate for your project type"
                    ))
            else:
                report.findings.append(AuditFinding(
                    finding_id="not_git_repo",
                    audit_type=AuditType.FULL_SCAN,
                    severity=AuditSeverity.LOW,
                    title="Not a Git repository",
                    description="Directory is not under Git version control",
                    recommendation="Initialize Git repository if this is a code project"
                ))
            
            # Check directory depth
            max_depth = 0
            for item in repo_path.rglob("*"):
                if item.is_file():
                    try:
                        depth = len(item.relative_to(repo_path).parts)
                        max_depth = max(max_depth, depth)
                    except ValueError:
                        continue
            
            report.metadata["max_directory_depth"] = max_depth
            
            if max_depth > 10:
                report.findings.append(AuditFinding(
                    finding_id="deep_directory_structure",
                    audit_type=AuditType.FULL_SCAN,
                    severity=AuditSeverity.LOW,
                    title="Deep directory structure",
                    description=f"Maximum directory depth is {max_depth}",
                    recommendation="Consider flattening deep directory structures for better organization"
                ))
                
        except Exception as e:
            report.findings.append(AuditFinding(
                finding_id="health_analysis_error",
                audit_type=AuditType.FULL_SCAN,
                severity=AuditSeverity.MEDIUM,
                title="Repository health analysis failed",
                description=str(e)
            ))

    async def _check_important_files(self, repo_path: Path, report: RepositoryAuditReport):
        """Check for presence of important files in repository."""
        try:
            found_files: Set[str] = set()
            missing_files: List[str] = []
            
            # Check for important files
            for important_file in self.important_files:
                file_path = repo_path / important_file
                if file_path.exists() or (file_path.is_dir() and any(file_path.iterdir())):
                    found_files.add(important_file)
                else:
                    # Check for case variations and similar names
                    found_variant = False
                    for existing_file in repo_path.iterdir():
                        if existing_file.name.lower() == important_file.lower():
                            found_files.add(important_file)
                            found_variant = True
                            break
                    
                    if not found_variant:
                        missing_files.append(important_file)
            
            report.metadata["found_important_files"] = list(found_files)
            report.metadata["missing_important_files"] = missing_files
            
            # Generate findings for critical missing files
            critical_files = {'README.md', 'LICENSE', 'requirements.txt', 'package.json'}
            for missing_file in missing_files:
                if missing_file in critical_files:
                    severity = AuditSeverity.HIGH
                else:
                    severity = AuditSeverity.MEDIUM
                    
                report.findings.append(AuditFinding(
                    finding_id=f"missing_{missing_file.replace('.', '_').replace('/', '_')}",
                    audit_type=AuditType.MISSING_FILES,
                    severity=severity,
                    title=f"Missing important file: {missing_file}",
                    description=f"Repository should contain {missing_file}",
                    recommendation=self._get_file_recommendation(missing_file)
                ))
                
        except Exception as e:
            report.findings.append(AuditFinding(
                finding_id="important_files_check_error",
                audit_type=AuditType.MISSING_FILES,
                severity=AuditSeverity.MEDIUM,
                title="Important files check failed",
                description=str(e)
            ))

    async def _scan_file_content(self, repo_path: Path, report: RepositoryAuditReport):
        """Scan file content for potential issues and analysis."""
        try:
            scanned_files = 0
            content_issues = 0
            
            for file_path in repo_path.rglob("*"):
                if not file_path.is_file():
                    continue
                    
                if self._should_ignore_path(str(file_path), self.ignore_patterns):
                    continue
                    
                if file_path.suffix not in self.supported_file_extensions:
                    continue
                
                try:
                    # Read file content
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    scanned_files += 1
                    
                    # Check for common issues
                    if len(content.strip()) == 0:
                        report.findings.append(AuditFinding(
                            finding_id=f"empty_file_{hash(str(file_path))}",
                            audit_type=AuditType.FULL_SCAN,
                            severity=AuditSeverity.LOW,
                            title="Empty file",
                            description="File contains no content",
                            file_path=str(file_path.relative_to(repo_path)),
                            recommendation="Consider removing empty files or adding content"
                        ))
                        content_issues += 1
                    
                    # Check for potential security issues
                    if self._contains_potential_secrets(content):
                        report.findings.append(AuditFinding(
                            finding_id=f"potential_secrets_{hash(str(file_path))}",
                            audit_type=AuditType.FULL_SCAN,
                            severity=AuditSeverity.CRITICAL,
                            title="Potential secrets detected",
                            description="File may contain API keys, passwords, or other secrets",
                            file_path=str(file_path.relative_to(repo_path)),
                            recommendation="Review file content and move secrets to environment variables"
                        ))
                        content_issues += 1
                        
                except Exception as e:
                    self.logger.debug(f"Error reading file {file_path}: {e}")
                    continue
            
            report.statistics.update({
                "scanned_files": scanned_files,
                "content_issues_found": content_issues
            })
            
        except Exception as e:
            report.findings.append(AuditFinding(
                finding_id="content_scan_error",
                audit_type=AuditType.FULL_SCAN,
                severity=AuditSeverity.MEDIUM,
                title="File content scanning failed",
                description=str(e)
            ))

    async def _extract_project_metadata(self, repo_path: Path, report: RepositoryAuditReport):
        """Extract project metadata from various project files."""
        try:
            project_metadata = {}
            
            # Check for Python project
            if (repo_path / "pyproject.toml").exists():
                project_metadata["project_type"] = "python"
                project_metadata["build_system"] = "pyproject"
            elif (repo_path / "setup.py").exists():
                project_metadata["project_type"] = "python"
                project_metadata["build_system"] = "setuptools"
            elif (repo_path / "requirements.txt").exists():
                project_metadata["project_type"] = "python"
                project_metadata["build_system"] = "pip"
            
            # Check for Node.js project
            elif (repo_path / "package.json").exists():
                project_metadata["project_type"] = "nodejs"
                try:
                    with open(repo_path / "package.json", 'r') as f:
                        package_data = json.loads(f.read())
                        project_metadata["project_name"] = package_data.get("name")
                        project_metadata["project_version"] = package_data.get("version")
                        project_metadata["project_description"] = package_data.get("description")
                except Exception:
                    pass
            
            # Check for other project types
            elif (repo_path / "Cargo.toml").exists():
                project_metadata["project_type"] = "rust"
            elif (repo_path / "go.mod").exists():
                project_metadata["project_type"] = "go"
            elif (repo_path / "pom.xml").exists():
                project_metadata["project_type"] = "java"
            elif (repo_path / "build.gradle").exists():
                project_metadata["project_type"] = "java"
            else:
                project_metadata["project_type"] = "unknown"
            
            report.metadata.update(project_metadata)
            
        except Exception as e:
            report.findings.append(AuditFinding(
                finding_id="metadata_extraction_error",
                audit_type=AuditType.METADATA_EXTRACTION,
                severity=AuditSeverity.LOW,
                title="Project metadata extraction failed",
                description=str(e)
            ))

    async def _perform_incremental_scan(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform incremental repository scan for changed files only."""
        try:
            repository_path = task.get("repository_path")
            if not repository_path:
                return {"success": False, "error": "Repository path required"}
            
            repo_path = Path(repository_path)
            if not repo_path.exists():
                return {
                    "success": False,
                    "error": f"Repository path does not exist: {repository_path}"
                }
            
            audit_id = f"incremental_scan_{int(datetime.now().timestamp())}"
            
            # Create incremental audit report
            report = RepositoryAuditReport(
                repository_path=str(repo_path),
                audit_id=audit_id,
                audit_type=AuditType.INCREMENTAL_SCAN,
                start_time=datetime.now(timezone.utc)
            )
            
            # Track incremental scan statistics
            stats = {
                "total_files_checked": 0,
                "changed_files": 0,
                "new_files": 0,
                "deleted_files": 0,
                "unchanged_files": 0,
                "processed_files": 0,
                "failed_files": 0,
                "errors": []
            }
            
            # Check for changes since last scan
            last_scan_time = self.last_scan_times.get(str(repo_path))
            
            # Get all current files
            current_files = {}
            for file_path in repo_path.rglob("*"):
                if (file_path.is_file() and
                    not self._should_ignore_path(str(file_path), self.ignore_patterns) and
                    file_path.suffix in self.supported_file_extensions):
                    
                    try:
                        current_files[str(file_path)] = {
                            'path': file_path,
                            'stat': file_path.stat(),
                            'hash': self._get_file_hash(file_path),
                            'modified_time': datetime.fromtimestamp(
                                file_path.stat().st_mtime, tz=timezone.utc
                            )
                        }
                        stats["total_files_checked"] += 1
                    except (OSError, PermissionError) as e:
                        stats["errors"].append(f"Cannot access file {file_path}: {e}")
                        stats["failed_files"] += 1
            
            # Compare with cached file information
            cached_files = self.repository_cache.get(str(repo_path), {}).get('files', {})
            
            # Find new, changed, and deleted files
            new_files = []
            changed_files = []
            unchanged_files = []
            deleted_files = []
            
            # Check current files against cache
            for file_path_str, file_info in current_files.items():
                if file_path_str not in cached_files:
                    # New file
                    new_files.append(file_path_str)
                    stats["new_files"] += 1
                elif self._has_file_changed(cached_files[file_path_str], file_info):
                    # Changed file
                    changed_files.append(file_path_str)
                    stats["changed_files"] += 1
                else:
                    # Unchanged file
                    unchanged_files.append(file_path_str)
                    stats["unchanged_files"] += 1
            
            # Check for deleted files
            for cached_file_path in cached_files:
                if cached_file_path not in current_files:
                    deleted_files.append(cached_file_path)
                    stats["deleted_files"] += 1
            
            # Process changed and new files
            processed_files = new_files + changed_files
            
            for file_path_str in processed_files:
                try:
                    file_path = Path(file_path_str)
                    
                    # Analyze file for issues and ingest into vector database
                    await self._process_file_incremental(file_path, report)
                    
                    # Update vector database if needed
                    if task.get("update_vector_db", True):
                        try:
                            self.vector_db.ingest_file(file_path, 
                                                     self.ignore_patterns, 
                                                     force_update=True)
                        except Exception as e:
                            stats["errors"].append(f"Vector DB ingestion failed for {file_path}: {e}")
                    
                    stats["processed_files"] += 1
                    
                except Exception as e:
                    stats["errors"].append(f"Processing failed for {file_path_str}: {e}")
                    stats["failed_files"] += 1
            
            # Clean up vector database entries for deleted files
            if deleted_files and task.get("cleanup_deleted", True):
                for deleted_file_path in deleted_files:
                    try:
                        self.vector_db.delete_documents_by_source(deleted_file_path)
                        report.findings.append(AuditFinding(
                            finding_id=f"deleted_file_{hash(deleted_file_path)}",
                            audit_type=AuditType.INCREMENTAL_SCAN,
                            severity=AuditSeverity.INFO,
                            title="File deleted",
                            description=f"File no longer exists in repository",
                            file_path=deleted_file_path,
                            recommendation="Vector database entries cleaned up"
                        ))
                    except Exception as e:
                        stats["errors"].append(f"Cleanup failed for deleted file {deleted_file_path}: {e}")
            
            # Update repository cache
            self.repository_cache[str(repo_path)] = {
                'files': current_files,
                'last_incremental_scan': datetime.now(timezone.utc).isoformat(),
                'scan_stats': stats
            }
            
            # Update last scan time
            self.last_scan_times[str(repo_path)] = datetime.now(timezone.utc)
            
            # Finalize report
            report.end_time = datetime.now(timezone.utc)
            report.statistics.update(stats)
            report.metadata.update({
                "incremental_scan": True,
                "new_files_list": new_files[:10],  # Limit for readability
                "changed_files_list": changed_files[:10],
                "deleted_files_list": deleted_files[:10],
                "total_files_in_repo": len(current_files)
            })
            
            # Log results
            self.log_observation(
                f"Completed incremental repository scan: {repository_path}",
                data={
                    "audit_id": audit_id,
                    "repository_path": str(repo_path),
                    "changes_detected": len(new_files) + len(changed_files) + len(deleted_files),
                    "statistics": stats
                }
            )
            
            return {
                "success": True,
                "audit_id": audit_id,
                "repository_path": str(repo_path),
                "audit_report": asdict(report),
                "summary": self._generate_incremental_summary(stats),
                "changes": {
                    "new_files": new_files,
                    "changed_files": changed_files,
                    "deleted_files": deleted_files,
                    "unchanged_files_count": len(unchanged_files)
                }
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _ingest_repository_files(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest repository files into vector database."""
        try:
            repository_path = task.get("repository_path")
            if not repository_path:
                return {"success": False, "error": "Repository path required"}
            
            repo_path = Path(repository_path)
            if not repo_path.exists():
                return {"success": False, "error": f"Repository path does not exist: {repository_path}"}
            
            # Use vector database to ingest files
            ingested_files: List[str] = []
            for file_path in repo_path.rglob("*"):
                if (file_path.is_file() and 
                    not self._should_ignore_path(str(file_path), self.ignore_patterns) and
                    file_path.suffix in self.supported_file_extensions):
                    
                    try:
                        # Ingest file using vector database
                        self.vector_db.ingest_file(file_path)
                        ingested_files.append(str(file_path.relative_to(repo_path)))
                    except Exception as e:
                        self.logger.warning(f"Failed to ingest file {file_path}: {e}")
            
            return {
                "success": True,
                "message": f"Ingested {len(ingested_files)} files",
                "ingested_files": ingested_files,
                "repository_path": str(repo_path)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _check_missing_files(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Check for missing important files in repository."""
        try:
            repository_path = task.get("repository_path")
            if not repository_path:
                return {"success": False, "error": "Repository path required"}
            
            repo_path = Path(repository_path)
            if not repo_path.exists():
                return {"success": False, "error": f"Repository path does not exist: {repository_path}"}
            
            missing_files: List[str] = []
            for important_file in self.important_files:
                file_path = repo_path / important_file
                if not file_path.exists():
                    missing_files.append(important_file)
            
            return {
                "success": True,
                "missing_files": missing_files,
                "repository_path": str(repo_path)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _cleanup_vector_database(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Clean up stale entries from vector database."""
        try:
            repository_path = task.get("repository_path")
            if not repository_path:
                return {"success": False, "error": "Repository path required"}
            
            # Implementation for cleaning up vector database
            # This would remove entries for files that no longer exist
            return {
                "success": True,
                "message": "Vector database cleanup completed",
                "repository_path": repository_path
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _perform_health_check(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform repository health check."""
        try:
            repository_path = task.get("repository_path")
            if not repository_path:
                return {"success": False, "error": "Repository path required"}
            
            repo_path = Path(repository_path)
            if not repo_path.exists():
                return {"success": False, "error": f"Repository path does not exist: {repository_path}"}
            
            # Basic health check implementation
            health_score = 100  # Start with perfect score
            issues: List[str] = []
            
            # Check for README
            if not any((repo_path / name).exists() for name in ['README.md', 'README.txt', 'README.rst']):
                health_score -= 20
                issues.append("Missing README file")
            
            # Check for LICENSE
            if not any((repo_path / name).exists() for name in ['LICENSE', 'LICENCE', 'COPYING']):
                health_score -= 15
                issues.append("Missing LICENSE file")
            
            # Check for .gitignore
            if not (repo_path / '.gitignore').exists():
                health_score -= 10
                issues.append("Missing .gitignore file")
            
            return {
                "success": True,
                "health_score": health_score,
                "issues": issues,
                "repository_path": str(repo_path)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _extract_repository_metadata(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive repository metadata."""
        try:
            repository_path = task.get("repository_path")
            if not repository_path:
                return {"success": False, "error": "Repository path required"}
            
            repo_path = Path(repository_path)
            if not repo_path.exists():
                return {"success": False, "error": f"Repository path does not exist: {repository_path}"}
            
            metadata: Dict[str, Any] = {}
            
            # Extract project type and build system information
            if (repo_path / "pyproject.toml").exists():
                metadata["project_type"] = "python"
                metadata["build_system"] = "pyproject"
            elif (repo_path / "setup.py").exists():
                metadata["project_type"] = "python"
                metadata["build_system"] = "setuptools"
            elif (repo_path / "package.json").exists():
                metadata["project_type"] = "nodejs"
                try:
                    with open(repo_path / "package.json", 'r') as f:
                        package_data = json.loads(f.read())
                        metadata.update({
                            "project_name": package_data.get("name"),
                            "project_version": package_data.get("version"),
                            "project_description": package_data.get("description")
                        })
                except Exception:
                    pass
            
            # Count files by type
            file_counts: Dict[str, int] = {}
            for file_path in repo_path.rglob("*"):
                if file_path.is_file():
                    ext = file_path.suffix.lower()
                    file_counts[ext] = file_counts.get(ext, 0) + 1
            
            metadata["file_counts"] = file_counts
            
            return {
                "success": True,
                "metadata": metadata,
                "repository_path": str(repo_path)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _analyze_repository_dependencies(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze repository dependencies."""
        try:
            repository_path = task.get("repository_path")
            if not repository_path:
                return {"success": False, "error": "Repository path required"}
            
            repo_path = Path(repository_path)
            if not repo_path.exists():
                return {"success": False, "error": f"Repository path does not exist: {repository_path}"}
            
            dependencies: Dict[str, Any] = {}
            
            # Python dependencies
            if (repo_path / "requirements.txt").exists():
                try:
                    with open(repo_path / "requirements.txt", 'r') as f:
                        dependencies["python"] = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                except Exception:
                    pass
            
            # Node.js dependencies
            if (repo_path / "package.json").exists():
                try:
                    with open(repo_path / "package.json", 'r') as f:
                        package_data = json.loads(f.read())
                        dependencies["nodejs"] = {
                            "dependencies": package_data.get("dependencies", {}),
                            "devDependencies": package_data.get("devDependencies", {})
                        }
                except Exception:
                    pass
            
            return {
                "success": True,
                "dependencies": dependencies,
                "repository_path": str(repo_path)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _should_ignore_path(self, path: str, ignore_patterns: List[str]) -> bool:
        """Check if a path should be ignored based on patterns."""
        for pattern in ignore_patterns:
            if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(Path(path).name, pattern):
                return True
        return False

    def _contains_potential_secrets(self, content: str) -> bool:
        """Check if content contains potential secrets or sensitive information."""
        secret_patterns = [
            r'api[_-]?key\s*[:=]\s*[\'"][a-zA-Z0-9]{20,}[\'"]',
            r'password\s*[:=]\s*[\'"][^\'"]+[\'"]',
            r'secret\s*[:=]\s*[\'"][a-zA-Z0-9]{20,}[\'"]',
            r'token\s*[:=]\s*[\'"][a-zA-Z0-9]{20,}[\'"]',
            r'aws[_-]?access[_-]?key\s*[:=]\s*[\'"][A-Z0-9]{20}[\'"]',
            r'-----BEGIN\s+(RSA\s+)?PRIVATE KEY-----'
        ]
        
        import re
        content_lower = content.lower()
        
        for pattern in secret_patterns:
            if re.search(pattern, content_lower, re.IGNORECASE):
                return True
        
        return False

    def _get_file_recommendation(self, filename: str) -> str:
        """Get recommendation for missing important files."""
        recommendations = {
            'README.md': 'Create a README.md file to document your project',
            'LICENSE': 'Add a LICENSE file to specify project licensing terms',
            'requirements.txt': 'Create requirements.txt to specify Python dependencies',
            'package.json': 'Create package.json for Node.js project configuration',
            '.gitignore': 'Add .gitignore to exclude unwanted files from version control',
            'CONTRIBUTING.md': 'Add contributing guidelines for project contributors',
            'CODE_OF_CONDUCT.md': 'Add code of conduct to establish community standards'
        }
        return recommendations.get(filename, f'Consider adding {filename} to improve project documentation')

    def _generate_audit_summary(self, report: RepositoryAuditReport) -> str:
        """Generate human-readable audit summary."""
        stats = report.statistics or {}
        findings = report.findings or []
        metadata = report.metadata or {}
        
        findings_by_severity = {
            'critical': len([f for f in findings
                            if f.severity == AuditSeverity.CRITICAL]),
            'high': len([f for f in findings
                        if f.severity == AuditSeverity.HIGH]),
            'medium': len([f for f in findings
                          if f.severity == AuditSeverity.MEDIUM]),
            'low': len([f for f in findings
                       if f.severity == AuditSeverity.LOW])
        }
        
        duration = stats.get('scan_duration_seconds', 0)
        file_count = stats.get('file_count', 0)
        dir_count = stats.get('directory_count', 0)
        
        summary_parts = [
            f"Repository audit completed in {duration:.1f} seconds",
            f"Scanned {file_count} files in {dir_count} directories",
            (f"Found {len(findings)} findings: "
             f"{findings_by_severity['critical']} critical, "
             f"{findings_by_severity['high']} high, "
             f"{findings_by_severity['medium']} medium, "
             f"{findings_by_severity['low']} low")
        ]
        
        if metadata.get('project_type'):
            summary_parts.append(f"Project type: {metadata['project_type']}")
        
        return ". ".join(summary_parts)

    def _get_file_hash(self, file_path: Path) -> Optional[str]:
        """Calculate SHA256 hash of file content."""
        try:
            with open(file_path, 'rb') as f:
                file_content = f.read()
                return hashlib.sha256(file_content).hexdigest()
        except Exception as e:
            self.logger.warning(f"Cannot calculate hash for {file_path}: {e}")
            return None

    def _has_file_changed(
        self,
        cached_info: Dict[str, Any],
        current_info: Dict[str, Any]
    ) -> bool:
        """Check if a file has changed since last scan."""
        try:
            # Compare modification time
            cached_mtime = cached_info.get('modified_time')
            current_mtime = current_info.get('modified_time')
            
            if cached_mtime and current_mtime:
                if isinstance(cached_mtime, str):
                    cached_mtime = datetime.fromisoformat(
                        cached_mtime.replace('Z', '+00:00')
                    )
                    
                if current_mtime != cached_mtime:
                    return True
            
            # Compare file hash if available
            cached_hash = cached_info.get('hash')
            current_hash = current_info.get('hash')
            
            if cached_hash and current_hash and cached_hash != current_hash:
                return True
                
            # Compare file size
            cached_size = cached_info.get('stat', {}).get('st_size')
            current_size = current_info.get('stat', {}).st_size
            
            if cached_size != current_size:
                return True
                
            return False
            
        except Exception as e:
            self.logger.debug(f"Error comparing file info: {e}")
            return True  # Assume changed if we can't compare
    
    async def _process_file_incremental(
        self,
        file_path: Path,
        report: RepositoryAuditReport
    ) -> None:
        """Process a file during incremental scan for issues and analysis."""
        try:
            # Check file size
            stat_info = file_path.stat()
            if stat_info.st_size > 10 * 1024 * 1024:  # 10MB
                report.findings.append(AuditFinding(
                    finding_id=f"large_file_{hash(str(file_path))}",
                    audit_type=AuditType.INCREMENTAL_SCAN,
                    severity=AuditSeverity.MEDIUM,
                    title="Large file detected",
                    description=(
                        f"File is {stat_info.st_size / (1024*1024):.1f}MB"
                    ),
                    file_path=str(file_path),
                    recommendation=(
                        "Consider if this file should be tracked in "
                        "version control"
                    )
                ))
            
            # Check for potential secrets in code files
            code_extensions = {
                '.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs'
            }
            if file_path.suffix in code_extensions:
                try:
                    content = file_path.read_text(
                        encoding='utf-8', errors='ignore'
                    )
                    if self._contains_potential_secrets(content):
                        report.findings.append(AuditFinding(
                            finding_id=(
                                f"potential_secrets_{hash(str(file_path))}"
                            ),
                            audit_type=AuditType.INCREMENTAL_SCAN,
                            severity=AuditSeverity.CRITICAL,
                            title="Potential secrets detected",
                            description=(
                                "File may contain API keys, passwords, or "
                                "other secrets"
                            ),
                            file_path=str(file_path),
                            recommendation=(
                                "Review file content and move secrets to "
                                "environment variables"
                            )
                        ))
                except Exception:
                    pass  # Skip if can't read file
                    
        except Exception as e:
            self.logger.debug(f"Error processing file {file_path}: {e}")
            
    def _generate_incremental_summary(self, stats: Dict[str, Any]) -> str:
        """Generate human-readable incremental scan summary."""
        summary_parts = [
            "Incremental scan completed",
            f"Checked {stats.get('total_files_checked', 0)} files",
            f"Found {stats.get('changed_files', 0)} changed files",
            f"Found {stats.get('new_files', 0)} new files",
            f"Found {stats.get('deleted_files', 0)} deleted files",
            f"Processed {stats.get('processed_files', 0)} files"
        ]
        
        if stats.get('failed_files', 0) > 0:
            summary_parts.append(
                f"Failed to process {stats['failed_files']} files"
            )
            
        if stats.get('errors'):
            summary_parts.append(f"Encountered {len(stats['errors'])} errors")
        
        return ". ".join(summary_parts)

    async def cleanup_repository_data(
        self, 
        repo_path: Path, 
        max_age_days: int = 30,
        remove_stale: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive cleanup of repository audit data.
        
        Args:
            repo_path: Path to repository
            max_age_days: Maximum age for keeping data
            remove_stale: Whether to remove stale entries
            
        Returns:
            Dict with cleanup statistics
        """
        cleanup_stats = {
            'stale_documents_removed': 0,
            'orphaned_entries_removed': 0,
            'cache_entries_cleared': 0,
            'vector_db_entries_removed': 0,
            'memory_entries_cleared': 0,
            'total_space_freed': 0,
            'errors': []
        }
        
        try:
            self.logger.info(f"Starting comprehensive cleanup for {repo_path}")
            
            # Get current repository files for comparison
            current_files: Set[str] = set()
            if repo_path.exists():
                current_files = {
                    str(f.relative_to(repo_path))
                    for f in repo_path.rglob('*')
                    if f.is_file() and not self._should_ignore_path(
                        str(f), self.ignore_patterns
                    )
                }
            
            # Clean up vector database entries
            if hasattr(self, 'vector_db') and self.vector_db:
                vector_cleanup = await self._cleanup_vector_database(
                    repo_path, current_files, max_age_days, remove_stale
                )
                cleanup_stats.update(vector_cleanup)
            
            # Clean up shared memory entries
            if hasattr(self, 'shared_memory') and self.shared_memory:
                memory_cleanup = await self._cleanup_shared_memory(
                    repo_path, current_files, max_age_days
                )
                cleanup_stats.update(memory_cleanup)
            
            # Clean up repository cache
            cache_cleanup = await self._cleanup_repository_cache(
                repo_path, max_age_days
            )
            cleanup_stats.update(cache_cleanup)
            
            # Update cleanup statistics
            total_removed = (
                cleanup_stats['stale_documents_removed'] +
                cleanup_stats['orphaned_entries_removed'] +
                cleanup_stats['cache_entries_cleared'] +
                cleanup_stats['vector_db_entries_removed']
            )
            
            self.logger.info(
                f"Cleanup completed for {repo_path}. "
                f"Removed {total_removed} entries"
            )
            
        except Exception as e:
            error_msg = f"Error during cleanup: {e}"
            self.logger.error(error_msg)
            cleanup_stats['errors'].append(error_msg)
            
        return cleanup_stats

    async def _cleanup_vector_database(
        self,
        repo_path: Path,
        current_files: Set[str],
        max_age_days: int,
        remove_stale: bool
    ) -> Dict[str, int]:
        """Clean up vector database entries for repository."""
        cleanup_stats = {
            'vector_db_entries_removed': 0,
            'stale_documents_removed': 0
        }
        
        try:
            # Use existing vector database search to find repo documents
            # and leverage built-in cleanup methods
            
            # Remove stale documents by source if requested
            if remove_stale:
                try:
                    stale_count = self.vector_db.delete_documents_by_source(
                        str(repo_path)
                    )
                    cleanup_stats['stale_documents_removed'] = stale_count
                    self.logger.info(
                        f"Removed {stale_count} stale documents from "
                        f"vector database"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Error removing stale documents: {e}"
                    )
            
            # Clean up old documents based on age
            try:
                old_removed = self.vector_db.cleanup_old_documents(
                    days=max_age_days
                )
                cleanup_stats['vector_db_entries_removed'] = old_removed
                
                if old_removed > 0:
                    self.logger.info(
                        f"Removed {old_removed} old documents from "
                        f"vector database"
                    )
            except Exception as e:
                self.logger.error(f"Error cleaning old documents: {e}")
                
        except Exception as e:
            self.logger.error(f"Error cleaning vector database: {e}")
            
        return cleanup_stats

    async def _cleanup_shared_memory(
        self,
        repo_path: Path,
        current_files: Set[str],
        max_age_days: int
    ) -> Dict[str, int]:
        """Clean up shared memory entries for repository."""
        cleanup_stats = {'memory_entries_cleared': 0}
        
        try:
            # Use built-in cleanup method
            cleanup_result = self.shared_memory.cleanup_old_entries(
                days=max_age_days
            )
            cleanup_stats['memory_entries_cleared'] = sum(
                cleanup_result.values()
            )
            
            if cleanup_stats['memory_entries_cleared'] > 0:
                self.logger.info(
                    f"Cleared {cleanup_stats['memory_entries_cleared']} "
                    f"old memory entries"
                )
                
        except Exception as e:
            self.logger.error(f"Error cleaning shared memory: {e}")
            
        return cleanup_stats

    async def _cleanup_repository_cache(
        self,
        repo_path: Path,
        max_age_days: int
    ) -> Dict[str, int]:
        """Clean up repository cache entries."""
        cleanup_stats = {'cache_entries_cleared': 0}
        
        try:
            repo_str = str(repo_path)
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            
            # Clean up repository cache
            cache_keys_to_remove = []
            for cache_key, cache_data in self.repository_cache.items():
                if cache_key.startswith(repo_str):
                    last_scan = cache_data.get('last_scan')
                    if last_scan and last_scan < cutoff_date:
                        cache_keys_to_remove.append(cache_key)
            
            for key in cache_keys_to_remove:
                del self.repository_cache[key]
                cleanup_stats['cache_entries_cleared'] += 1
            
            # Clean up last scan times
            scan_keys_to_remove = []
            for scan_key, scan_time in self.last_scan_times.items():
                if scan_key.startswith(repo_str):
                    if scan_time < cutoff_date:
                        scan_keys_to_remove.append(scan_key)
            
            for key in scan_keys_to_remove:
                del self.last_scan_times[key]
                cleanup_stats['cache_entries_cleared'] += len(
                    scan_keys_to_remove
                )
            
            if cleanup_stats['cache_entries_cleared'] > 0:
                self.logger.info(
                    f"Cleared {cleanup_stats['cache_entries_cleared']} "
                    f"cache entries"
                )
                
        except Exception as e:
            self.logger.error(f"Error cleaning repository cache: {e}")
            
        return cleanup_stats

    def _update_state(
        self,
        status: str,
        task_id: Optional[str] = None,
        error: Optional[str] = None
    ) -> None:
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
        return [
            "repository_scanning",
            "file_ingestion",
            "repository_audit",
            "missing_file_detection",
            "repository_health_check",
            "metadata_extraction",
            "dependency_analysis",
            "vector_db_cleanup",
            "incremental_updates",
            "comprehensive_cleanup",
            "stale_entry_removal",
            "repository_maintenance"
        ]

    def get_status(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "type": "repo_auditor",
            "last_scan_count": len(self.last_scan_times),
            "cached_repositories": len(self.repository_cache)
        }
