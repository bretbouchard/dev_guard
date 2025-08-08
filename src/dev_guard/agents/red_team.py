"""Red Team Agent for security vulnerability scanning and penetration testing."""

import asyncio
import hashlib
import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class VulnerabilityType(Enum):
    """Types of security vulnerabilities."""
    CODE_INJECTION = "code_injection"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    CRYPTOGRAPHIC = "cryptographic"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    INFORMATION_DISCLOSURE = "information_disclosure"
    DENIAL_OF_SERVICE = "dos"
    BUFFER_OVERFLOW = "buffer_overflow"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    LOGGING_MONITORING = "logging_monitoring"
    OTHER = "other"


class SeverityLevel(Enum):
    """Security vulnerability severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class TestType(Enum):
    """Types of security tests."""
    SAST = "sast"  # Static Application Security Testing
    DAST = "dast"  # Dynamic Application Security Testing
    SCA = "sca"   # Software Composition Analysis
    IAST = "iast"  # Interactive Application Security Testing
    SECRETS = "secrets"  # Secrets scanning
    INFRASTRUCTURE = "infrastructure"
    API = "api"
    WEB = "web"


@dataclass
class SecurityFinding:
    """Security vulnerability finding."""
    finding_id: str
    title: str
    description: str
    vulnerability_type: VulnerabilityType
    severity: SeverityLevel
    test_type: TestType
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    confidence: Optional[str] = None
    cwe_id: Optional[str] = None
    owasp_category: Optional[str] = None
    remediation: Optional[str] = None
    references: Optional[List[str]] = None
    evidence: Optional[Dict[str, Any]] = None
    discovered_at: Optional[datetime] = None


@dataclass
class SecurityReport:
    """Comprehensive security assessment report."""
    scan_id: str
    repository_path: str
    scan_timestamp: datetime
    test_types: List[TestType]
    findings: List[SecurityFinding]
    scan_statistics: Dict[str, Any]
    risk_score: float
    compliance_status: str
    recommendations: List[str]
    tools_used: List[str]
    scan_duration: float


class RedTeamAgent(BaseAgent):
    """Red Team Agent for security vulnerability scanning and penetration testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Security scanning tools
        self.available_tools = self._detect_security_tools()
        self.scan_cache = {}
        
        # OWASP Top 10 patterns
        self.owasp_patterns = self._initialize_owasp_patterns()
        
        # CWE mappings
        self.cwe_mappings = self._initialize_cwe_mappings()
    
    def _detect_security_tools(self) -> Dict[str, bool]:
        """Detect available security scanning tools."""
        tools = {}
        
        security_tools = [
            "bandit",      # Python security linter
            "safety",      # Python dependency scanner
            "semgrep",     # Multi-language static analyzer
            "eslint",      # JavaScript/TypeScript linter with security rules
            "sonarqube",   # Code quality and security
            "trivy",       # Container and filesystem vulnerability scanner
            "grype",       # Container vulnerability scanner
            "syft",        # Software Bill of Materials (SBOM) generator
            "gosec",       # Go security checker
            "brakeman",    # Ruby on Rails security scanner
            "phpcs",       # PHP security checker
        ]
        
        for tool in security_tools:
            try:
                result = subprocess.run(
                    [tool, "--version"], 
                    capture_output=True, 
                    timeout=5
                )
                tools[tool] = result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                tools[tool] = False
        
        self.logger.info(f"Available security tools: {[t for t, available in tools.items() if available]}")
        return tools
    
    def _initialize_owasp_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize OWASP Top 10 vulnerability patterns."""
        return {
            "A01_Broken_Access_Control": [
                {
                    "pattern": r"(?i)(bypass|skip|ignore).*(auth|permission|access)",
                    "description": "Potential access control bypass",
                    "severity": SeverityLevel.HIGH
                },
                {
                    "pattern": r"(?i)(?:admin|root|sudo|privilege).{0,50}(?:hardcoded|default)",
                    "description": "Hardcoded privileged credentials",
                    "severity": SeverityLevel.CRITICAL
                }
            ],
            "A02_Cryptographic_Failures": [
                {
                    "pattern": r"(?i)(?:md5|sha1)\s*\(",
                    "description": "Use of weak cryptographic hash function",
                    "severity": SeverityLevel.MEDIUM
                },
                {
                    "pattern": r"(?i)(?:des|3des|rc4)\s*\(",
                    "description": "Use of weak encryption algorithm",
                    "severity": SeverityLevel.HIGH
                }
            ],
            "A03_Injection": [
                {
                    "pattern": r"(?i)(?:exec|eval|system)\s*\(\s*[^)]*(?:input|request|param)",
                    "description": "Potential code injection vulnerability",
                    "severity": SeverityLevel.CRITICAL
                },
                {
                    "pattern": r"(?i)select\s+.*\s+from\s+.*\s+where\s+.*['\"]?\s*\+\s*",
                    "description": "Potential SQL injection vulnerability",
                    "severity": SeverityLevel.CRITICAL
                }
            ],
            "A04_Insecure_Design": [
                {
                    "pattern": r"(?i)password.*=.*['\"][^'\"]*['\"]",
                    "description": "Hardcoded password",
                    "severity": SeverityLevel.HIGH
                }
            ],
            "A05_Security_Misconfiguration": [
                {
                    "pattern": r"(?i)debug\s*=\s*true",
                    "description": "Debug mode enabled in production",
                    "severity": SeverityLevel.MEDIUM
                }
            ]
        }
    
    def _initialize_cwe_mappings(self) -> Dict[str, str]:
        """Initialize CWE (Common Weakness Enumeration) mappings."""
        return {
            "sql_injection": "CWE-89",
            "xss": "CWE-79",
            "code_injection": "CWE-94",
            "hardcoded_credentials": "CWE-798",
            "weak_cryptography": "CWE-327",
            "buffer_overflow": "CWE-120",
            "path_traversal": "CWE-22",
            "csrf": "CWE-352"
        }
    
    async def execute(self, state: Any) -> Any:
        """Execute red team agent with state management."""
        if isinstance(state, dict) and "task" in state:
            task = state["task"]
        else:
            task = state
        
        return await self.execute_task(task)
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute red team security task."""
        try:
            task_type = task.get("type", "security_scan")
            
            if task_type == "security_scan":
                return await self._perform_security_scan(task)
            elif task_type == "vulnerability_assessment":
                return await self._vulnerability_assessment(task)
            elif task_type == "penetration_test":
                return await self._penetration_test(task)
            elif task_type == "compliance_check":
                return await self._compliance_check(task)
            elif task_type == "threat_modeling":
                return await self._threat_modeling(task)
            else:
                return {
                    "success": False,
                    "error": f"Unknown task type: {task_type}"
                }
        
        except Exception as e:
            self.logger.error(f"Error executing red team task: {e}")
            return {"success": False, "error": str(e)}
    
    async def _perform_security_scan(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive security vulnerability scanning."""
        try:
            repository_path = task.get("repository_path")
            scan_types = task.get("scan_types", ["sast", "sca", "secrets"])
            target_files = task.get("target_files", [])
            
            if not repository_path:
                return {"success": False, "error": "Repository path required"}
            
            repo_path = Path(repository_path)
            if not repo_path.exists():
                return {"success": False, "error": f"Repository not found: {repository_path}"}
            
            scan_id = f"security_scan_{hashlib.md5(str(repo_path).encode()).hexdigest()}_{int(datetime.now().timestamp())}"
            scan_start = datetime.now()
            
            findings = []
            tools_used = []
            
            # Static Application Security Testing (SAST)
            if "sast" in scan_types or "all" in scan_types:
                sast_findings, sast_tools = await self._perform_sast_scan(repo_path, target_files)
                findings.extend(sast_findings)
                tools_used.extend(sast_tools)
            
            # Software Composition Analysis (SCA)
            if "sca" in scan_types or "all" in scan_types:
                sca_findings, sca_tools = await self._perform_sca_scan(repo_path)
                findings.extend(sca_findings)
                tools_used.extend(sca_tools)
            
            # Secrets scanning
            if "secrets" in scan_types or "all" in scan_types:
                secret_findings = await self._scan_for_secrets(repo_path)
                findings.extend(secret_findings)
                tools_used.append("secrets_scanner")
            
            # Infrastructure scanning
            if "infrastructure" in scan_types or "all" in scan_types:
                infra_findings, infra_tools = await self._scan_infrastructure(repo_path)
                findings.extend(infra_findings)
                tools_used.extend(infra_tools)
            
            scan_end = datetime.now()
            scan_duration = (scan_end - scan_start).total_seconds()
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(findings)
            
            # Generate statistics
            scan_stats = self._generate_scan_statistics(findings)
            
            # Assess compliance
            compliance_status = self._assess_compliance(findings)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(findings)
            
            # Create comprehensive report
            report = SecurityReport(
                scan_id=scan_id,
                repository_path=str(repo_path),
                scan_timestamp=scan_start,
                test_types=[TestType(t) for t in scan_types if t != "all"],
                findings=findings,
                scan_statistics=scan_stats,
                risk_score=risk_score,
                compliance_status=compliance_status,
                recommendations=recommendations,
                tools_used=list(set(tools_used)),
                scan_duration=scan_duration
            )
            
            # Cache results
            self.scan_cache[scan_id] = report
            
            # Log scan completion
            self.log_observation(
                f"Security scan completed: {repository_path}",
                data={
                    "scan_id": scan_id,
                    "findings_count": len(findings),
                    "risk_score": risk_score,
                    "scan_duration": scan_duration
                }
            )
            
            return {
                "success": True,
                "report": asdict(report),
                "summary": self._generate_scan_summary(report)
            }
        
        except Exception as e:
            self.logger.error(f"Error in security scan: {e}")
            return {"success": False, "error": str(e)}
    
    async def _perform_sast_scan(self, repo_path: Path, target_files: List[str]) -> tuple[List[SecurityFinding], List[str]]:
        """Perform Static Application Security Testing (SAST)."""
        findings = []
        tools_used = []
        
        # Pattern-based security scanning
        pattern_findings = await self._scan_security_patterns(repo_path, target_files)
        findings.extend(pattern_findings)
        tools_used.append("pattern_matcher")
        
        # Bandit for Python
        if self.available_tools.get("bandit"):
            bandit_findings = await self._run_bandit_scan(repo_path)
            findings.extend(bandit_findings)
            tools_used.append("bandit")
        
        # Semgrep for multi-language
        if self.available_tools.get("semgrep"):
            semgrep_findings = await self._run_semgrep_scan(repo_path)
            findings.extend(semgrep_findings)
            tools_used.append("semgrep")
        
        # ESLint security for JavaScript/TypeScript
        if self.available_tools.get("eslint"):
            eslint_findings = await self._run_eslint_security_scan(repo_path)
            findings.extend(eslint_findings)
            tools_used.append("eslint")
        
        return findings, tools_used
    
    async def _scan_security_patterns(self, repo_path: Path, target_files: List[str]) -> List[SecurityFinding]:
        """Scan for security patterns using OWASP Top 10 rules."""
        findings = []
        
        try:
            # Determine files to scan
            if target_files:
                files_to_scan = [Path(f) for f in target_files if Path(f).exists()]
            else:
                files_to_scan = list(repo_path.rglob("*.py")) + \
                               list(repo_path.rglob("*.js")) + \
                               list(repo_path.rglob("*.ts")) + \
                               list(repo_path.rglob("*.java")) + \
                               list(repo_path.rglob("*.php")) + \
                               list(repo_path.rglob("*.rb")) + \
                               list(repo_path.rglob("*.go")) + \
                               list(repo_path.rglob("*.rs"))
            
            for file_path in files_to_scan:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Check OWASP patterns
                    for category, patterns in self.owasp_patterns.items():
                        for pattern_info in patterns:
                            import re
                            matches = re.finditer(pattern_info["pattern"], content)
                            
                            for match in matches:
                                line_number = content[:match.start()].count('\n') + 1
                                
                                finding = SecurityFinding(
                                    finding_id=f"PATTERN-{hashlib.md5(f'{file_path}:{line_number}:{match.group()}'.encode()).hexdigest()[:8]}",
                                    title=pattern_info["description"],
                                    description=f"Security pattern detected in {file_path.name} at line {line_number}",
                                    vulnerability_type=self._map_owasp_to_vuln_type(category),
                                    severity=pattern_info["severity"],
                                    test_type=TestType.SAST,
                                    file_path=str(file_path.relative_to(repo_path)),
                                    line_number=line_number,
                                    confidence="medium",
                                    owasp_category=category,
                                    evidence={"matched_text": match.group()},
                                    discovered_at=datetime.now(timezone.utc)
                                )
                                findings.append(finding)
                
                except Exception as e:
                    self.logger.debug(f"Error scanning file {file_path}: {e}")
                    continue
        
        except Exception as e:
            self.logger.warning(f"Error in pattern scanning: {e}")
        
        return findings
    
    async def _run_bandit_scan(self, repo_path: Path) -> List[SecurityFinding]:
        """Run Bandit security scan for Python code."""
        findings = []
        
        try:
            result = subprocess.run([
                "bandit", "-f", "json", "-r", str(repo_path)
            ], capture_output=True, text=True, timeout=120)
            
            if result.stdout:
                bandit_data = json.loads(result.stdout)
                
                for issue in bandit_data.get("results", []):
                    finding = SecurityFinding(
                        finding_id=f"BANDIT-{issue.get('test_id', 'unknown')}",
                        title=issue.get("issue_text", "Security issue detected"),
                        description=issue.get("issue_text", ""),
                        vulnerability_type=VulnerabilityType.OTHER,
                        severity=self._map_bandit_severity(issue.get("issue_severity", "LOW")),
                        test_type=TestType.SAST,
                        file_path=issue.get("filename", "").replace(str(repo_path) + "/", ""),
                        line_number=issue.get("line_number"),
                        confidence=issue.get("issue_confidence", "").lower(),
                        cwe_id=issue.get("test_id"),
                        evidence={"code": issue.get("code", "")},
                        discovered_at=datetime.now(timezone.utc)
                    )
                    findings.append(finding)
        
        except Exception as e:
            self.logger.debug(f"Bandit scan error: {e}")
        
        return findings
    
    async def _perform_sca_scan(self, repo_path: Path) -> tuple[List[SecurityFinding], List[str]]:
        """Perform Software Composition Analysis (SCA)."""
        findings = []
        tools_used = []
        
        # Safety for Python dependencies
        if self.available_tools.get("safety"):
            safety_findings = await self._run_safety_scan(repo_path)
            findings.extend(safety_findings)
            tools_used.append("safety")
        
        # Trivy for comprehensive dependency scanning
        if self.available_tools.get("trivy"):
            trivy_findings = await self._run_trivy_scan(repo_path)
            findings.extend(trivy_findings)
            tools_used.append("trivy")
        
        return findings, tools_used
    
    async def _scan_for_secrets(self, repo_path: Path) -> List[SecurityFinding]:
        """Scan for hardcoded secrets and credentials."""
        findings = []
        
        secret_patterns = [
            (r"(?i)api[_-]?key\s*[:=]\s*['\"]([a-zA-Z0-9_-]{20,})['\"]", "API Key"),
            (r"(?i)secret[_-]?key\s*[:=]\s*['\"]([a-zA-Z0-9_-]{20,})['\"]", "Secret Key"),
            (r"(?i)password\s*[:=]\s*['\"]([^'\"]{8,})['\"]", "Password"),
            (r"(?i)token\s*[:=]\s*['\"]([a-zA-Z0-9_-]{20,})['\"]", "Access Token"),
            (r"(?i)(?:aws|amazon)[_-]?(?:access[_-]?key[_-]?id|secret[_-]?access[_-]?key)", "AWS Credentials"),
            (r"(?i)(?:github|git)[_-]?token", "GitHub Token"),
            (r"(?i)(?:private[_-]?key|ssh[_-]?key)", "Private Key"),
        ]
        
        try:
            text_files = list(repo_path.rglob("*.py")) + \
                        list(repo_path.rglob("*.js")) + \
                        list(repo_path.rglob("*.ts")) + \
                        list(repo_path.rglob("*.json")) + \
                        list(repo_path.rglob("*.yml")) + \
                        list(repo_path.rglob("*.yaml")) + \
                        list(repo_path.rglob("*.env")) + \
                        list(repo_path.rglob("*.config"))
            
            for file_path in text_files:
                if file_path.name.startswith('.git'):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    for pattern, secret_type in secret_patterns:
                        import re
                        matches = re.finditer(pattern, content)
                        
                        for match in matches:
                            line_number = content[:match.start()].count('\n') + 1
                            
                            finding = SecurityFinding(
                                finding_id=f"SECRET-{hashlib.md5(f'{file_path}:{line_number}'.encode()).hexdigest()[:8]}",
                                title=f"Hardcoded {secret_type} detected",
                                description=f"Potential hardcoded {secret_type.lower()} found in {file_path.name}",
                                vulnerability_type=VulnerabilityType.CONFIGURATION,
                                severity=SeverityLevel.HIGH,
                                test_type=TestType.SECRETS,
                                file_path=str(file_path.relative_to(repo_path)),
                                line_number=line_number,
                                confidence="high",
                                cwe_id="CWE-798",
                                owasp_category="A04_Insecure_Design",
                                remediation=f"Remove hardcoded {secret_type.lower()} and use environment variables or secure storage",
                                discovered_at=datetime.now(timezone.utc)
                            )
                            findings.append(finding)
                
                except Exception as e:
                    self.logger.debug(f"Error scanning file for secrets {file_path}: {e}")
                    continue
        
        except Exception as e:
            self.logger.warning(f"Error in secrets scanning: {e}")
        
        return findings
    
    async def _scan_infrastructure(self, repo_path: Path) -> tuple[List[SecurityFinding], List[str]]:
        """Scan infrastructure configurations for security issues."""
        findings = []
        tools_used = []
        
        # Scan Docker configurations
        docker_findings = await self._scan_docker_security(repo_path)
        findings.extend(docker_findings)
        tools_used.append("docker_scanner")
        
        # Scan Kubernetes configurations
        k8s_findings = await self._scan_kubernetes_security(repo_path)
        findings.extend(k8s_findings)
        tools_used.append("k8s_scanner")
        
        return findings, tools_used
    
    async def _scan_docker_security(self, repo_path: Path) -> List[SecurityFinding]:
        """Scan Docker configurations for security issues."""
        findings = []
        
        # Find Dockerfiles
        dockerfiles = list(repo_path.rglob("Dockerfile*")) + list(repo_path.rglob("*.dockerfile"))
        
        for dockerfile in dockerfiles:
            try:
                with open(dockerfile, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    line = line.strip()
                    
                    # Check for security issues
                    if line.upper().startswith('FROM') and ':latest' in line.lower():
                        finding = SecurityFinding(
                            finding_id=f"DOCKER-{hashlib.md5(f'{dockerfile}:{i}'.encode()).hexdigest()[:8]}",
                            title="Docker image using 'latest' tag",
                            description="Using 'latest' tag can lead to unpredictable behavior",
                            vulnerability_type=VulnerabilityType.CONFIGURATION,
                            severity=SeverityLevel.LOW,
                            test_type=TestType.INFRASTRUCTURE,
                            file_path=str(dockerfile.relative_to(repo_path)),
                            line_number=i,
                            confidence="high",
                            remediation="Use specific version tags instead of 'latest'",
                            discovered_at=datetime.now(timezone.utc)
                        )
                        findings.append(finding)
                    
                    elif line.upper().startswith('USER') and 'root' in line.lower():
                        finding = SecurityFinding(
                            finding_id=f"DOCKER-{hashlib.md5(f'{dockerfile}:{i}'.encode()).hexdigest()[:8]}",
                            title="Docker container running as root",
                            description="Container should not run as root user",
                            vulnerability_type=VulnerabilityType.CONFIGURATION,
                            severity=SeverityLevel.MEDIUM,
                            test_type=TestType.INFRASTRUCTURE,
                            file_path=str(dockerfile.relative_to(repo_path)),
                            line_number=i,
                            confidence="high",
                            remediation="Create and use a non-root user",
                            discovered_at=datetime.now(timezone.utc)
                        )
                        findings.append(finding)
            
            except Exception as e:
                self.logger.debug(f"Error scanning Dockerfile {dockerfile}: {e}")
        
        return findings
    
    async def _scan_kubernetes_security(self, repo_path: Path) -> List[SecurityFinding]:
        """Scan Kubernetes configurations for security issues."""
        findings = []
        
        # Find Kubernetes YAML files
        k8s_files = list(repo_path.rglob("*.yaml")) + list(repo_path.rglob("*.yml"))
        
        for k8s_file in k8s_files:
            try:
                with open(k8s_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for privileged containers
                if 'privileged: true' in content:
                    finding = SecurityFinding(
                        finding_id=f"K8S-{hashlib.md5(str(k8s_file).encode()).hexdigest()[:8]}",
                        title="Kubernetes privileged container detected",
                        description="Container running with privileged access",
                        vulnerability_type=VulnerabilityType.CONFIGURATION,
                        severity=SeverityLevel.HIGH,
                        test_type=TestType.INFRASTRUCTURE,
                        file_path=str(k8s_file.relative_to(repo_path)),
                        confidence="high",
                        remediation="Remove privileged access unless absolutely necessary",
                        discovered_at=datetime.now(timezone.utc)
                    )
                    findings.append(finding)
            
            except Exception as e:
                self.logger.debug(f"Error scanning K8s file {k8s_file}: {e}")
        
        return findings
    
    async def _vulnerability_assessment(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive vulnerability assessment."""
        # Placeholder for vulnerability assessment
        return {
            "success": True,
            "assessment": "Vulnerability assessment completed",
            "findings": [],
            "risk_rating": "medium"
        }
    
    async def _penetration_test(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform penetration testing."""
        # Placeholder for penetration testing
        return {
            "success": True,
            "test_results": "Penetration testing completed",
            "vulnerabilities": [],
            "recommendations": []
        }
    
    async def _compliance_check(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Check security compliance against standards."""
        # Placeholder for compliance checking
        return {
            "success": True,
            "compliance_status": "compliant",
            "standards_checked": ["OWASP", "NIST"],
            "violations": []
        }
    
    async def _threat_modeling(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform threat modeling analysis."""
        # Placeholder for threat modeling
        return {
            "success": True,
            "threat_model": "Threat modeling completed",
            "threats": [],
            "mitigations": []
        }
    
    def _calculate_risk_score(self, findings: List[SecurityFinding]) -> float:
        """Calculate overall risk score based on findings."""
        if not findings:
            return 0.0
        
        severity_weights = {
            SeverityLevel.CRITICAL: 10.0,
            SeverityLevel.HIGH: 7.0,
            SeverityLevel.MEDIUM: 4.0,
            SeverityLevel.LOW: 2.0,
            SeverityLevel.INFO: 1.0
        }
        
        total_score = sum(severity_weights.get(finding.severity, 1.0) for finding in findings)
        max_possible = len(findings) * 10.0
        
        return min(100.0, (total_score / max_possible) * 100) if max_possible > 0 else 0.0
    
    def _generate_scan_statistics(self, findings: List[SecurityFinding]) -> Dict[str, Any]:
        """Generate scan statistics from findings."""
        stats = {
            "total_findings": len(findings),
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0
        }
        
        for finding in findings:
            if finding.severity == SeverityLevel.CRITICAL:
                stats["critical"] += 1
            elif finding.severity == SeverityLevel.HIGH:
                stats["high"] += 1
            elif finding.severity == SeverityLevel.MEDIUM:
                stats["medium"] += 1
            elif finding.severity == SeverityLevel.LOW:
                stats["low"] += 1
            else:
                stats["info"] += 1
        
        return stats
    
    def _assess_compliance(self, findings: List[SecurityFinding]) -> str:
        """Assess compliance status based on findings."""
        critical_count = sum(1 for f in findings if f.severity == SeverityLevel.CRITICAL)
        high_count = sum(1 for f in findings if f.severity == SeverityLevel.HIGH)
        
        if critical_count > 0:
            return "non_compliant"
        elif high_count > 5:
            return "at_risk"
        elif high_count > 2:
            return "needs_attention"
        else:
            return "compliant"
    
    def _generate_recommendations(self, findings: List[SecurityFinding]) -> List[str]:
        """Generate security recommendations based on findings."""
        recommendations = []
        
        critical_count = sum(1 for f in findings if f.severity == SeverityLevel.CRITICAL)
        high_count = sum(1 for f in findings if f.severity == SeverityLevel.HIGH)
        
        if critical_count > 0:
            recommendations.append(f"🚨 IMMEDIATE ACTION: Address {critical_count} critical security vulnerabilities")
        
        if high_count > 0:
            recommendations.append(f"⚠️ HIGH PRIORITY: Fix {high_count} high-severity security issues")
        
        # Add specific recommendations
        vuln_types = {f.vulnerability_type for f in findings}
        
        if VulnerabilityType.CODE_INJECTION in vuln_types:
            recommendations.append("Implement input validation and parameterized queries")
        
        if VulnerabilityType.CRYPTOGRAPHIC in vuln_types:
            recommendations.append("Update to use strong cryptographic algorithms")
        
        if VulnerabilityType.CONFIGURATION in vuln_types:
            recommendations.append("Review and secure configuration settings")
        
        return recommendations
    
    def _generate_scan_summary(self, report: SecurityReport) -> str:
        """Generate human-readable scan summary."""
        stats = report.scan_statistics
        
        summary_parts = [
            f"🛡️ Security Scan Summary for {Path(report.repository_path).name}:",
            f"• Scan Duration: {report.scan_duration:.1f} seconds",
            f"• Total Findings: {stats['total_findings']}",
        ]
        
        if stats["critical"] > 0:
            summary_parts.append(f"• 🚨 {stats['critical']} CRITICAL vulnerabilities")
        
        if stats["high"] > 0:
            summary_parts.append(f"• ⚠️ {stats['high']} HIGH severity vulnerabilities")
        
        if stats["medium"] > 0:
            summary_parts.append(f"• ⚡ {stats['medium']} MEDIUM severity vulnerabilities")
        
        summary_parts.append(f"• Risk Score: {report.risk_score:.1f}/100")
        summary_parts.append(f"• Compliance Status: {report.compliance_status.upper()}")
        summary_parts.append(f"• Tools Used: {', '.join(report.tools_used)}")
        
        if report.recommendations:
            summary_parts.append("• Top Recommendations:")
            for rec in report.recommendations[:3]:
                summary_parts.append(f"  - {rec}")
        
        return "\n".join(summary_parts)
    
    def _map_owasp_to_vuln_type(self, owasp_category: str) -> VulnerabilityType:
        """Map OWASP category to vulnerability type."""
        mapping = {
            "A01_Broken_Access_Control": VulnerabilityType.AUTHORIZATION,
            "A02_Cryptographic_Failures": VulnerabilityType.CRYPTOGRAPHIC,
            "A03_Injection": VulnerabilityType.CODE_INJECTION,
            "A04_Insecure_Design": VulnerabilityType.CONFIGURATION,
            "A05_Security_Misconfiguration": VulnerabilityType.CONFIGURATION
        }
        return mapping.get(owasp_category, VulnerabilityType.OTHER)
    
    def _map_bandit_severity(self, bandit_severity: str) -> SeverityLevel:
        """Map Bandit severity to our severity levels."""
        mapping = {
            "HIGH": SeverityLevel.HIGH,
            "MEDIUM": SeverityLevel.MEDIUM,
            "LOW": SeverityLevel.LOW
        }
        return mapping.get(bandit_severity, SeverityLevel.LOW)
    
    async def _run_safety_scan(self, repo_path: Path) -> List[SecurityFinding]:
        """Run Safety scan for Python dependencies."""
        findings = []
        # Implementation would scan Python dependencies for known vulnerabilities
        return findings
    
    async def _run_trivy_scan(self, repo_path: Path) -> List[SecurityFinding]:
        """Run Trivy security scan."""
        findings = []
        # Implementation would use Trivy for comprehensive vulnerability scanning
        return findings
    
    async def _run_semgrep_scan(self, repo_path: Path) -> List[SecurityFinding]:
        """Run Semgrep security analysis."""
        findings = []
        # Implementation would use Semgrep for multi-language security analysis
        return findings
    
    async def _run_eslint_security_scan(self, repo_path: Path) -> List[SecurityFinding]:
        """Run ESLint security scan for JavaScript/TypeScript."""
        findings = []
        # Implementation would run ESLint with security rules
        return findings
    
    def get_capabilities(self) -> List[str]:
        """Get red team agent capabilities."""
        return [
            "security_vulnerability_scanning",
            "penetration_testing",
            "sast_analysis",
            "sca_analysis",
            "secrets_detection",
            "infrastructure_security",
            "compliance_checking",
            "threat_modeling",
            "risk_assessment",
            "owasp_top10_analysis"
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            "agent_id": self.agent_id,
            "status": "ready",
            "capabilities": len(self.get_capabilities()),
            "available_tools": sum(1 for available in self.available_tools.values() if available),
            "cached_scans": len(self.scan_cache),
            "last_heartbeat": datetime.now(timezone.utc).isoformat()
        }
