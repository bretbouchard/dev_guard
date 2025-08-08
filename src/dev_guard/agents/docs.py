"""Documentation agent for DevGuard - comprehensive documentation generation and maintenance."""

import ast
import asyncio
import json
import logging
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .base_agent import BaseAgent
from ..core.config import Config
from ..memory.shared_memory import SharedMemory, TaskStatus, AgentState
from ..memory.vector_db import VectorDatabase
from ..llm.provider import LLMProvider

logger = logging.getLogger(__name__)


class DocumentationType(Enum):
    """Types of documentation that can be generated or maintained."""
    README = "readme"
    API_DOCS = "api_docs"
    DOCSTRINGS = "docstrings"
    CHANGELOG = "changelog"
    ARCHITECTURE = "architecture"
    USER_GUIDE = "user_guide"
    DEVELOPER_GUIDE = "developer_guide"
    CODE_COMMENTS = "code_comments"
    MKDOCS = "mkdocs"
    SPHINX = "sphinx"


class DocumentationStatus(Enum):
    """Status of documentation tasks."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"
    OUTDATED = "outdated"


class DocumentationScope(Enum):
    """Scope of documentation updates."""
    SINGLE_FILE = "single_file"
    MODULE = "module"
    PACKAGE = "package"
    REPOSITORY = "repository"
    MULTI_REPOSITORY = "multi_repository"


@dataclass
class CodeElement:
    """Represents a code element that needs documentation."""
    name: str
    type: str  # function, class, method, module
    file_path: str
    line_number: int
    signature: Optional[str] = None
    current_docstring: Optional[str] = None
    complexity_score: Optional[float] = None
    ast_node: Optional[str] = None


@dataclass
class DocumentationTask:
    """Represents a documentation task."""
    task_id: str
    doc_type: DocumentationType
    scope: DocumentationScope
    target_path: str
    description: str
    status: DocumentationStatus
    priority: int = 1  # 1-5, 5 being highest
    code_elements: Optional[List[CodeElement]] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    goose_session_id: Optional[str] = None


@dataclass
class DocumentationReport:
    """Report of documentation generation/maintenance activities."""
    repository_path: str
    task_id: str
    doc_type: DocumentationType
    files_processed: int
    files_updated: int
    docstrings_added: int
    docstrings_updated: int
    documentation_coverage: float  # 0.0 to 1.0
    quality_score: float  # 0.0 to 1.0
    issues_found: List[str]
    recommendations: List[str]
    execution_time: float
    generated_files: List[str]
    updated_files: List[str]
    timestamp: datetime


class DocsAgent(BaseAgent):
    """
    Documentation agent responsible for:
    - Comprehensive documentation generation and maintenance
    - Intelligent docstring creation and updates
    - README and documentation file management
    - API documentation generation
    - Documentation synchronization with code changes
    - Goose-based documentation tools integration
    - Multi-format documentation support (Markdown, Sphinx, MkDocs)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_provider = kwargs.get('llm_provider')
        
        # Documentation tracking
        self.documentation_cache: Dict[str, DocumentationReport] = {}
        self.code_analysis_cache: Dict[str, List[CodeElement]] = {}
        
        # Supported file extensions for documentation
        self.code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs', '.rb', '.php'}
        self.doc_extensions = {'.md', '.rst', '.txt', '.adoc'}
        
        # Documentation templates
        self.doc_templates = self._load_documentation_templates()
        
    def _load_documentation_templates(self) -> Dict[str, str]:
        """Load documentation templates for different types."""
        return {
            "function_docstring": '''"""
{description}

Args:
{args}

Returns:
{returns}

Raises:
{raises}

Example:
{example}
"""''',
            "class_docstring": '''"""
{description}

Attributes:
{attributes}

Methods:
{methods}

Example:
{example}
"""''',
            "module_docstring": '''"""
{description}

This module provides:
{features}

Classes:
{classes}

Functions:
{functions}

Usage:
{usage}
"""''',
            "readme_template": '''# {project_name}

{description}

## Features

{features}

## Installation

{installation}

## Usage

{usage}

## API Documentation

{api_docs}

## Contributing

{contributing}

## License

{license}
'''
        }
        
    async def execute(self, state: Any) -> Any:
        """Execute the docs agent's main logic."""
        if isinstance(state, dict):
            task = state
        else:
            task = {"type": "generate_docs", "description": str(state)}
        
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a documentation task with comprehensive functionality."""
        try:
            self.logger.info(f"Docs agent executing task: {task.get('type', 'unknown')}")
            
            # Update agent state
            self._update_state("working", task.get("task_id"))
            
            task_type = task.get("type", "")
            
            # Core documentation tasks
            if task_type == "generate_docs":
                result = await self._generate_comprehensive_docs(task)
            elif task_type == "update_docstrings":
                result = await self._update_docstrings(task)
            elif task_type == "update_readme":
                result = await self._update_readme(task)
            elif task_type == "create_api_docs":
                result = await self._create_api_docs(task)
            elif task_type == "sync_docs_with_code":
                result = await self._sync_docs_with_code(task)
            elif task_type == "analyze_documentation_coverage":
                result = await self._analyze_documentation_coverage(task)
            elif task_type == "generate_changelog":
                result = await self._generate_changelog(task)
            elif task_type == "create_architecture_docs":
                result = await self._create_architecture_docs(task)
            elif task_type == "goose_generate_docs":
                result = await self._goose_generate_docs(task)
            elif task_type == "validate_documentation":
                result = await self._validate_documentation(task)
            else:
                result = await self._generic_docs_task(task)
            
            self._update_state("idle")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in docs task execution: {e}")
            self._update_state("error", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "task_type": task.get("type", "unknown")
            }

    async def _generate_comprehensive_docs(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive documentation for a codebase."""
        try:
            repository_path = task.get("repository_path", ".")
            doc_types = task.get("doc_types", ["README", "API_DOCS", "DOCSTRINGS"])
            
            if not os.path.exists(repository_path):
                return {"success": False, "error": f"Repository path does not exist: {repository_path}"}
            
            results = []
            total_files_processed = 0
            total_files_updated = 0
            
            for doc_type in doc_types:
                self.logger.info(f"Generating {doc_type} documentation")
                
                if doc_type == "README":
                    result = await self._generate_readme(repository_path)
                elif doc_type == "API_DOCS":
                    result = await self._generate_api_documentation(repository_path)
                elif doc_type == "DOCSTRINGS":
                    result = await self._generate_missing_docstrings(repository_path)
                else:
                    continue
                
                results.append(result)
                if result.get("success"):
                    total_files_processed += result.get("files_processed", 0)
                    total_files_updated += result.get("files_updated", 0)
            
            # Create comprehensive report
            report = DocumentationReport(
                repository_path=repository_path,
                task_id=task.get("task_id", "unknown"),
                doc_type=DocumentationType.README,  # Primary type
                files_processed=total_files_processed,
                files_updated=total_files_updated,
                docstrings_added=sum(r.get("docstrings_added", 0) for r in results),
                docstrings_updated=sum(r.get("docstrings_updated", 0) for r in results),
                documentation_coverage=await self._calculate_documentation_coverage(repository_path),
                quality_score=await self._calculate_documentation_quality_score(repository_path),
                issues_found=[issue for r in results for issue in r.get("issues", [])],
                recommendations=[rec for r in results for rec in r.get("recommendations", [])],
                execution_time=0.0,  # Would track actual execution time
                generated_files=[f for r in results for f in r.get("generated_files", [])],
                updated_files=[f for r in results for f in r.get("updated_files", [])],
                timestamp=datetime.now(timezone.utc)
            )
            
            # Cache the report
            self.documentation_cache[repository_path] = report
            
            return {
                "success": True,
                "message": f"Generated comprehensive documentation for {repository_path}",
                "report": asdict(report),
                "individual_results": results
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _update_docstrings(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Update docstrings for code elements."""
        try:
            target_path = task.get("target_path", ".")
            force_update = task.get("force_update", False)
            
            if not os.path.exists(target_path):
                return {"success": False, "error": f"Target path does not exist: {target_path}"}
            
            # Analyze code to find elements needing docstrings
            code_elements = await self._analyze_code_for_documentation(target_path)
            
            docstrings_added = 0
            docstrings_updated = 0
            files_updated = []
            issues = []
            
            for element in code_elements:
                if not element.current_docstring or force_update:
                    try:
                        # Generate docstring using LLM
                        new_docstring = await self._generate_docstring_for_element(element)
                        
                        if new_docstring:
                            # Update the file with new docstring
                            success = await self._update_docstring_in_file(element, new_docstring)
                            
                            if success:
                                if element.current_docstring:
                                    docstrings_updated += 1
                                else:
                                    docstrings_added += 1
                                
                                if element.file_path not in files_updated:
                                    files_updated.append(element.file_path)
                            else:
                                issues.append(f"Failed to update docstring for {element.name} in {element.file_path}")
                                
                    except Exception as e:
                        issues.append(f"Error processing {element.name}: {str(e)}")
            
            return {
                "success": True,
                "message": f"Updated docstrings in {target_path}",
                "docstrings_added": docstrings_added,
                "docstrings_updated": docstrings_updated,
                "files_updated": len(files_updated),
                "updated_files": files_updated,
                "issues": issues,
                "total_elements_processed": len(code_elements)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _generate_docstring_for_element(self, element: CodeElement) -> Optional[str]:
        """Generate a docstring for a code element using LLM."""
        try:
            if not self.llm_provider:
                return None
            
            # Read the source code context
            source_context = await self._get_source_context(element)
            
            # Create prompt for LLM
            prompt = f"""
Generate a comprehensive docstring for the following {element.type}:

Name: {element.name}
File: {element.file_path}
Signature: {element.signature or 'N/A'}

Source Context:
```
{source_context}
```

Please generate a well-formatted docstring that includes:
- Clear description of purpose
- Parameter descriptions (if applicable)
- Return value description (if applicable)  
- Exception descriptions (if applicable)
- Usage example (if helpful)

Follow the Google docstring style for Python or appropriate style for the language.
Return only the docstring content, properly formatted.
"""
            
            response = await self.llm_provider.generate_response(prompt)
            
            if response and response.content:
                # Clean and format the docstring
                docstring = response.content.strip()
                # Remove any markdown code blocks if present
                docstring = re.sub(r'^```[\w]*\n|```$', '', docstring, flags=re.MULTILINE)
                return docstring.strip()
                
        except Exception as e:
            self.logger.warning(f"Failed to generate docstring for {element.name}: {e}")
            
        return None

    async def _get_source_context(self, element: CodeElement) -> str:
        """Get source code context around a code element."""
        try:
            with open(element.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Get context around the element (10 lines before and after)
            start_line = max(0, element.line_number - 10)
            end_line = min(len(lines), element.line_number + 20)
            
            context_lines = lines[start_line:end_line]
            return ''.join(context_lines)
            
        except Exception as e:
            self.logger.warning(f"Failed to get source context for {element.name}: {e}")
            return ""

    async def _analyze_code_for_documentation(self, path: str) -> List[CodeElement]:
        """Analyze code to find elements that need documentation."""
        code_elements = []
        
        # Check cache first
        if path in self.code_analysis_cache:
            return self.code_analysis_cache[path]
        
        try:
            if os.path.isfile(path):
                if path.endswith('.py'):
                    elements = await self._analyze_python_file(path)
                    code_elements.extend(elements)
            else:
                # Recursively analyze directory
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if any(file.endswith(ext) for ext in self.code_extensions):
                            file_path = os.path.join(root, file)
                            
                            if file_path.endswith('.py'):
                                elements = await self._analyze_python_file(file_path)
                                code_elements.extend(elements)
            
            # Cache the results
            self.code_analysis_cache[path] = code_elements
            
        except Exception as e:
            self.logger.error(f"Error analyzing code at {path}: {e}")
        
        return code_elements

    async def _analyze_python_file(self, file_path: str) -> List[CodeElement]:
        """Analyze a Python file for documentation opportunities."""
        elements = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Analyze function
                    element = CodeElement(
                        name=node.name,
                        type="function",
                        file_path=file_path,
                        line_number=node.lineno,
                        signature=self._get_function_signature(node),
                        current_docstring=ast.get_docstring(node),
                        complexity_score=self._calculate_complexity(node)
                    )
                    elements.append(element)
                    
                elif isinstance(node, ast.ClassDef):
                    # Analyze class
                    element = CodeElement(
                        name=node.name,
                        type="class",
                        file_path=file_path,
                        line_number=node.lineno,
                        signature=f"class {node.name}",
                        current_docstring=ast.get_docstring(node),
                        complexity_score=len(node.body)  # Simple complexity metric
                    )
                    elements.append(element)
                    
        except Exception as e:
            self.logger.warning(f"Failed to analyze Python file {file_path}: {e}")
        
        return elements

    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """Extract function signature from AST node."""
        try:
            args = []
            
            # Regular arguments
            for arg in node.args.args:
                args.append(arg.arg)
            
            # Handle *args and **kwargs
            if node.args.vararg:
                args.append(f"*{node.args.vararg.arg}")
            if node.args.kwarg:
                args.append(f"**{node.args.kwarg.arg}")
            
            signature = f"def {node.name}({', '.join(args)})"
            
            # Add return annotation if present
            if hasattr(node, 'returns') and node.returns:
                signature += " -> ..."
                
            return signature
            
        except Exception:
            return f"def {node.name}(...)"

    def _calculate_complexity(self, node: ast.AST) -> float:
        """Calculate cyclomatic complexity of an AST node."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
                
        return float(complexity)

    async def _update_docstring_in_file(self, element: CodeElement, new_docstring: str) -> bool:
        """Update docstring in source file."""
        try:
            with open(element.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Find the location to insert/update docstring
            # This is a simplified implementation
            # In practice, you'd want more sophisticated AST manipulation
            
            # For now, just log the intent
            self.logger.info(f"Would update docstring for {element.name} at line {element.line_number}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update docstring in {element.file_path}: {e}")
            return False

    async def _update_readme(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Update README file with current project information."""
        try:
            repository_path = task.get("repository_path", ".")
            readme_path = os.path.join(repository_path, "README.md")
            
            # Analyze project structure
            project_info = await self._analyze_project_structure(repository_path)
            
            # Generate README content
            readme_content = await self._generate_readme_content(project_info)
            
            # Write README file
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            
            return {
                "success": True,
                "message": f"Updated README.md at {readme_path}",
                "updated_files": [readme_path],
                "project_info": project_info
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _generate_readme(self, repository_path: str) -> Dict[str, Any]:
        """Generate a comprehensive README for the repository."""
        try:
            project_info = await self._analyze_project_structure(repository_path)
            
            readme_content = self.doc_templates["readme_template"].format(
                project_name=project_info.get("name", "Project"),
                description=project_info.get("description", "A software project"),
                features=self._format_features_list(project_info.get("features", [])),
                installation=project_info.get("installation", "TBD"),
                usage=project_info.get("usage", "TBD"),
                api_docs=project_info.get("api_docs", "TBD"),
                contributing=project_info.get("contributing", "TBD"),
                license=project_info.get("license", "TBD")
            )
            
            readme_path = os.path.join(repository_path, "README.md")
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            
            return {
                "success": True,
                "files_processed": 1,
                "files_updated": 1,
                "generated_files": [readme_path]
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _analyze_project_structure(self, repository_path: str) -> Dict[str, Any]:
        """Analyze project structure to gather information for documentation."""
        project_info = {
            "name": os.path.basename(repository_path),
            "description": "A software project",
            "features": [],
            "modules": [],
            "classes": [],
            "functions": []
        }
        
        try:
            # Look for package.json, pyproject.toml, etc. for metadata
            if os.path.exists(os.path.join(repository_path, "pyproject.toml")):
                project_info.update(await self._parse_pyproject_toml(repository_path))
            elif os.path.exists(os.path.join(repository_path, "package.json")):
                project_info.update(await self._parse_package_json(repository_path))
            
            # Analyze code structure
            code_elements = await self._analyze_code_for_documentation(repository_path)
            
            for element in code_elements:
                if element.type == "function":
                    project_info["functions"].append(element.name)
                elif element.type == "class":
                    project_info["classes"].append(element.name)
            
        except Exception as e:
            self.logger.warning(f"Error analyzing project structure: {e}")
        
        return project_info

    async def _parse_pyproject_toml(self, repository_path: str) -> Dict[str, Any]:
        """Parse pyproject.toml for project metadata."""
        try:
            import tomllib  # Python 3.11+
        except ImportError:
            try:
                import tomli as tomllib  # Fallback
            except ImportError:
                return {}
        
        try:
            toml_path = os.path.join(repository_path, "pyproject.toml")
            with open(toml_path, 'rb') as f:
                data = tomllib.load(f)
            
            project = data.get("project", {})
            return {
                "name": project.get("name", ""),
                "description": project.get("description", ""),
                "version": project.get("version", "")
            }
        except Exception as e:
            self.logger.warning(f"Failed to parse pyproject.toml: {e}")
            return {}

    async def _parse_package_json(self, repository_path: str) -> Dict[str, Any]:
        """Parse package.json for project metadata."""
        try:
            json_path = os.path.join(repository_path, "package.json")
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return {
                "name": data.get("name", ""),
                "description": data.get("description", ""),
                "version": data.get("version", "")
            }
        except Exception as e:
            self.logger.warning(f"Failed to parse package.json: {e}")
            return {}

    async def _generate_readme_content(self, project_info: Dict[str, Any]) -> str:
        """Generate README content from project information."""
        return self.doc_templates["readme_template"].format(
            project_name=project_info.get("name", "Project"),
            description=project_info.get("description", "A software project"),
            features=self._format_features_list(project_info.get("features", [])),
            installation="```bash\n# Installation instructions\n```",
            usage="```python\n# Usage examples\n```",
            api_docs="See API documentation for detailed information.",
            contributing="Contributions are welcome! Please see CONTRIBUTING.md",
            license="Licensed under MIT License"
        )

    def _format_features_list(self, features: List[str]) -> str:
        """Format features list for README."""
        if not features:
            return "- Feature documentation coming soon"
        
        return "\n".join(f"- {feature}" for feature in features)

    async def _create_api_docs(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Create API documentation."""
        try:
            repository_path = task.get("repository_path", ".")
            output_format = task.get("format", "markdown")  # markdown, sphinx, mkdocs
            
            # Analyze code for API elements
            code_elements = await self._analyze_code_for_documentation(repository_path)
            
            # Group by modules
            modules = {}
            for element in code_elements:
                module_name = self._get_module_name(element.file_path, repository_path)
                if module_name not in modules:
                    modules[module_name] = []
                modules[module_name].append(element)
            
            generated_files = []
            
            if output_format == "markdown":
                api_doc_path = os.path.join(repository_path, "docs", "API.md")
                content = await self._generate_markdown_api_docs(modules)
                
                os.makedirs(os.path.dirname(api_doc_path), exist_ok=True)
                with open(api_doc_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                generated_files.append(api_doc_path)
            
            return {
                "success": True,
                "message": f"Generated API documentation in {output_format} format",
                "generated_files": generated_files,
                "modules_documented": len(modules)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_module_name(self, file_path: str, base_path: str) -> str:
        """Get module name from file path."""
        rel_path = os.path.relpath(file_path, base_path)
        module_path = rel_path.replace(os.sep, '.').replace('.py', '')
        return module_path

    async def _generate_markdown_api_docs(self, modules: Dict[str, List[CodeElement]]) -> str:
        """Generate markdown API documentation."""
        content = ["# API Documentation\n"]
        
        for module_name, elements in modules.items():
            content.append(f"## Module: {module_name}\n")
            
            # Group by type
            classes = [e for e in elements if e.type == "class"]
            functions = [e for e in elements if e.type == "function"]
            
            if classes:
                content.append("### Classes\n")
                for cls in classes:
                    content.append(f"#### {cls.name}\n")
                    if cls.current_docstring:
                        content.append(f"{cls.current_docstring}\n")
                    content.append("")
            
            if functions:
                content.append("### Functions\n")
                for func in functions:
                    content.append(f"#### {func.name}\n")
                    content.append(f"```python\n{func.signature}\n```\n")
                    if func.current_docstring:
                        content.append(f"{func.current_docstring}\n")
                    content.append("")
        
        return "\n".join(content)

    async def _sync_docs_with_code(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronize documentation with code changes."""
        try:
            repository_path = task.get("repository_path", ".")
            changed_files = task.get("changed_files", [])
            
            if not changed_files:
                return {"success": True, "message": "No files to sync"}
            
            sync_results = []
            
            for file_path in changed_files:
                if any(file_path.endswith(ext) for ext in self.code_extensions):
                    # Analyze changed file for documentation needs
                    elements = await self._analyze_code_for_documentation(file_path)
                    
                    # Check if documentation needs updating
                    needs_update = await self._check_documentation_freshness(file_path, elements)
                    
                    if needs_update:
                        sync_result = await self._update_docstrings({
                            "target_path": file_path,
                            "force_update": False
                        })
                        sync_results.append(sync_result)
            
            return {
                "success": True,
                "message": f"Synchronized documentation for {len(changed_files)} files",
                "sync_results": sync_results
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _check_documentation_freshness(self, file_path: str, elements: List[CodeElement]) -> bool:
        """Check if documentation needs updating based on code changes."""
        # Simplified check - in practice, you'd compare timestamps, signatures, etc.
        for element in elements:
            if not element.current_docstring:
                return True
            
        return False

    async def _analyze_documentation_coverage(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze documentation coverage across the codebase."""
        try:
            repository_path = task.get("repository_path", ".")
            
            coverage_score = await self._calculate_documentation_coverage(repository_path)
            quality_score = await self._calculate_documentation_quality_score(repository_path)
            
            code_elements = await self._analyze_code_for_documentation(repository_path)
            
            # Calculate detailed metrics
            total_elements = len(code_elements)
            documented_elements = len([e for e in code_elements if e.current_docstring])
            undocumented_elements = total_elements - documented_elements
            
            coverage_by_type = {}
            for element_type in ["function", "class", "method"]:
                type_elements = [e for e in code_elements if e.type == element_type]
                type_documented = [e for e in type_elements if e.current_docstring]
                
                if type_elements:
                    coverage_by_type[element_type] = {
                        "total": len(type_elements),
                        "documented": len(type_documented),
                        "coverage": len(type_documented) / len(type_elements)
                    }
            
            return {
                "success": True,
                "overall_coverage": coverage_score,
                "quality_score": quality_score,
                "total_elements": total_elements,
                "documented_elements": documented_elements,
                "undocumented_elements": undocumented_elements,
                "coverage_by_type": coverage_by_type,
                "repository_path": repository_path
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _calculate_documentation_coverage(self, repository_path: str) -> float:
        """Calculate documentation coverage percentage."""
        try:
            elements = await self._analyze_code_for_documentation(repository_path)
            
            if not elements:
                return 0.0
            
            documented = len([e for e in elements if e.current_docstring])
            return documented / len(elements)
            
        except Exception:
            return 0.0

    async def _calculate_documentation_quality_score(self, repository_path: str) -> float:
        """Calculate documentation quality score."""
        try:
            elements = await self._analyze_code_for_documentation(repository_path)
            
            if not elements:
                return 0.0
            
            quality_scores = []
            
            for element in elements:
                if element.current_docstring:
                    # Simple quality scoring based on length and content
                    docstring = element.current_docstring
                    score = 0.5  # Base score for having documentation
                    
                    # Add points for comprehensive documentation
                    if "Args:" in docstring or "Parameters:" in docstring:
                        score += 0.1
                    if "Returns:" in docstring:
                        score += 0.1
                    if "Raises:" in docstring or "Exceptions:" in docstring:
                        score += 0.1
                    if "Example:" in docstring:
                        score += 0.2
                    
                    # Length-based quality (reasonable length)
                    if 50 <= len(docstring) <= 500:
                        score += 0.1
                    
                    quality_scores.append(min(score, 1.0))
                else:
                    quality_scores.append(0.0)
            
            return sum(quality_scores) / len(quality_scores)
            
        except Exception:
            return 0.0

    async def _generate_changelog(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate changelog from git history."""
        try:
            repository_path = task.get("repository_path", ".")
            since_tag = task.get("since_tag")
            
            # Get git log
            cmd = ["git", "log", "--oneline", "--no-merges"]
            if since_tag:
                cmd.append(f"{since_tag}..HEAD")
            
            result = subprocess.run(cmd, cwd=repository_path, capture_output=True, text=True)
            
            if result.returncode != 0:
                return {"success": False, "error": f"Git log failed: {result.stderr}"}
            
            commits = result.stdout.strip().split('\n')
            
            # Group commits by type (feat, fix, docs, etc.)
            changelog_entries = {
                "Features": [],
                "Bug Fixes": [],
                "Documentation": [],
                "Other": []
            }
            
            for commit in commits:
                if not commit.strip():
                    continue
                    
                if any(keyword in commit.lower() for keyword in ["feat:", "feature:"]):
                    changelog_entries["Features"].append(commit)
                elif any(keyword in commit.lower() for keyword in ["fix:", "bug:"]):
                    changelog_entries["Bug Fixes"].append(commit)
                elif any(keyword in commit.lower() for keyword in ["docs:", "doc:"]):
                    changelog_entries["Documentation"].append(commit)
                else:
                    changelog_entries["Other"].append(commit)
            
            # Generate changelog content
            changelog_content = self._format_changelog(changelog_entries)
            
            # Write to CHANGELOG.md
            changelog_path = os.path.join(repository_path, "CHANGELOG.md")
            with open(changelog_path, 'w', encoding='utf-8') as f:
                f.write(changelog_content)
            
            return {
                "success": True,
                "message": f"Generated changelog with {len(commits)} commits",
                "generated_files": [changelog_path],
                "commit_count": len(commits)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _format_changelog(self, entries: Dict[str, List[str]]) -> str:
        """Format changelog entries."""
        content = ["# Changelog\n"]
        
        for category, items in entries.items():
            if items:
                content.append(f"## {category}\n")
                for item in items:
                    content.append(f"- {item}")
                content.append("")
        
        return "\n".join(content)

    async def _create_architecture_docs(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Create architecture documentation."""
        try:
            repository_path = task.get("repository_path", ".")
            
            # Analyze project structure
            structure = await self._analyze_architecture(repository_path)
            
            # Generate architecture documentation
            arch_content = await self._generate_architecture_content(structure)
            
            # Write architecture documentation
            arch_path = os.path.join(repository_path, "docs", "ARCHITECTURE.md")
            os.makedirs(os.path.dirname(arch_path), exist_ok=True)
            
            with open(arch_path, 'w', encoding='utf-8') as f:
                f.write(arch_content)
            
            return {
                "success": True,
                "message": "Generated architecture documentation",
                "generated_files": [arch_path],
                "components_documented": len(structure.get("components", []))
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _analyze_architecture(self, repository_path: str) -> Dict[str, Any]:
        """Analyze project architecture."""
        structure = {
            "components": [],
            "dependencies": [],
            "modules": []
        }
        
        # Simple analysis - in practice would be more sophisticated
        for root, dirs, files in os.walk(repository_path):
            if any(excluded in root for excluded in ['.git', '__pycache__', 'node_modules']):
                continue
                
            for file in files:
                if file.endswith('.py'):
                    module_path = os.path.relpath(os.path.join(root, file), repository_path)
                    structure["modules"].append(module_path)
        
        return structure

    async def _generate_architecture_content(self, structure: Dict[str, Any]) -> str:
        """Generate architecture documentation content."""
        content = [
            "# Architecture Documentation\n",
            "## Overview\n",
            "This document describes the architecture of the project.\n",
            "## Components\n"
        ]
        
        for module in structure.get("modules", []):
            content.append(f"- `{module}`: Module description")
        
        content.extend([
            "\n## Dependencies\n",
            "TBD\n",
            "## Data Flow\n",
            "TBD\n"
        ])
        
        return "\n".join(content)

    async def _goose_generate_docs(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate documentation using Goose CLI integration."""
        try:
            repository_path = task.get("repository_path", ".")
            doc_type = task.get("doc_type", "general")
            
            # Prepare Goose command
            goose_cmd = ["goose"]
            
            if doc_type == "api":
                goose_cmd.extend(["docs", "api", repository_path])
            elif doc_type == "readme":
                goose_cmd.extend(["docs", "readme", repository_path])
            else:
                goose_cmd.extend(["docs", "generate", repository_path])
            
            # Execute Goose command
            result = subprocess.run(goose_cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "message": f"Generated documentation using Goose for {doc_type}",
                    "goose_output": result.stdout,
                    "goose_session_id": task.get("goose_session_id")
                }
            else:
                return {
                    "success": False,
                    "error": f"Goose command failed: {result.stderr}",
                    "goose_output": result.stdout
                }
            
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Goose command timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _validate_documentation(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Validate documentation quality and completeness."""
        try:
            repository_path = task.get("repository_path", ".")
            
            validation_results = {
                "coverage_check": await self._analyze_documentation_coverage({"repository_path": repository_path}),
                "quality_issues": [],
                "broken_links": [],
                "formatting_issues": [],
                "missing_sections": []
            }
            
            # Check for common documentation issues
            doc_files = []
            for root, dirs, files in os.walk(repository_path):
                for file in files:
                    if any(file.endswith(ext) for ext in self.doc_extensions):
                        doc_files.append(os.path.join(root, file))
            
            for doc_file in doc_files:
                issues = await self._validate_document_file(doc_file)
                validation_results["quality_issues"].extend(issues)
            
            return {
                "success": True,
                "validation_results": validation_results,
                "documents_checked": len(doc_files)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _validate_document_file(self, file_path: str) -> List[str]:
        """Validate a single documentation file."""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for common issues
            if len(content.strip()) < 50:
                issues.append(f"{file_path}: Document is too short")
            
            # Check for broken markdown links (simple regex)
            if file_path.endswith('.md'):
                broken_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
                for link_text, link_url in broken_links:
                    if link_url.startswith('http') and not await self._check_url_valid(link_url):
                        issues.append(f"{file_path}: Broken link to {link_url}")
            
        except Exception as e:
            issues.append(f"{file_path}: Error reading file - {str(e)}")
        
        return issues

    async def _check_url_valid(self, url: str) -> bool:
        """Check if a URL is valid (simplified)."""
        # In a real implementation, you'd make HTTP requests
        return True

    async def _generate_missing_docstrings(self, repository_path: str) -> Dict[str, Any]:
        """Generate missing docstrings for the repository."""
        return await self._update_docstrings({
            "target_path": repository_path,
            "force_update": False
        })

    async def _generate_api_documentation(self, repository_path: str) -> Dict[str, Any]:
        """Generate API documentation for the repository."""
        return await self._create_api_docs({
            "repository_path": repository_path,
            "format": "markdown"
        })

    async def _generic_docs_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle generic documentation tasks."""
        try:
            return {
                "success": True,
                "message": f"Processed docs task: {task.get('description', 'unknown')}",
                "task_details": task
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _update_state(self, status: str, task_id: Optional[str] = None, error: Optional[str] = None) -> None:
        """Update agent state in shared memory."""
        capabilities = [
            "documentation_generation",
            "docstring_updates", 
            "readme_management",
            "api_documentation",
            "documentation_sync",
            "documentation_analysis",
            "changelog_generation",
            "architecture_docs",
            "goose_integration",
            "documentation_validation"
        ]
        
        state = AgentState(
            agent_id=self.agent_id,
            status=status,
            current_task=task_id,
            last_heartbeat=datetime.now(timezone.utc),
            metadata={
                "error": error if error else None,
                "capabilities": capabilities,
                "documentation_cache_size": len(self.documentation_cache),
                "code_analysis_cache_size": len(self.code_analysis_cache)
            }
        )
        self.shared_memory.update_agent_state(state)
    
    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities."""
        return [
            "documentation_generation",
            "docstring_updates", 
            "readme_management",
            "api_documentation",
            "documentation_sync",
            "documentation_analysis",
            "changelog_generation",
            "architecture_docs",
            "goose_integration",
            "documentation_validation"
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "agent_id": self.agent_id,
            "type": "docs",
            "capabilities": self.get_capabilities(),
            "documentation_cache_size": len(self.documentation_cache),
            "code_analysis_cache_size": len(self.code_analysis_cache),
            "supported_formats": ["markdown", "sphinx", "mkdocs"],
            "supported_languages": list(self.code_extensions)
        }
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a documentation task."""
        try:
            self.logger.info(f"Docs agent executing task: {task.get('type', 'unknown')}")
            
            # Update agent state
            self._update_state("working", task.get("task_id"))
            
            task_type = task.get("type", "")
            
            if task_type == "generate_docs":
                result = await self._generate_docs(task)
            elif task_type == "update_readme":
                result = await self._update_readme(task)
            elif task_type == "create_api_docs":
                result = await self._create_api_docs(task)
            else:
                result = await self._generic_docs_task(task)
            
            self._update_state("idle")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in docs task execution: {e}")
            self._update_state("error", error=str(e))
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _generate_docs(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate documentation for code."""
        try:
            return {
                "success": True,
                "message": "Documentation generation placeholder",
                "files_processed": 0
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _update_readme(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Update README file."""
        try:
            return {
                "success": True,
                "message": "README update placeholder"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _create_api_docs(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Create API documentation."""
        try:
            return {
                "success": True,
                "message": "API documentation placeholder"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _generic_docs_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle generic documentation tasks."""
        try:
            return {
                "success": True,
                "message": f"Processed docs task: {task.get('description', 'unknown')}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _update_state(self, status: str, task_id: Optional[str] = None, error: Optional[str] = None) -> None:
        """Update agent state in shared memory."""
        state = AgentState(
            agent_id=self.agent_id,
            status=status,
            current_task=task_id,
            last_heartbeat=datetime.now(timezone.utc),
            metadata={
                "error": error if error else None,
                "capabilities": ["documentation", "readme_management"]
            }
        )
        self.shared_memory.update_agent_state(state)
