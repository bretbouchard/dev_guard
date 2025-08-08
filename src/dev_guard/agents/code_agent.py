"""Code Agent - Uses Goose CLI (Block) to generate/refactor/modify code."""

import ast
import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid
import subprocess

from .base_agent import BaseAgent
from ..memory.shared_memory import MemoryEntry

logger = logging.getLogger(__name__)

class CodeAgent(BaseAgent):
    """Code Agent uses Goose CLI (Block) to generate/refactor/modify code."""

    def __init__(
        self, 
        agent_id: str, 
        config,  # Config type causes circular import
        shared_memory,  # SharedMemory type
        vector_db,  # VectorDatabase type
        working_directory: Optional[str] = None
    ):
        """Initialize the Code Agent with Goose CLI configuration."""
        super().__init__(agent_id, config, shared_memory, vector_db)
        self.goose_path = self._find_goose_cli()
        self.session_id = None
        self.working_directory = working_directory or os.getcwd()
        
    def _find_goose_cli(self) -> str:
        """Find the Goose CLI executable in the system PATH."""
        # Check common installation paths
        paths_to_check = [
            os.path.expanduser("~/.local/bin/goose"),
            "/usr/local/bin/goose"
        ]
        
        for path in paths_to_check:
            if os.path.exists(path):
                return path
        
        # Check if goose is in PATH by trying to find it
        import shutil
        goose_in_path = shutil.which("goose")
        if goose_in_path:
            return "goose"
                
        raise RuntimeError("Goose CLI not found. Please install Goose CLI first.")

    async def execute(self, state: Any) -> Any:
        """Execute code generation/modification tasks using Goose CLI."""
        self.set_status("busy", str(uuid.uuid4()))
        self.update_heartbeat()
        
        try:
            # Extract task from state
            if isinstance(state, dict):
                task = state
            else:
                # Parse state for task information
                task = self._parse_state_for_task(state)
                
            if not task:
                self.logger.warning("No valid code task found in state")
                self.set_status("idle")
                return state
                
            # Execute the code task using Goose
            result = await self._execute_code_task(task)
            
            # Log the result with Goose metadata
            await self._log_goose_result(task, result)
            
            self.set_status("idle")
            return {
                "success": result.get("success", True),
                "task": task,
                "result": result,
                "agent": "code_agent"
            }
            
        except Exception as e:
            self.logger.error(f"Error in code agent execution: {e}")
            self.set_status("error")
            return {
                "success": False,
                "error": str(e),
                "agent": "code_agent"
            }

    async def _execute_code_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific code task using appropriate Goose commands."""
        task_type = task.get("type", "generate")
        
        if task_type == "generate":
            return await self.generate_code(
                task.get("prompt", ""),
                task.get("file_path"),
                task.get("context", {})
            )
        elif task_type == "fix":
            return await self.fix_code(
                task.get("file_path", ""),
                task.get("error_description", ""),
                task.get("context", {})
            )
        elif task_type == "test":
            return await self.write_tests(
                task.get("file_path", ""),
                task.get("context", {})
            )
        elif task_type == "refactor":
            return await self.refactor_code(
                task.get("file_path", ""),
                task.get("refactor_description", ""),
                task.get("context", {})
            )
        else:
            # Generic code task
            return await self.run_goose_session(
                task.get("prompt", ""),
                task.get("file_path"),
                task.get("goose_args", [])
            )

    async def generate_code(
        self,
        prompt: str,
        file_path: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate code using Goose CLI with automatic quality checks and pattern matching."""
        try:
            # Step 1: Search for similar patterns to inform code generation
            pattern_results = await self.search_similar_patterns(prompt, file_path)
            
            # Step 2: Enhance prompt with pattern context if available
            enhanced_prompt = prompt
            if pattern_results.get("success") and pattern_results.get("recommended_patterns"):
                pattern_context = self._build_pattern_context(pattern_results["recommended_patterns"])
                enhanced_prompt = f"{prompt}\n\nSimilar patterns found:\n{pattern_context}"
                logger.info(f"Enhanced prompt with {len(pattern_results['recommended_patterns'])} patterns")
            
            args = ["session", "start"]
            
            if file_path:
                args.extend(["--file", file_path])
                
            # Start a Goose session for code generation with enhanced prompt
            session_result = await self._run_goose_command(args, input_text=enhanced_prompt)
            
            result = {
                "success": session_result["success"],
                "generated_code": session_result.get("output", ""),
                "file_path": file_path,
                "session_id": self.session_id,
                "goose_output": session_result,
                "pattern_analysis": pattern_results
            }
            
            # Step 3: Apply quality checks and formatting if code was generated successfully
            if session_result["success"] and file_path and Path(file_path).exists():
                quality_result = await self.quality_check_and_format(file_path, auto_fix=True)
                result["quality_check"] = quality_result
                result["formatting_applied"] = quality_result.get("formatting_applied", [])
                result["issues_found"] = quality_result.get("issues_found", [])
                result["auto_fixes_applied"] = quality_result.get("auto_fixes_applied", [])
                
                # Step 4: Analyze the generated code structure for future reference
                if Path(file_path).suffix == ".py":
                    structure_analysis = await self.analyze_code_structure(file_path)
                    result["structure_analysis"] = structure_analysis
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def fix_code(
        self,
        file_path: str,
        error_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Fix code issues using Goose CLI with automatic quality checks."""
        fix_prompt = f"Fix the following issue in {file_path}: {error_description}"
        
        args = ["session", "start", "--file", file_path]
        session_result = await self._run_goose_command(args, input_text=fix_prompt)
        
        result = {
            "success": session_result["success"],
            "file_path": file_path,
            "fix_description": error_description,
            "session_id": self.session_id,
            "goose_output": session_result
        }
        
        # Apply quality checks and formatting after fixing
        if session_result["success"] and Path(file_path).exists():
            quality_result = await self.quality_check_and_format(file_path, auto_fix=True)
            result["quality_check"] = quality_result
            result["formatting_applied"] = quality_result.get("formatting_applied", [])
            result["issues_found"] = quality_result.get("issues_found", [])
            result["auto_fixes_applied"] = quality_result.get("auto_fixes_applied", [])
        
        return result

    async def write_tests(
        self,
        file_path: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate tests for code using Goose CLI with quality checks."""
        test_prompt = f"Write comprehensive tests for the code in {file_path}"
        
        args = ["session", "start", "--file", file_path]
        session_result = await self._run_goose_command(args, input_text=test_prompt)
        
        result = {
            "success": session_result["success"],
            "source_file": file_path,
            "test_file": session_result.get("test_file"),
            "session_id": self.session_id,
            "goose_output": session_result
        }
        
        # Apply quality checks to generated test file if available
        test_file = session_result.get("test_file")
        if session_result["success"] and test_file and Path(test_file).exists():
            quality_result = await self.quality_check_and_format(test_file, auto_fix=True)
            result["test_quality_check"] = quality_result
        
        return result

    async def refactor_code(
        self,
        file_path: str,
        refactor_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Refactor code using Goose CLI with AST analysis and quality checks."""
        try:
            # Step 1: Analyze current code structure
            structure_before = await self.analyze_code_structure(file_path)
            
            # Step 2: Search for similar refactoring patterns
            refactor_query = f"refactor {refactor_description} similar to code structure"
            pattern_results = await self.search_similar_patterns(refactor_query, file_path)
            
            # Step 3: Build enhanced refactoring prompt
            enhanced_prompt = f"Refactor the code in {file_path}: {refactor_description}"
            if pattern_results.get("success") and pattern_results.get("recommended_patterns"):
                pattern_context = self._build_pattern_context(pattern_results["recommended_patterns"])
                enhanced_prompt += f"\n\nSimilar refactoring patterns:\n{pattern_context}"
            
            # Step 4: Perform refactoring with Goose
            args = ["session", "start", "--file", file_path]
            session_result = await self._run_goose_command(args, input_text=enhanced_prompt)
            
            result = {
                "success": session_result["success"],
                "file_path": file_path,
                "refactor_description": refactor_description,
                "session_id": self.session_id,
                "goose_output": session_result,
                "structure_before": structure_before,
                "pattern_analysis": pattern_results
            }
            
            # Step 5: Apply quality checks after refactoring
            if session_result["success"] and Path(file_path).exists():
                quality_result = await self.quality_check_and_format(file_path, auto_fix=True)
                result["quality_check"] = quality_result
                result["formatting_applied"] = quality_result.get("formatting_applied", [])
                result["issues_found"] = quality_result.get("issues_found", [])
                result["auto_fixes_applied"] = quality_result.get("auto_fixes_applied", [])
                
                # Step 6: Analyze structure after refactoring to measure improvement
                structure_after = await self.analyze_code_structure(file_path)
                result["structure_after"] = structure_after
                result["refactoring_impact"] = self._analyze_refactoring_impact(structure_before, structure_after)
            
            return result
            
        except Exception as e:
            logger.error(f"Error refactoring code: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path,
                "refactor_description": refactor_description
            }

    async def run_goose_session(
        self,
        prompt: str,
        file_path: Optional[str] = None,
        goose_args: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run a generic Goose CLI session."""
        args = ["session", "start"]
        
        if goose_args:
            args.extend(goose_args)
            
        if file_path:
            args.extend(["--file", file_path])
            
        session_result = await self._run_goose_command(args, input_text=prompt)
        
        return {
            "success": session_result["success"],
            "prompt": prompt,
            "file_path": file_path,
            "session_id": self.session_id,
            "goose_output": session_result
        }

    async def _run_goose_command(
        self,
        args: List[str],
        input_text: Optional[str] = None,
        cwd: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a Goose CLI command with proper error handling and enhanced metadata capture."""
        try:
            # Prepare the full command
            cmd = [self.goose_path] + args
            working_dir = cwd or self.working_directory
            
            # Capture start time for performance tracking
            start_time = datetime.now(timezone.utc)
            
            # Execute the command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir
            )
            
            # Send input and get output
            stdout, stderr = await process.communicate(
                input=input_text.encode() if input_text else None
            )
            
            end_time = datetime.now(timezone.utc)
            execution_duration = (end_time - start_time).total_seconds()
            
            # Generate session ID if not exists
            if not self.session_id:
                self.session_id = str(uuid.uuid4())
            
            # Enhanced result format aligned with Goose tool call export format
            result = {
                "success": process.returncode == 0,
                "output": stdout.decode(),
                "error": stderr.decode() if stderr else "",
                "return_code": process.returncode,
                "command": " ".join(cmd),
                "session_id": self.session_id,
                # Enhanced metadata for tool call compatibility
                "tool_call": {
                    "type": "goose_cli",
                    "function": "session",
                    "arguments": {
                        "command": args,
                        "input_text": input_text,
                        "working_directory": working_dir
                    },
                    "timestamp": start_time.isoformat(),
                    "duration_seconds": execution_duration,
                    "metadata": {
                        "agent_id": self.agent_id,
                        "session_id": self.session_id,
                        "working_directory": working_dir,
                        "command_line": " ".join(cmd),
                        "exit_code": process.returncode,
                        "output_format": "text"
                    }
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error running Goose command: {e}")
            end_time = datetime.now(timezone.utc) if 'start_time' in locals() else datetime.now(timezone.utc)
            start_time = locals().get('start_time', end_time)
            execution_duration = (end_time - start_time).total_seconds()
            
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "return_code": -1,
                "command": " ".join([self.goose_path] + args),
                "session_id": self.session_id,
                "tool_call": {
                    "type": "goose_cli",
                    "function": "session",
                    "arguments": {
                        "command": args,
                        "input_text": input_text,
                        "working_directory": cwd or self.working_directory
                    },
                    "timestamp": start_time.isoformat(),
                    "duration_seconds": execution_duration,
                    "metadata": {
                        "agent_id": self.agent_id,
                        "session_id": self.session_id,
                        "working_directory": cwd or self.working_directory,
                        "command_line": " ".join([self.goose_path] + args),
                        "exit_code": -1,
                        "output_format": "text",
                        "error": str(e)
                    }
                }
            }

    def _parse_state_for_task(self, state: Any) -> Optional[Dict[str, Any]]:
        """Parse state to extract code task information."""
        if isinstance(state, str):
            return {
                "type": "generate",
                "prompt": state,
                "file_path": None
            }
        elif hasattr(state, 'get'):
            return {
                "type": state.get("type", "generate"),
                "prompt": state.get("description", ""),
                "file_path": state.get("file_path"),
                "context": state.get("context", {})
            }
        else:
            return None

    async def _log_goose_result(self, task: Dict[str, Any], result: Dict[str, Any]):
        """Log Goose execution results to shared memory with enhanced tool call format."""
        try:
            # Extract enhanced tool call information
            tool_call_info = result.get("tool_call", {})
            
            # Create enhanced goose patch format aligned with Goose export format
            enhanced_goose_patch = {
                # Core execution data
                "command": result.get("command", ""),
                "session_id": result.get("session_id"),
                "output": result.get("output", ""),
                "error": result.get("error", ""),
                "return_code": result.get("return_code", -1),
                
                # Enhanced tool call metadata for Goose compatibility
                "tool_call": {
                    "type": tool_call_info.get("type", "goose_cli"),
                    "function": tool_call_info.get("function", "session"),
                    "arguments": tool_call_info.get("arguments", {}),
                    "timestamp": tool_call_info.get("timestamp"),
                    "duration_seconds": tool_call_info.get("duration_seconds", 0),
                    "metadata": tool_call_info.get("metadata", {})
                },
                
                # Additional DevGuard-specific metadata
                "devguard_metadata": {
                    "task_type": task.get("type", "unknown"),
                    "agent_id": self.agent_id,
                    "working_directory": tool_call_info.get("metadata", {}).get("working_directory"),
                    "file_path": task.get("file_path"),
                    "execution_context": {
                        "prompt_used": task.get("prompt"),
                        "task_description": task.get("description", ""),
                        "quality_checks_applied": result.get("quality_checks_applied", [])
                    }
                },
                
                # Markdown export compatibility
                "markdown_export": {
                    "format_version": "1.0",
                    "exportable": True,
                    "session_name": f"devguard-{self.agent_id}-{result.get('session_id', 'unknown')[:8]}",
                    "summary": f"DevGuard {task.get('type', 'code')} operation on {task.get('file_path', 'unknown file')}"
                }
            }
            
            # Create memory entry with enhanced Goose patch
            memory_entry = MemoryEntry(
                agent_id=self.agent_id,
                type="result",
                content={
                    "task": task,
                    "result": result,
                    "success": result.get("success", False)
                },
                tags={"goose", "code_generation", task.get("type", "unknown"), "enhanced_format"},
                parent_id=None,  # Optional root-level memory entry
                goose_patch=enhanced_goose_patch,
                ast_summary=None,  # Will be populated by AST analysis if applicable
                goose_strategy=task.get("type", "generate"),
                file_path=task.get("file_path")
            )
            
            self.shared_memory.add_memory(memory_entry)
            
        except Exception as e:
            self.logger.error(f"Error logging enhanced Goose result: {e}")

    async def analyze_code_structure(self, file_path: str) -> Dict[str, Any]:
        """Analyze the AST structure of a Python file."""
        try:
            if not Path(file_path).exists():
                return {
                    "success": False,
                    "error": f"File does not exist: {file_path}"
                }
            
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            try:
                tree = ast.parse(source_code)
            except SyntaxError as e:
                return {
                    "success": False,
                    "error": f"Syntax error in file: {e}"
                }
            
            structure = {
                "success": True,
                "file_path": file_path,
                "classes": [],
                "functions": [],
                "imports": [],
                "complexity_metrics": {},
                "ast_nodes": []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    structure["classes"].append({
                        "name": node.name,
                        "line": node.lineno,
                        "methods": [method.name for method in node.body if isinstance(method, ast.FunctionDef)],
                        "decorators": [self._extract_decorator_name(dec) for dec in node.decorator_list]
                    })
                elif isinstance(node, ast.FunctionDef):
                    structure["functions"].append({
                        "name": node.name,
                        "line": node.lineno,
                        "args": [arg.arg for arg in node.args.args],
                        "decorators": [self._extract_decorator_name(dec) for dec in node.decorator_list],
                        "returns": self._extract_return_annotation(node)
                    })
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    structure["imports"].append(self._extract_import_info(node))
                
                # Collect node types for complexity analysis
                node_type = type(node).__name__
                structure["ast_nodes"].append(node_type)
            
            # Calculate complexity metrics
            structure["complexity_metrics"] = {
                "total_nodes": len(structure["ast_nodes"]),
                "classes_count": len(structure["classes"]),
                "functions_count": len(structure["functions"]),
                "imports_count": len(structure["imports"]),
                "cyclomatic_complexity": self._calculate_cyclomatic_complexity(tree)
            }
            
            return structure
            
        except Exception as e:
            logger.error(f"Error analyzing code structure: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _extract_decorator_name(self, decorator) -> str:
        """Extract decorator name from AST node."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return f"{self._extract_attr_name(decorator.value)}.{decorator.attr}"
        else:
            return "unknown"
    
    def _extract_attr_name(self, node) -> str:
        """Extract attribute name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._extract_attr_name(node.value)}.{node.attr}"
        else:
            return "unknown"
    
    def _extract_return_annotation(self, func_node) -> Optional[str]:
        """Extract return type annotation from function node."""
        if func_node.returns:
            if isinstance(func_node.returns, ast.Name):
                return func_node.returns.id
            elif isinstance(func_node.returns, ast.Attribute):
                return f"{self._extract_attr_name(func_node.returns.value)}.{func_node.returns.attr}"
            else:
                return "complex_annotation"
        return None
    
    def _extract_import_info(self, import_node) -> Dict[str, str]:
        """Extract import information from AST node."""
        if isinstance(import_node, ast.Import):
            return {
                "type": "import",
                "module": import_node.names[0].name,
                "alias": import_node.names[0].asname
            }
        elif isinstance(import_node, ast.ImportFrom):
            return {
                "type": "from_import",
                "module": import_node.module or "",
                "names": [alias.name for alias in import_node.names],
                "level": import_node.level
            }
        return {}
    
    def _calculate_cyclomatic_complexity(self, tree) -> int:
        """Calculate cyclomatic complexity of the code."""
        complexity = 1  # Base complexity
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.With)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        return complexity

    async def search_similar_patterns(self, query: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Search for similar code patterns using Goose memory and AST analysis."""
        try:
            results = {
                "success": True,
                "query": query,
                "goose_matches": [],
                "vector_matches": [],
                "ast_matches": [],
                "recommended_patterns": []
            }
            
            # Step 1: Try Goose memory search first (preferred method)
            goose_results = await self._search_goose_memory(query, file_path)
            if goose_results.get("success") and goose_results.get("matches"):
                results["goose_matches"] = goose_results["matches"]
                logger.info(f"Found {len(results['goose_matches'])} matches in Goose memory")
            
            # Step 2: If Goose memory doesn't provide enough results, use vector search
            if len(results["goose_matches"]) < 3:
                vector_results = await self._search_vector_db(query, file_path)
                if vector_results.get("success"):
                    results["vector_matches"] = vector_results.get("matches", [])
                    logger.info(f"Found {len(results['vector_matches'])} matches in vector DB")
            
            # Step 3: If we have a target file, find AST-similar patterns
            if file_path and Path(file_path).exists():
                ast_results = await self._search_ast_similar(file_path, query)
                if ast_results.get("success"):
                    results["ast_matches"] = ast_results.get("matches", [])
            
            # Step 4: Generate recommendations based on all matches
            results["recommended_patterns"] = self._generate_pattern_recommendations(
                results["goose_matches"],
                results["vector_matches"], 
                results["ast_matches"]
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar patterns: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _search_goose_memory(self, query: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Search Goose memory for similar patterns using Goose CLI."""
        try:
            # Use Goose CLI to search memory with a lookup command
            cmd = ["session", "start", "--prompt", f"Search memory for patterns similar to: {query}"]
            if file_path:
                cmd.extend(["--file", file_path])
            
            result = await self._run_goose_command(cmd)
            
            if result.get("success"):
                # Parse Goose output for memory matches
                matches = self._parse_goose_memory_response(result.get("output", ""))
                return {
                    "success": True,
                    "matches": matches,
                    "raw_output": result.get("output", "")
                }
            else:
                logger.warning("Goose memory search failed, will fallback to vector DB")
                return {"success": False, "error": result.get("error", "Unknown error")}
                
        except Exception as e:
            logger.error(f"Error in Goose memory search: {e}")
            return {"success": False, "error": str(e)}

    async def _search_vector_db(self, query: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Search vector database for similar code patterns."""
        try:
            # Use the vector database to find similar code
            search_results = await asyncio.to_thread(
                self.vector_db.search,
                query=query,
                n_results=5,
                filter_metadata={"file_type": "python"} if not file_path else None
            )
            
            matches = []
            for doc in search_results.get("documents", []):
                matches.append({
                    "content": doc.get("content", ""),
                    "metadata": doc.get("metadata", {}),
                    "similarity_score": doc.get("score", 0.0),
                    "source": "vector_db"
                })
            
            return {
                "success": True,
                "matches": matches
            }
            
        except Exception as e:
            logger.error(f"Error in vector DB search: {e}")
            return {"success": False, "error": str(e)}

    async def _search_ast_similar(self, file_path: str, query: str) -> Dict[str, Any]:
        """Find AST-structurally similar code patterns."""
        try:
            # Analyze the target file structure
            target_structure = await self.analyze_code_structure(file_path)
            if not target_structure.get("success"):
                return {"success": False, "error": "Could not analyze target file structure"}
            
            # Search for files with similar structural patterns
            matches = []
            
            # Get similar files from vector DB based on structural similarity
            structure_query = self._build_structure_query(target_structure)
            vector_results = await self._search_vector_db(structure_query, file_path)
            
            if vector_results.get("success"):
                for match in vector_results["matches"]:
                    file_match_path = match["metadata"].get("file_path")
                    if file_match_path and Path(file_match_path).exists():
                        match_structure = await self.analyze_code_structure(file_match_path)
                        if match_structure.get("success"):
                            similarity_score = self._calculate_structural_similarity(
                                target_structure, match_structure
                            )
                            
                            if similarity_score > 0.3:  # Threshold for similarity
                                matches.append({
                                    "file_path": file_match_path,
                                    "structure": match_structure,
                                    "similarity_score": similarity_score,
                                    "source": "ast_analysis"
                                })
            
            return {
                "success": True,
                "matches": sorted(matches, key=lambda x: x["similarity_score"], reverse=True)
            }
            
        except Exception as e:
            logger.error(f"Error in AST similarity search: {e}")
            return {"success": False, "error": str(e)}

    def _parse_goose_memory_response(self, output: str) -> List[Dict[str, Any]]:
        """Parse Goose CLI memory search response."""
        matches = []
        try:
            # Look for memory entries in Goose output
            lines = output.split('\n')
            current_match = {}
            
            for line in lines:
                line = line.strip()
                if line.startswith("Memory entry:") or line.startswith("Found pattern:"):
                    if current_match:
                        matches.append(current_match)
                        current_match = {}
                elif line.startswith("Code:"):
                    current_match["code"] = line.replace("Code:", "").strip()
                elif line.startswith("Context:"):
                    current_match["context"] = line.replace("Context:", "").strip()
                elif line.startswith("Confidence:"):
                    try:
                        current_match["confidence"] = float(line.replace("Confidence:", "").strip())
                    except:
                        current_match["confidence"] = 0.5
            
            if current_match:
                matches.append(current_match)
                
        except Exception as e:
            logger.error(f"Error parsing Goose memory response: {e}")
        
        return matches

    def _build_structure_query(self, structure: Dict[str, Any]) -> str:
        """Build a search query based on code structure."""
        query_parts = []
        
        # Add class information
        for cls in structure.get("classes", []):
            query_parts.append(f"class {cls['name']}")
            for method in cls["methods"]:
                query_parts.append(f"def {method}")
        
        # Add function information
        for func in structure.get("functions", []):
            query_parts.append(f"def {func['name']}")
        
        # Add import information
        for imp in structure.get("imports", []):
            if imp.get("type") == "import":
                query_parts.append(f"import {imp['module']}")
            else:
                query_parts.append(f"from {imp.get('module', '')} import")
        
        return " ".join(query_parts[:10])  # Limit query size

    def _calculate_structural_similarity(self, struct1: Dict[str, Any], struct2: Dict[str, Any]) -> float:
        """Calculate structural similarity between two code structures."""
        try:
            score = 0.0
            total_weight = 0.0
            
            # Class similarity
            classes1 = {c["name"] for c in struct1.get("classes", [])}
            classes2 = {c["name"] for c in struct2.get("classes", [])}
            if classes1 or classes2:
                class_similarity = len(classes1 & classes2) / max(len(classes1 | classes2), 1)
                score += class_similarity * 0.3
                total_weight += 0.3
            
            # Function similarity
            funcs1 = {f["name"] for f in struct1.get("functions", [])}
            funcs2 = {f["name"] for f in struct2.get("functions", [])}
            if funcs1 or funcs2:
                func_similarity = len(funcs1 & funcs2) / max(len(funcs1 | funcs2), 1)
                score += func_similarity * 0.4
                total_weight += 0.4
            
            # Import similarity
            imports1 = {i.get("module", "") for i in struct1.get("imports", [])}
            imports2 = {i.get("module", "") for i in struct2.get("imports", [])}
            if imports1 or imports2:
                import_similarity = len(imports1 & imports2) / max(len(imports1 | imports2), 1)
                score += import_similarity * 0.2
                total_weight += 0.2
            
            # Complexity similarity
            comp1 = struct1.get("complexity_metrics", {})
            comp2 = struct2.get("complexity_metrics", {})
            if comp1 and comp2:
                complexity_diff = abs(comp1.get("cyclomatic_complexity", 0) - comp2.get("cyclomatic_complexity", 0))
                complexity_similarity = max(0, 1 - complexity_diff / 10)
                score += complexity_similarity * 0.1
                total_weight += 0.1
            
            return score / max(total_weight, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating structural similarity: {e}")
            return 0.0

    def _generate_pattern_recommendations(self, goose_matches: List[Dict], vector_matches: List[Dict], ast_matches: List[Dict]) -> List[Dict[str, Any]]:
        """Generate recommended patterns based on all search results."""
        recommendations = []
        
        try:
            # Process Goose memory matches (highest priority)
            for match in goose_matches[:3]:
                recommendations.append({
                    "source": "goose_memory",
                    "confidence": match.get("confidence", 0.8),
                    "pattern": match.get("code", ""),
                    "context": match.get("context", ""),
                    "reason": "Historical pattern from Goose memory"
                })
            
            # Process vector matches
            for match in vector_matches[:2]:
                if match.get("similarity_score", 0) > 0.7:
                    recommendations.append({
                        "source": "vector_search",
                        "confidence": match.get("similarity_score", 0.7),
                        "pattern": match.get("content", "")[:200] + "...",
                        "context": match.get("metadata", {}),
                        "reason": "Similar code from codebase"
                    })
            
            # Process AST matches
            for match in ast_matches[:2]:
                if match.get("similarity_score", 0) > 0.5:
                    recommendations.append({
                        "source": "ast_analysis",
                        "confidence": match.get("similarity_score", 0.5),
                        "pattern": f"Structurally similar file: {match.get('file_path', '')}",
                        "context": match.get("structure", {}),
                        "reason": "Structurally similar code pattern"
                    })
            
            # Sort by confidence
            recommendations.sort(key=lambda x: x["confidence"], reverse=True)
            return recommendations[:5]  # Return top 5
            
        except Exception as e:
            logger.error(f"Error generating pattern recommendations: {e}")
            return []

    def _build_pattern_context(self, patterns: List[Dict[str, Any]]) -> str:
        """Build context string from recommended patterns."""
        if not patterns:
            return ""
        
        context_parts = []
        for i, pattern in enumerate(patterns[:3], 1):  # Use top 3 patterns
            source = pattern.get("source", "unknown")
            confidence = pattern.get("confidence", 0.0)
            pattern_text = pattern.get("pattern", "")
            reason = pattern.get("reason", "")
            
            context_parts.append(f"""
Pattern {i} (from {source}, confidence: {confidence:.2f}):
{pattern_text}
Reason: {reason}
""")
        
        return "\n".join(context_parts)

    def _analyze_refactoring_impact(self, structure_before: Dict[str, Any], structure_after: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the impact of refactoring by comparing code structures."""
        if not structure_before.get("success") or not structure_after.get("success"):
            return {"success": False, "error": "Could not analyze structures"}
        
        try:
            before_metrics = structure_before.get("complexity_metrics", {})
            after_metrics = structure_after.get("complexity_metrics", {})
            
            impact = {
                "success": True,
                "complexity_change": {
                    "before": before_metrics.get("cyclomatic_complexity", 0),
                    "after": after_metrics.get("cyclomatic_complexity", 0),
                    "improvement": before_metrics.get("cyclomatic_complexity", 0) - after_metrics.get("cyclomatic_complexity", 0)
                },
                "structure_changes": {
                    "classes_change": after_metrics.get("classes_count", 0) - before_metrics.get("classes_count", 0),
                    "functions_change": after_metrics.get("functions_count", 0) - before_metrics.get("functions_count", 0),
                    "total_nodes_change": after_metrics.get("total_nodes", 0) - before_metrics.get("total_nodes", 0)
                },
                "quality_assessment": "improved" if after_metrics.get("cyclomatic_complexity", 0) < before_metrics.get("cyclomatic_complexity", 0) else "unchanged"
            }
            
            # Additional analysis
            if impact["complexity_change"]["improvement"] > 0:
                impact["quality_assessment"] = "improved"
            elif impact["complexity_change"]["improvement"] < -2:  # Complexity increased significantly
                impact["quality_assessment"] = "degraded"
            
            return impact
            
        except Exception as e:
            logger.error(f"Error analyzing refactoring impact: {e}")
            return {"success": False, "error": str(e)}

    async def format_code(self, file_path: str) -> Dict[str, Any]:
        """Format code using black and isort."""
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                return {
                    "success": False,
                    "error": f"File does not exist: {file_path}"
                }

            results = {
                "success": True,
                "formatters_applied": [],
                "errors": []
            }

            # Apply black formatting
            black_result = await self._run_black(file_path)
            if black_result["success"]:
                results["formatters_applied"].append("black")
            else:
                results["errors"].append(f"Black error: {black_result.get('error', 'Unknown error')}")

            # Apply isort import sorting
            isort_result = await self._run_isort(file_path)
            if isort_result["success"]:
                results["formatters_applied"].append("isort")
            else:
                results["errors"].append(f"isort error: {isort_result.get('error', 'Unknown error')}")

            # Update success status
            results["success"] = len(results["formatters_applied"]) > 0

            return results

        except Exception as e:
            logger.error(f"Error formatting code: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def lint_code(self, file_path: str) -> Dict[str, Any]:
        """Lint code using ruff and mypy."""
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                return {
                    "success": False,
                    "error": f"File does not exist: {file_path}"
                }

            results = {
                "success": True,
                "linters_run": [],
                "issues": [],
                "errors": []
            }

            # Run ruff linting
            ruff_result = await self._run_ruff(file_path)
            results["linters_run"].append("ruff")
            if ruff_result["success"]:
                results["issues"].extend(ruff_result.get("issues", []))
            else:
                results["errors"].append(f"Ruff error: {ruff_result.get('error', 'Unknown error')}")

            # Run mypy type checking
            mypy_result = await self._run_mypy(file_path)
            results["linters_run"].append("mypy")
            if mypy_result["success"]:
                results["issues"].extend(mypy_result.get("issues", []))
            else:
                results["errors"].append(f"Mypy error: {mypy_result.get('error', 'Unknown error')}")

            return results

        except Exception as e:
            logger.error(f"Error linting code: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def quality_check_and_format(self, file_path: str, auto_fix: bool = True) -> Dict[str, Any]:
        """Run complete quality check and formatting pipeline."""
        try:
            results = {
                "success": True,
                "file_path": file_path,
                "steps_completed": [],
                "issues_found": [],
                "formatting_applied": [],
                "auto_fixes_applied": [],
                "errors": []
            }

            # Step 1: Format code
            format_result = await self.format_code(file_path)
            if format_result["success"]:
                results["steps_completed"].append("formatting")
                results["formatting_applied"] = format_result.get("formatters_applied", [])
            else:
                results["errors"].extend(format_result.get("errors", []))

            # Step 2: Run linting
            lint_result = await self.lint_code(file_path)
            if lint_result["success"]:
                results["steps_completed"].append("linting")
                results["issues_found"] = lint_result.get("issues", [])
            else:
                results["errors"].extend(lint_result.get("errors", []))

            # Step 3: Auto-fix if requested and fixable issues exist
            if auto_fix and results["issues_found"]:
                fix_result = await self._auto_fix_issues(file_path, results["issues_found"])
                if fix_result["success"]:
                    results["steps_completed"].append("auto_fix")
                    results["auto_fixes_applied"] = fix_result.get("fixes_applied", [])

            # Update overall success
            results["success"] = len(results["errors"]) == 0

            return results

        except Exception as e:
            logger.error(f"Error in quality check and format pipeline: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _run_black(self, file_path: str) -> Dict[str, Any]:
        """Run black formatter on a file."""
        try:
            cmd = ["python", "-m", "black", file_path]
            result = await self._run_command_simple(cmd)
            
            return {
                "success": result["returncode"] == 0,
                "output": result.get("stdout", ""),
                "error": result.get("stderr", "") if result["returncode"] != 0 else None
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _run_isort(self, file_path: str) -> Dict[str, Any]:
        """Run isort import sorting on a file."""
        try:
            cmd = ["python", "-m", "isort", file_path]
            result = await self._run_command_simple(cmd)
            
            return {
                "success": result["returncode"] == 0,
                "output": result.get("stdout", ""),
                "error": result.get("stderr", "") if result["returncode"] != 0 else None
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _run_ruff(self, file_path: str) -> Dict[str, Any]:
        """Run ruff linting on a file."""
        try:
            cmd = ["python", "-m", "ruff", "check", file_path, "--output-format=json"]
            result = await self._run_command_simple(cmd)
            
            issues = []
            if result.get("stdout"):
                try:
                    ruff_output = json.loads(result["stdout"])
                    for issue in ruff_output:
                        issues.append({
                            "type": "ruff",
                            "file": issue.get("filename", file_path),
                            "line": issue.get("location", {}).get("row", 0),
                            "column": issue.get("location", {}).get("column", 0),
                            "code": issue.get("code", ""),
                            "message": issue.get("message", ""),
                            "severity": "error" if issue.get("code", "").startswith("E") else "warning"
                        })
                except json.JSONDecodeError:
                    pass
            
            return {
                "success": True,
                "issues": issues,
                "output": result.get("stdout", "")
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _run_mypy(self, file_path: str) -> Dict[str, Any]:
        """Run mypy type checking on a file."""
        try:
            cmd = ["python", "-m", "mypy", file_path, "--show-column-numbers", "--show-error-codes"]
            result = await self._run_command_simple(cmd)
            
            issues = []
            if result.get("stdout"):
                for line in result["stdout"].split("\n"):
                    if ":" in line and "error:" in line:
                        parts = line.split(":")
                        if len(parts) >= 4:
                            issues.append({
                                "type": "mypy",
                                "file": parts[0],
                                "line": int(parts[1]) if parts[1].isdigit() else 0,
                                "column": int(parts[2]) if parts[2].isdigit() else 0,
                                "message": ":".join(parts[3:]).strip(),
                                "severity": "error"
                            })
            
            return {
                "success": True,
                "issues": issues,
                "output": result.get("stdout", "")
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _auto_fix_issues(self, file_path: str, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Attempt to auto-fix common code issues."""
        try:
            fixes_applied = []
            
            # Check if ruff can auto-fix any issues
            ruff_fixable = [issue for issue in issues if issue.get("type") == "ruff"]
            if ruff_fixable:
                cmd = ["python", "-m", "ruff", "check", file_path, "--fix"]
                result = await self._run_command_simple(cmd)
                if result["returncode"] == 0:
                    fixes_applied.append("ruff_auto_fix")

            # Re-run formatters to clean up after fixes
            black_result = await self._run_black(file_path)
            if black_result["success"]:
                fixes_applied.append("black_reformat")
            
            isort_result = await self._run_isort(file_path)
            if isort_result["success"]:
                fixes_applied.append("isort_reformat")

            return {
                "success": len(fixes_applied) > 0,
                "fixes_applied": fixes_applied
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _run_command_simple(self, cmd: List[str]) -> Dict[str, Any]:
        """Run a simple command and return result."""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_directory
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                "returncode": process.returncode,
                "stdout": stdout.decode('utf-8') if stdout else "",
                "stderr": stderr.decode('utf-8') if stderr else ""
            }
        except Exception as e:
            return {
                "returncode": 1,
                "stdout": "",
                "stderr": str(e)
            }

    def get_capabilities(self) -> List[str]:
        """Return the capabilities of the Code Agent."""
        return [
            "code_generation",
            "code_fixing",
            "test_writing", 
            "code_refactoring",
            "goose_cli_integration",
            "code_formatting",
            "code_linting",
            "quality_checking",
            "auto_fixing",
            "ast_analysis",
            "pattern_matching",
            "goose_memory_search",
            "structural_similarity",
            "refactoring_impact_analysis"
        ]
