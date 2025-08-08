"""
Custom assertion helpers for DevGuard testing.
Provides domain-specific assertions for agents, memory, and vector operations.
"""

from datetime import datetime
from typing import Any


def assert_memory_entry_valid(entry: dict[str, Any]) -> None:
    """Assert that a memory entry has all required fields and valid structure"""
    required_fields = ["id", "agent_id", "timestamp", "type", "content"]
    
    for field in required_fields:
        assert field in entry, f"Memory entry missing required field: {field}"
    
    assert isinstance(entry["id"], str), "Memory entry ID must be string"
    assert len(entry["id"]) > 0, "Memory entry ID cannot be empty"
    
    assert isinstance(entry["agent_id"], str), "Agent ID must be string"
    assert len(entry["agent_id"]) > 0, "Agent ID cannot be empty"
    
    assert isinstance(entry["timestamp"], (datetime, str)), "Timestamp must be datetime or string"
    
    valid_types = ["task", "observation", "decision", "result", "error"]
    assert entry["type"] in valid_types, f"Memory type must be one of {valid_types}"
    
    assert isinstance(entry["content"], dict), "Memory content must be dictionary"


def assert_task_status_valid(task: dict[str, Any]) -> None:
    """Assert that a task status has all required fields and valid structure"""
    required_fields = ["id", "agent_id", "status", "description", "created_at", "updated_at"]
    
    for field in required_fields:
        assert field in task, f"Task missing required field: {field}"
    
    valid_statuses = ["pending", "running", "completed", "failed", "cancelled"]
    assert task["status"] in valid_statuses, f"Task status must be one of {valid_statuses}"
    
    assert isinstance(task["description"], str), "Task description must be string"
    assert len(task["description"]) > 0, "Task description cannot be empty"
    
    assert isinstance(task["created_at"], (datetime, str)), "Created timestamp must be datetime or string"
    assert isinstance(task["updated_at"], (datetime, str)), "Updated timestamp must be datetime or string"


def assert_agent_state_valid(state: dict[str, Any]) -> None:
    """Assert that an agent state has all required fields and valid structure"""
    required_fields = ["agent_id", "status", "last_heartbeat"]
    
    for field in required_fields:
        assert field in state, f"Agent state missing required field: {field}"
    
    valid_statuses = ["idle", "busy", "error", "stopped"]
    assert state["status"] in valid_statuses, f"Agent status must be one of {valid_statuses}"
    
    assert isinstance(state["last_heartbeat"], (datetime, str)), "Heartbeat must be datetime or string"


def assert_vector_document_valid(document: dict[str, Any]) -> None:
    """Assert that a vector document has valid structure"""
    required_fields = ["id", "content", "metadata"]
    
    for field in required_fields:
        assert field in document, f"Document missing required field: {field}"
    
    assert isinstance(document["content"], str), "Document content must be string"
    assert len(document["content"]) > 0, "Document content cannot be empty"
    
    assert isinstance(document["metadata"], dict), "Document metadata must be dictionary"


def assert_llm_response_valid(response: dict[str, Any]) -> None:
    """Assert that an LLM response has valid structure"""
    required_fields = ["content"]
    
    for field in required_fields:
        assert field in response, f"LLM response missing required field: {field}"
    
    assert isinstance(response["content"], str), "LLM response content must be string"
    
    if "usage" in response:
        assert isinstance(response["usage"], dict), "Usage must be dictionary"
        if "prompt_tokens" in response["usage"]:
            assert isinstance(response["usage"]["prompt_tokens"], int), "Prompt tokens must be integer"
        if "completion_tokens" in response["usage"]:
            assert isinstance(response["usage"]["completion_tokens"], int), "Completion tokens must be integer"


def assert_code_quality_metrics(metrics: dict[str, Any]) -> None:
    """Assert that code quality metrics have valid structure"""
    if "coverage" in metrics:
        assert isinstance(metrics["coverage"], (int, float)), "Coverage must be numeric"
        assert 0 <= metrics["coverage"] <= 100, "Coverage must be between 0 and 100"
    
    if "complexity" in metrics:
        assert isinstance(metrics["complexity"], (int, float)), "Complexity must be numeric"
        assert metrics["complexity"] >= 0, "Complexity cannot be negative"
    
    if "lines_of_code" in metrics:
        assert isinstance(metrics["lines_of_code"], int), "Lines of code must be integer"
        assert metrics["lines_of_code"] >= 0, "Lines of code cannot be negative"


def assert_security_scan_result(result: dict[str, Any]) -> None:
    """Assert that a security scan result has valid structure"""
    if "vulnerabilities" in result:
        assert isinstance(result["vulnerabilities"], list), "Vulnerabilities must be list"
        
        for vuln in result["vulnerabilities"]:
            assert isinstance(vuln, dict), "Each vulnerability must be dictionary"
            required_vuln_fields = ["type", "severity", "description"]
            
            for field in required_vuln_fields:
                assert field in vuln, f"Vulnerability missing required field: {field}"
            
            valid_severities = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
            assert vuln["severity"] in valid_severities, f"Severity must be one of {valid_severities}"


def assert_git_operation_result(result: dict[str, Any]) -> None:
    """Assert that a Git operation result has valid structure"""
    if "commit_hash" in result:
        assert isinstance(result["commit_hash"], str), "Commit hash must be string"
        assert len(result["commit_hash"]) >= 7, "Commit hash too short"
    
    if "files_changed" in result:
        assert isinstance(result["files_changed"], list), "Files changed must be list"
        for file_path in result["files_changed"]:
            assert isinstance(file_path, str), "File path must be string"
    
    if "diff" in result:
        assert isinstance(result["diff"], str), "Diff must be string"


def assert_performance_within_bounds(
    actual_time: float,
    expected_max: float,
    operation_name: str = "operation"
) -> None:
    """Assert that an operation completed within expected time bounds"""
    assert actual_time <= expected_max, (
        f"{operation_name} took {actual_time:.3f}s, "
        f"expected <= {expected_max:.3f}s"
    )
    assert actual_time >= 0, f"{operation_name} time cannot be negative"


def assert_memory_usage_reasonable(
    memory_mb: float,
    max_memory_mb: float,
    operation_name: str = "operation"
) -> None:
    """Assert that memory usage is within reasonable bounds"""
    assert memory_mb <= max_memory_mb, (
        f"{operation_name} used {memory_mb:.1f}MB, "
        f"expected <= {max_memory_mb:.1f}MB"
    )
    assert memory_mb >= 0, f"{operation_name} memory usage cannot be negative"


def assert_agent_coordination_valid(coordination_data: dict[str, Any]) -> None:
    """Assert that agent coordination data is valid"""
    required_fields = ["agents", "tasks", "dependencies"]
    
    for field in required_fields:
        assert field in coordination_data, f"Coordination data missing field: {field}"
    
    assert isinstance(coordination_data["agents"], list), "Agents must be list"
    assert isinstance(coordination_data["tasks"], list), "Tasks must be list"
    assert isinstance(coordination_data["dependencies"], dict), "Dependencies must be dictionary"
    
    # Validate that all task dependencies reference valid tasks
    task_ids = {task["id"] for task in coordination_data["tasks"] if "id" in task}
    for task_id, deps in coordination_data["dependencies"].items():
        assert task_id in task_ids, f"Dependency references unknown task: {task_id}"
        for dep in deps:
            assert dep in task_ids, f"Task {task_id} depends on unknown task: {dep}"


def assert_configuration_valid(config: dict[str, Any]) -> None:
    """Assert that configuration has valid structure and values"""
    if "database" in config:
        db_config = config["database"]
        assert isinstance(db_config, dict), "Database config must be dictionary"
        if "url" in db_config:
            assert isinstance(db_config["url"], str), "Database URL must be string"
    
    if "llm" in config:
        llm_config = config["llm"]
        assert isinstance(llm_config, dict), "LLM config must be dictionary"
        if "provider" in llm_config:
            assert isinstance(llm_config["provider"], str), "LLM provider must be string"
        if "temperature" in llm_config:
            temp = llm_config["temperature"]
            assert isinstance(temp, (int, float)), "Temperature must be numeric"
            assert 0 <= temp <= 2, "Temperature must be between 0 and 2"
    
    if "agents" in config:
        agents_config = config["agents"]
        assert isinstance(agents_config, dict), "Agents config must be dictionary"
        if "max_retries" in agents_config:
            retries = agents_config["max_retries"]
            assert isinstance(retries, int), "Max retries must be integer"
            assert retries >= 0, "Max retries cannot be negative"


def assert_mcp_response_valid(response: dict[str, Any]) -> None:
    """Assert that an MCP (Model Context Protocol) response is valid"""
    required_fields = ["jsonrpc", "id"]
    
    for field in required_fields:
        assert field in response, f"MCP response missing required field: {field}"
    
    assert response["jsonrpc"] == "2.0", "MCP response must use JSON-RPC 2.0"
    
    # Either result or error must be present, but not both
    has_result = "result" in response
    has_error = "error" in response
    
    assert has_result or has_error, "MCP response must have either result or error"
    assert not (has_result and has_error), "MCP response cannot have both result and error"
    
    if has_error:
        error = response["error"]
        assert isinstance(error, dict), "MCP error must be dictionary"
        assert "code" in error, "MCP error must have code"
        assert "message" in error, "MCP error must have message"
        assert isinstance(error["code"], int), "MCP error code must be integer"
        assert isinstance(error["message"], str), "MCP error message must be string"