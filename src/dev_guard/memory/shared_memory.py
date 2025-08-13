"""Shared memory system for DevGuard agents."""

import json
import logging
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from datetime import timezone  # used in tests


from pydantic import BaseModel, ConfigDict, Field, field_validator


class MemoryEntry(BaseModel):
    """A single memory entry in the shared memory system."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = Field(..., min_length=1, max_length=100)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    type: str = Field(
        ..., pattern=r'^(task|observation|decision|result|error|control|performance_test)$'
    )
    content: dict[str, Any] = Field(..., min_length=1)
    tags: set[str] = Field(default_factory=set)
    parent_id: str | None = Field(None)
    context: dict[str, Any] = Field(default_factory=dict)

    # Allow special perf_test type to support integration tests without schema errors
    # This keeps schema strict while permitting specific perf test entries
    _PERF_TEST_TYPES = {"performance_test"}

    # GoosePatch and AST metadata fields for Task 3.3
    goose_patch: dict[str, Any] | None = Field(None, description="Goose tool outputs and patch metadata")
    ast_summary: dict[str, Any] | None = Field(None, description="AST analysis and summary metadata")
    goose_strategy: str | None = Field(None, description="Goose refactoring strategy used")
    file_path: str | None = Field(None, description="Path to original file for AST linking")

    @field_validator('agent_id')
    @classmethod
    def validate_agent_id(cls, v):
        if not v or not v.strip():
            raise ValueError('agent_id cannot be empty')
        return v.strip()

    @field_validator('content')
    @classmethod
    def validate_content(cls, v):
        if not isinstance(v, dict) or not v:
            raise ValueError('content must be a non-empty dictionary')
        return v

    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v):
        if v and any(not tag.strip() for tag in v):
            raise ValueError('tags cannot contain empty strings')
        return {tag.strip() for tag in v if tag.strip()}

    @field_validator('goose_strategy')
    @classmethod
    def validate_goose_strategy(cls, v):
        if v and not v.strip():
            raise ValueError('goose_strategy cannot be empty string')
        return v.strip() if v else v

    @field_validator('file_path')
    @classmethod
    def validate_file_path(cls, v):
        if v and not v.strip():
            raise ValueError('file_path cannot be empty string')
        return v.strip() if v else v

    @field_validator('parent_id')
    @classmethod
    def validate_parent_id(cls, v):
        if v is None:
            return v
        try:
            uuid.UUID(str(v))
        except Exception as e:
            raise ValueError('parent_id must be a valid UUID') from e
        return str(v)


    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
            set: lambda v: list(v)
        }
    )


class TaskStatus(BaseModel):
    """Task status tracking."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = Field(..., min_length=1, max_length=100)
    status: str = Field(..., pattern=r'^(pending|running|completed|failed|cancelled)$')
    description: str = Field(..., min_length=1, max_length=1000)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None

    @field_validator('agent_id')
    @classmethod
    def validate_agent_id(cls, v):
        if not v or not v.strip():
            raise ValueError('agent_id cannot be empty')
        return v.strip()

    @field_validator('description')
    @classmethod
    def validate_description(cls, v):
        if not v or not v.strip():
            raise ValueError('description cannot be empty')
        return v.strip()

    @field_validator('dependencies')
    @classmethod
    def validate_dependencies(cls, v):
        if v:
            for dep in v:
                if not isinstance(dep, str) or not dep.strip():
                    raise ValueError('dependencies must be non-empty strings')
        return v

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


class AgentState(BaseModel):
    """Current state of an agent."""
    agent_id: str = Field(..., min_length=1, max_length=100)
    status: str = Field(..., pattern=r'^(idle|busy|error|stopped)$')
    current_task: str | None = Field(None)
    last_heartbeat: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # test helper: return True if DB can be opened
    def _test_connection(self) -> bool:  # pragma: no cover - used by tests
        try:
            with self._get_connection() as _:
                return True
        except Exception:
            return False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator('current_task')
    @classmethod
    def validate_current_task(cls, v):
        if v is None:
            return v
        try:
            uuid.UUID(str(v))
        except Exception as e:
            raise ValueError('current_task must be a valid UUID') from e
        return str(v)

    @field_validator('agent_id')
    @classmethod
    def validate_agent_id(cls, v):
        if not v or not v.strip():
            raise ValueError('agent_id cannot be empty')
        return v.strip()

    @field_validator('last_heartbeat')
    @classmethod
    def validate_heartbeat(cls, v):
        # Heartbeat cannot be in the future (with 1 minute tolerance)
        now = datetime.now(UTC)
        if v > now + timedelta(minutes=1):
            raise ValueError('last_heartbeat cannot be in the future')
        return v

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


class SharedMemoryError(Exception):
    """Base exception for shared memory operations."""
    pass


class SharedMemory:
    # Simple per-db_path singleton to align test patches with swarm instances
    _instances: dict[str, "SharedMemory"] = {}

    def __new__(cls, db_path: str = "./data/shared_memory.db"):
        try:
            key = str(Path(db_path).resolve())
        except Exception:
            key = str(db_path)
        inst = cls._instances.get(key)
        if inst is None:
            inst = super().__new__(cls)
            cls._instances[key] = inst
        return inst

    """Shared memory system for agent coordination and state management."""

    def __init__(self, db_path: str = "./data/shared_memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize the SQLite database schema."""
        try:
            with self._get_connection() as conn:
                # Enable foreign key constraints
                conn.execute("PRAGMA foreign_keys = ON")

                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS memory_entries (
                        id TEXT PRIMARY KEY,
                        agent_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        type TEXT NOT NULL CHECK (type IN (
                            'task', 'observation', 'decision',
                            'result', 'error', 'control', 'performance_test'
                        )),
                        content TEXT NOT NULL,
                        tags TEXT,
                        parent_id TEXT,
                        context TEXT,
                        FOREIGN KEY (parent_id) REFERENCES memory_entries (id) ON DELETE SET NULL
                    );

                    CREATE TABLE IF NOT EXISTS task_status (
                        id TEXT PRIMARY KEY,
                        agent_id TEXT NOT NULL,
                        status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
                        description TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        metadata TEXT,
                        dependencies TEXT,
                        result TEXT,
                        error TEXT
                    );

                    CREATE TABLE IF NOT EXISTS agent_states (
                        agent_id TEXT PRIMARY KEY,
                        status TEXT NOT NULL CHECK (status IN ('idle', 'busy', 'error', 'stopped')),
                        current_task TEXT,
                        last_heartbeat TEXT NOT NULL,
                        metadata TEXT,
                        FOREIGN KEY (current_task) REFERENCES task_status (id) ON DELETE SET NULL

                    );

                    -- Indexes for performance
                    CREATE INDEX IF NOT EXISTS idx_memory_agent_id ON memory_entries(agent_id);
                    CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_entries(type);
                    CREATE INDEX IF NOT EXISTS idx_memory_timestamp ON memory_entries(timestamp);
                    CREATE INDEX IF NOT EXISTS idx_memory_parent_id ON memory_entries(parent_id);
                    CREATE INDEX IF NOT EXISTS idx_memory_tags ON memory_entries(tags);

                    CREATE INDEX IF NOT EXISTS idx_task_agent_id ON task_status(agent_id);
                    CREATE INDEX IF NOT EXISTS idx_task_status ON task_status(status);
                    CREATE INDEX IF NOT EXISTS idx_task_created_at ON task_status(created_at);
                    CREATE INDEX IF NOT EXISTS idx_task_updated_at ON task_status(updated_at);

                    CREATE INDEX IF NOT EXISTS idx_agent_status ON agent_states(status);
                    CREATE INDEX IF NOT EXISTS idx_agent_heartbeat ON agent_states(last_heartbeat);

                    -- Task 3.3: Add GoosePatch and AST metadata fields to memory_entries
                    -- Add columns if they don't exist (migration for existing databases)
                    PRAGMA table_info(memory_entries);
                """)

                # Check if new columns exist, add them if missing (Task 3.3 migration)
                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(memory_entries)")
                existing_columns = [column[1] for column in cursor.fetchall()]

                new_columns = {
                    'goose_patch': 'TEXT',
                    'ast_summary': 'TEXT',
                    'goose_strategy': 'TEXT',
                    'file_path': 'TEXT'
                }

                for column_name, column_type in new_columns.items():
                    if column_name not in existing_columns:
                        conn.execute(f"ALTER TABLE memory_entries ADD COLUMN {column_name} {column_type}")
                        self.logger.info(f"Added new column {column_name} to memory_entries table")

                # Add indexes for new columns
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_file_path ON memory_entries(file_path)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_goose_strategy ON memory_entries(goose_strategy)")

                conn.commit()
                self.logger.info("Database schema initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise SharedMemoryError(f"Database initialization failed: {e}")

    @contextmanager
    def _get_connection(self):
        """Get a database connection with automatic cleanup."""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Database connection error: {e}")
            raise SharedMemoryError(f"Database connection failed: {e}")
        finally:
            if conn:
                conn.close()

    def add_memory(self, entry: MemoryEntry) -> str:
        """Add a memory entry to shared memory."""
        try:
            # Coerce various entry shapes into MemoryEntry for tests that import from different package paths
            if isinstance(entry, dict):
                entry = MemoryEntry(**entry)
            elif hasattr(entry, "model_dump") and callable(getattr(entry, "model_dump")):
                # Another Pydantic model instance; re-validate via dict
                entry = MemoryEntry.model_validate(entry.model_dump())
            elif not isinstance(entry, MemoryEntry):
                # Try structural validation (handles objects with __dict__)
                try:
                    entry = MemoryEntry.model_validate(entry)
                except Exception:
                    raise ValueError("entry must be a MemoryEntry-compatible object")

            with self._lock:
                with self._get_connection() as conn:
                    # Check if parent_id exists if provided
                    if entry.parent_id:
                        parent_exists = conn.execute(
                            "SELECT 1 FROM memory_entries WHERE id = ?", (entry.parent_id,)
                        ).fetchone()
                        if not parent_exists:
                            raise ValueError(f"Parent entry {entry.parent_id} does not exist")

                        conn.execute("SELECT 1 FROM memory_entries WHERE id = ?", (entry.parent_id,))
                        parent_exists = conn.execute(
                            "SELECT 1 FROM memory_entries WHERE id = ?", (entry.parent_id,)
                        ).fetchone()
                        if not parent_exists:
                            raise ValueError(f"Parent entry {entry.parent_id} does not exist")

                    # Map any non-schema types (e.g., 'performance_test') to a valid DB type
                    allowed_types = {"task", "observation", "decision", "result", "error", "control"}
                    db_type = entry.type if entry.type in allowed_types else "observation"
                    if db_type != entry.type:
                        # Preserve original type in context for traceability
                        entry.context = dict(entry.context)
                        entry.context["original_type"] = entry.type

                    conn.execute("""
                        INSERT INTO memory_entries
                        (id, agent_id, timestamp, type, content, tags, parent_id, context, goose_patch, ast_summary, goose_strategy, file_path)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        entry.id,
                        entry.agent_id,
                        entry.timestamp.isoformat(),
                        db_type,
                        json.dumps(entry.content),
                        json.dumps(list(entry.tags)),
                        entry.parent_id,
                        json.dumps(entry.context),
                        json.dumps(entry.goose_patch) if entry.goose_patch else None,
                        json.dumps(entry.ast_summary) if entry.ast_summary else None,
                        entry.goose_strategy,
                        entry.file_path
                    ))
                    conn.commit()

            self.logger.debug(f"Added memory entry {entry.id} for agent {entry.agent_id}")
            return entry.id

        except Exception as e:
            self.logger.error(f"Failed to add memory entry: {e}")
            raise SharedMemoryError(f"Failed to add memory entry: {e}")

    def get_memories(
        self,
        agent_id: str | None = None,
        memory_type: str | None = None,
        type: str | None = None,
        tags: list[str] | set[str] | None = None,
        limit: int = 100,
        since: datetime | None = None
    ) -> list[MemoryEntry]:
        """Retrieve memory entries based on filters."""
        with self._lock:
            query = "SELECT * FROM memory_entries WHERE 1=1"
            params = []

            if agent_id:
                query += " AND agent_id = ?"
                params.append(agent_id)

            effective_type = memory_type or type
            if effective_type:
                query += " AND type = ?"
                params.append(effective_type)

            if since:
                query += " AND timestamp >= ?"
                params.append(since.isoformat())

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            with self._get_connection() as conn:
                rows = conn.execute(query, params).fetchall()

            memories = []
            for row in rows:
                # Handle new GoosePatch and AST fields (Task 3.3)
                goose_patch = json.loads(row['goose_patch']) if row['goose_patch'] else None
                ast_summary = json.loads(row['ast_summary']) if row['ast_summary'] else None

                entry = MemoryEntry(
                    id=row['id'],
                    agent_id=row['agent_id'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    type=row['type'],
                    content=json.loads(row['content']),
                    tags=set(json.loads(row['tags'] or '[]')),
                    parent_id=row['parent_id'],
                    context=json.loads(row['context'] or '{}'),
                    goose_patch=goose_patch,
                    ast_summary=ast_summary,
                    goose_strategy=row['goose_strategy'],
                    file_path=row['file_path']
                )

                # Filter by tags if specified
                if tags:
                    tags_set = set(tags) if isinstance(tags, list) else set(tags)
                    if not tags_set.intersection(entry.tags):
                        continue

                memories.append(entry)

            return memories
    # Convenience helpers used in integration tests
    def get_recent_entries(self, limit: int = 50) -> list[MemoryEntry]:
        """Return most recent memory entries across all agents."""
        return self.get_memories(limit=limit)

    def search_entries(
        self,
        tags: list[str] | set[str] | None = None,
        agent_id: str | None = None,
        memory_type: str | None = None,
        content_filter: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """Search entries by optional agent, type, and tags (intersection).

        content_filter allows matching by exact key/value pairs within the content dict.
        """
        entries = self.get_memories(agent_id=agent_id, type=memory_type, tags=tags, limit=limit)
        if tags:
            tags_lower = {t.lower() for t in (tags if isinstance(tags, (list, set)) else [tags])}
            def has_tags(e: MemoryEntry) -> bool:
                etags = {t.lower() for t in (e.tags or set())}
                return bool(etags.intersection(tags_lower))
            entries = [e for e in entries if has_tags(e)]
        if content_filter:
            def match_content(e: MemoryEntry) -> bool:
                return all(e.content.get(k) == v for k, v in content_filter.items())
            entries = [e for e in entries if match_content(e)]
        return entries


    def update_memory(self, entry_id: str, **updates) -> bool:
        """Update a memory entry."""
        try:
            valid_fields = {'content', 'tags', 'context'}

            if not any(field in updates for field in valid_fields):
                return False

            with self._lock:
                with self._get_connection() as conn:
                    # Check if entry exists
                    existing = conn.execute(
                        "SELECT 1 FROM memory_entries WHERE id = ?", (entry_id,)
                    ).fetchone()
                    if not existing:
                        return False

                    set_clauses = []
                    params = []

                    for field, value in updates.items():
                        if field in valid_fields:
                            set_clauses.append(f"{field} = ?")
                            if field == 'tags' and isinstance(value, (set, list)):
                                params.append(json.dumps(list(value)))
                            elif field in ('content', 'context'):
                                params.append(json.dumps(value))
                            else:
                                params.append(value)

                    if not set_clauses:
                        return False

                    params.append(entry_id)
                    query = f"UPDATE memory_entries SET {', '.join(set_clauses)} WHERE id = ?"

                    cursor = conn.execute(query, params)
                    conn.commit()

                    success = cursor.rowcount > 0
                    if success:
                        self.logger.debug(f"Updated memory entry {entry_id}")
                    return success

        except Exception as e:
            self.logger.error(f"Failed to update memory entry {entry_id}: {e}")
            raise SharedMemoryError(f"Failed to update memory entry: {e}")

    def delete_memory(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        try:
            with self._lock:
                with self._get_connection() as conn:
                    cursor = conn.execute(
                        "DELETE FROM memory_entries WHERE id = ?", (entry_id,)
                    )
                    conn.commit()

                    success = cursor.rowcount > 0
                    if success:
                        self.logger.debug(f"Deleted memory entry {entry_id}")
                    return success

        except Exception as e:
            self.logger.error(f"Failed to delete memory entry {entry_id}: {e}")
            raise SharedMemoryError(f"Failed to delete memory entry: {e}")

    def get_memory_by_id(self, entry_id: str) -> MemoryEntry | None:
        """Get a specific memory entry by ID."""
        try:
            with self._lock:
                with self._get_connection() as conn:
                    row = conn.execute(
                        "SELECT * FROM memory_entries WHERE id = ?", (entry_id,)
                    ).fetchone()

                if not row:
                    return None

                # Handle new GoosePatch and AST fields (Task 3.3)
                goose_patch = json.loads(row['goose_patch']) if row['goose_patch'] else None
                ast_summary = json.loads(row['ast_summary']) if row['ast_summary'] else None

                return MemoryEntry(
                    id=row['id'],
                    agent_id=row['agent_id'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    type=row['type'],
                    content=json.loads(row['content']),
                    tags=set(json.loads(row['tags'] or '[]')),
                    parent_id=row['parent_id'],
                    context=json.loads(row['context'] or '{}'),
                    goose_patch=goose_patch,
                    ast_summary=ast_summary,
                    goose_strategy=row['goose_strategy'],
                    file_path=row['file_path']
                )

        except Exception as e:
            self.logger.error(f"Failed to get memory entry {entry_id}: {e}")
            raise SharedMemoryError(f"Failed to get memory entry: {e}")

    def create_task(self, task: TaskStatus) -> str:
        """Create a new task in the system."""
        try:
            # Validate the task
            if not isinstance(task, TaskStatus):
                raise ValueError("task must be a TaskStatus instance")

            with self._lock:
                with self._get_connection() as conn:
                    # Validate dependencies exist
                    for dep_id in task.dependencies:
                        dep_exists = conn.execute(
                            "SELECT 1 FROM task_status WHERE id = ?", (dep_id,)
                        ).fetchone()
                        if not dep_exists:
                            raise ValueError(f"Dependency task {dep_id} does not exist")

                    conn.execute("""
                        INSERT INTO task_status
                        (id, agent_id, status, description, created_at, updated_at,
                         metadata, dependencies, result, error)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        task.id,
                        task.agent_id,
                        task.status,
                        task.description,
                        task.created_at.isoformat(),
                        task.updated_at.isoformat(),
                        json.dumps(task.metadata),
                        json.dumps(task.dependencies),
                        json.dumps(task.result) if task.result else None,
                        task.error
                    ))
                    conn.commit()

            self.logger.debug(f"Created task {task.id} for agent {task.agent_id}")
            return task.id

        except Exception as e:
            self.logger.error(f"Failed to create task: {e}")
            raise SharedMemoryError(f"Failed to create task: {e}")

    def add_task(self, task: TaskStatus) -> bool:
        """Add a new task to the system (alias for create_task)."""
        try:
            self.create_task(task)
            return True
        except Exception:
            return False

    def update_task(self, task_id: str, **updates) -> bool:
        """Update a task's status or metadata."""
        try:
            valid_fields = {
                'status', 'description', 'metadata', 'dependencies',
                'result', 'error'
            }

            if not any(field in updates for field in valid_fields):
                return False

            with self._lock:
                with self._get_connection() as conn:
                    # Check if task exists
                    existing = conn.execute(
                        "SELECT 1 FROM task_status WHERE id = ?", (task_id,)
                    ).fetchone()
                    if not existing:
                        return False

                    # Validate dependencies if being updated
                    if 'dependencies' in updates:
                        for dep_id in updates['dependencies']:
                            dep_exists = conn.execute(
                                "SELECT 1 FROM task_status WHERE id = ?", (dep_id,)
                            ).fetchone()
                            if not dep_exists:
                                raise ValueError(f"Dependency task {dep_id} does not exist")

                    # Always update the timestamp
                    updates['updated_at'] = datetime.now(UTC).isoformat()

                    # Prepare the update query
                    set_clauses = []
                    params = []

                    for field, value in updates.items():
                        if field in valid_fields or field == 'updated_at':
                            set_clauses.append(f"{field} = ?")
                            if field in ('metadata', 'dependencies', 'result') and value is not None:
                                params.append(json.dumps(value))
                            else:
                                params.append(value)

                    if not set_clauses:
                        return False

                    params.append(task_id)
                    query = f"UPDATE task_status SET {', '.join(set_clauses)} WHERE id = ?"

                    cursor = conn.execute(query, params)
                    conn.commit()

                    success = cursor.rowcount > 0
                    if success:
                        self.logger.debug(f"Updated task {task_id}")
                    return success

        except Exception as e:
            self.logger.error(f"Failed to update task {task_id}: {e}")
            raise SharedMemoryError(f"Failed to update task: {e}")


    def update_task_status(self, task_id: str, status: str, result: dict[str, Any] | None = None) -> bool:
        """Convenience method to update only status and optional result."""
        return self.update_task(task_id, status=status, result=result)

    def get_task(self, task_id: str) -> TaskStatus | None:
        """Get a specific task by ID."""
        with self._lock:
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM task_status WHERE id = ?", (task_id,)
                ).fetchone()

            if not row:
                return None

            return TaskStatus(
                id=row['id'],
                agent_id=row['agent_id'],
                status=row['status'],
                description=row['description'],
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at']),
                metadata=json.loads(row['metadata'] or '{}'),
                dependencies=json.loads(row['dependencies'] or '[]'),
                result=json.loads(row['result']) if row['result'] else None,
                error=row['error']
            )

    def get_tasks(
        self,
        agent_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
        task_id: str | None = None,
    ) -> list[TaskStatus]:
        """Get tasks based on filters."""
        with self._lock:
            query = "SELECT * FROM task_status WHERE 1=1"
            params = []

            if agent_id:
                query += " AND agent_id = ?"
                params.append(agent_id)

            if status:
                query += " AND status = ?"
                params.append(status)

            if task_id:
                query += " AND id = ?"
                params.append(task_id)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            with self._get_connection() as conn:
                rows = conn.execute(query, params).fetchall()

            tasks = []
            for row in rows:
                task = TaskStatus(
                    id=row['id'],
                    agent_id=row['agent_id'],
                    status=row['status'],
                    description=row['description'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at']),
                    metadata=json.loads(row['metadata'] or '{}'),
                    dependencies=json.loads(row['dependencies'] or '[]'),
                    result=json.loads(row['result']) if row['result'] else None,
                    error=row['error']
                )
                tasks.append(task)

            return tasks

    def delete_task(self, task_id: str) -> bool:
        """Delete a task."""
        try:
            with self._lock:
                with self._get_connection() as conn:
                    cursor = conn.execute(
                        "DELETE FROM task_status WHERE id = ?", (task_id,)
                    )
                    conn.commit()

                    success = cursor.rowcount > 0
                    if success:
                        self.logger.debug(f"Deleted task {task_id}")
                    return success

        except Exception as e:
            self.logger.error(f"Failed to delete task {task_id}: {e}")
            raise SharedMemoryError(f"Failed to delete task: {e}")

    def update_agent_state(self, state: AgentState) -> None:
        """Update an agent's current state."""
        try:
            # Validate the state
            if not isinstance(state, AgentState):
                raise ValueError("state must be an AgentState instance")

            with self._lock:
                with self._get_connection() as conn:
                    # Validate current_task exists if provided
                    if state.current_task:
                        task_exists = conn.execute(
                            "SELECT 1 FROM task_status WHERE id = ?", (state.current_task,)
                        ).fetchone()
                        if not task_exists:
                            raise ValueError(f"Current task {state.current_task} does not exist")

                    conn.execute("""
                        INSERT OR REPLACE INTO agent_states
                        (agent_id, status, current_task, last_heartbeat, metadata)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        state.agent_id,
                        state.status,
                        state.current_task,
                        state.last_heartbeat.isoformat(),
                        json.dumps(state.metadata)
                    ))
                    conn.commit()

            self.logger.debug(f"Updated agent state for {state.agent_id}")

        except Exception as e:
            self.logger.error(f"Failed to update agent state for {state.agent_id}: {e}")
            raise SharedMemoryError(f"Failed to update agent state: {e}")

    def get_agent_state(self, agent_id: str) -> AgentState | None:
        """Get the current state of an agent."""
        with self._lock:
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM agent_states WHERE agent_id = ?", (agent_id,)
                ).fetchone()

            if not row:
                return None

            return AgentState(
                agent_id=row['agent_id'],
                status=row['status'],
                current_task=row['current_task'],
                last_heartbeat=datetime.fromisoformat(row['last_heartbeat']),
                metadata=json.loads(row['metadata'] or '{}')
            )

    def get_all_agent_states(self) -> list[AgentState]:
        """Get the current state of all agents."""
        with self._lock:
            with self._get_connection() as conn:
                rows = conn.execute("SELECT * FROM agent_states").fetchall()

            states = []
            for row in rows:
                state = AgentState(
                    agent_id=row['agent_id'],
                    status=row['status'],
                    current_task=row['current_task'],
                    last_heartbeat=datetime.fromisoformat(row['last_heartbeat']),
                    metadata=json.loads(row['metadata'] or '{}')
                )
                states.append(state)

            return states

    def get_agent_states(self) -> dict[str, AgentState]:
        """Get the current state of all agents as a dictionary keyed by agent_id."""
        states = self.get_all_agent_states()
        return {state.agent_id: state for state in states}

    def delete_agent_state(self, agent_id: str) -> bool:
        """Delete an agent's state."""
        try:
            with self._lock:
                with self._get_connection() as conn:
                    cursor = conn.execute(
                        "DELETE FROM agent_states WHERE agent_id = ?", (agent_id,)
                    )
                    conn.commit()

                    success = cursor.rowcount > 0
                    if success:
                        self.logger.debug(f"Deleted agent state for {agent_id}")
                    return success

        except Exception as e:
            self.logger.error(f"Failed to delete agent state for {agent_id}: {e}")
            raise SharedMemoryError(f"Failed to delete agent state: {e}")

    def cleanup_old_entries(self, days: int = 30) -> dict[str, int]:
        """Remove old entries from all tables."""
        try:
            cutoff = datetime.now(UTC).replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=days)

            deleted_counts = {}

            with self._lock:
                with self._get_connection() as conn:
                    # Clean up memory entries
                    cursor = conn.execute(
                        "DELETE FROM memory_entries WHERE timestamp < ?",
                        (cutoff.isoformat(),)
                    )
                    deleted_counts['memory_entries'] = cursor.rowcount

                    # Clean up completed/failed tasks older than cutoff
                    cursor = conn.execute(
                        "DELETE FROM task_status WHERE updated_at < ? AND status IN ('completed', 'failed', 'cancelled')",
                        (cutoff.isoformat(),)
                    )
                    deleted_counts['tasks'] = cursor.rowcount

                    # Clean up agent states with old heartbeats (more than 1 hour)
                    old_heartbeat = datetime.now(UTC) - timedelta(hours=1)
                    cursor = conn.execute(
                        "DELETE FROM agent_states WHERE last_heartbeat < ?",
                        (old_heartbeat.isoformat(),)
                    )
                    deleted_counts['agent_states'] = cursor.rowcount

                    conn.commit()

            total_deleted = sum(deleted_counts.values())
            self.logger.info(f"Cleanup completed: {total_deleted} entries deleted")
            return deleted_counts

        except Exception as e:
            self.logger.error(f"Failed to cleanup old entries: {e}")
            raise SharedMemoryError(f"Failed to cleanup old entries: {e}")

    def get_conversation_thread(self, entry_id: str) -> list[MemoryEntry]:
        """Get a conversation thread starting from a specific entry."""
        try:
            memories = []
            visited = set()  # Prevent infinite loops
            current_id = entry_id

            # Get the full thread by following parent_id chain
            while current_id and current_id not in visited:
                visited.add(current_id)

                with self._lock:
                    with self._get_connection() as conn:
                        row = conn.execute(
                            "SELECT * FROM memory_entries WHERE id = ?", (current_id,)
                        ).fetchone()

                    if not row:
                        break

                    entry = MemoryEntry(
                        id=row['id'],
                        agent_id=row['agent_id'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        type=row['type'],
                        content=json.loads(row['content']),
                        tags=set(json.loads(row['tags'] or '[]')),
                        parent_id=row['parent_id'],
                        context=json.loads(row['context'] or '{}')
                    )
                    memories.insert(0, entry)  # Insert at beginning for chronological order
                    current_id = entry.parent_id

            # Also get any child entries
            self._get_child_entries(entry_id, memories, visited)

            return sorted(memories, key=lambda x: x.timestamp)

        except Exception as e:
            self.logger.error(f"Failed to get conversation thread for {entry_id}: {e}")
            raise SharedMemoryError(f"Failed to get conversation thread: {e}")

    def _get_child_entries(self, parent_id: str, memories: list[MemoryEntry], visited: set[str]) -> None:
        """Recursively get child entries."""
        if parent_id in visited:
            return  # Prevent infinite recursion

        with self._lock:
            with self._get_connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM memory_entries WHERE parent_id = ? ORDER BY timestamp",
                    (parent_id,)
                ).fetchall()

            for row in rows:
                if row['id'] in visited:
                    continue  # Skip already visited entries

                visited.add(row['id'])
                entry = MemoryEntry(
                    id=row['id'],
                    agent_id=row['agent_id'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    type=row['type'],
                    content=json.loads(row['content']),
                    tags=set(json.loads(row['tags'] or '[]')),
                    parent_id=row['parent_id'],
                    context=json.loads(row['context'] or '{}')
                )
                memories.append(entry)

                # Recursively get children of this entry
                self._get_child_entries(entry.id, memories, visited)

    def search_memories(self, query: str, agent_id: str | None = None,
                       memory_type: str | None = None, limit: int = 50) -> list[MemoryEntry]:
        """Search memory entries by content."""
        try:
            with self._lock:
                sql_query = """
                    SELECT * FROM memory_entries
                    WHERE (content LIKE ? OR json_extract(context, '$.source') LIKE ?)
                """
                params = [f"%{query}%", f"%{query}%"]

                if agent_id:
                    sql_query += " AND agent_id = ?"
                    params.append(agent_id)

                if memory_type:
                    sql_query += " AND type = ?"
                    params.append(memory_type)

                sql_query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)

                with self._get_connection() as conn:
                    rows = conn.execute(sql_query, params).fetchall()

                memories = []
                for row in rows:
                    entry = MemoryEntry(
                        id=row['id'],
                        agent_id=row['agent_id'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        type=row['type'],
                        content=json.loads(row['content']),
                        tags=set(json.loads(row['tags'] or '[]')),
                        parent_id=row['parent_id'],
                        context=json.loads(row['context'] or '{}')
                    )
                    memories.append(entry)

                return memories

        except Exception as e:
            self.logger.error(f"Failed to search memories: {e}")
            raise SharedMemoryError(f"Failed to search memories: {e}")

    def get_statistics(self) -> dict[str, Any]:
        """Get database statistics."""
        try:
            with self._lock:
                with self._get_connection() as conn:
                    stats = {}

                    # Memory entries stats
                    row = conn.execute("SELECT COUNT(*) as count FROM memory_entries").fetchone()
                    stats['memory_entries_count'] = row['count']

                    # Memory entries by type
                    rows = conn.execute("""
                        SELECT type, COUNT(*) as count
                        FROM memory_entries
                        GROUP BY type
                    """).fetchall()
                    stats['memory_entries_by_type'] = {row['type']: row['count'] for row in rows}

                    # Tasks stats
                    row = conn.execute("SELECT COUNT(*) as count FROM task_status").fetchone()
                    stats['tasks_count'] = row['count']

                    # Tasks by status
                    rows = conn.execute("""
                        SELECT status, COUNT(*) as count
                        FROM task_status
                        GROUP BY status
                    """).fetchall()
                    stats['tasks_by_status'] = {row['status']: row['count'] for row in rows}

                    # Agent states stats
                    row = conn.execute("SELECT COUNT(*) as count FROM agent_states").fetchone()
                    stats['agent_states_count'] = row['count']

                    # Agent states by status
                    rows = conn.execute("""
                        SELECT status, COUNT(*) as count
                        FROM agent_states
                        GROUP BY status
                    """).fetchall()
                    stats['agent_states_by_status'] = {row['status']: row['count'] for row in rows}

                    return stats

        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            raise SharedMemoryError(f"Failed to get statistics: {e}")

    def get_audit_trail(self, agent_id: str | None = None,
                       start_time: datetime | None = None,
                       end_time: datetime | None = None,
                       limit: int = 100) -> list[MemoryEntry]:
        """Get audit trail of memory entries for analysis and replay."""
        try:
            with self._lock:
                query = "SELECT * FROM memory_entries WHERE 1=1"
                params = []

                if agent_id:
                    query += " AND agent_id = ?"
                    params.append(agent_id)

                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time.isoformat())

                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time.isoformat())

                query += " ORDER BY timestamp ASC LIMIT ?"
                params.append(limit)

                with self._get_connection() as conn:
                    rows = conn.execute(query, params).fetchall()

                entries = []
                for row in rows:
                    entry = MemoryEntry(
                        id=row['id'],
                        agent_id=row['agent_id'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        type=row['type'],
                        content=json.loads(row['content']),
                        tags=set(json.loads(row['tags'] or '[]')),
                        parent_id=row['parent_id'],
                        context=json.loads(row['context'] or '{}')
                    )
                    entries.append(entry)

                return entries

        except Exception as e:
            self.logger.error(f"Failed to get audit trail: {e}")
            raise SharedMemoryError(f"Failed to get audit trail: {e}")

    def replay_actions(self, entries: list[MemoryEntry]) -> dict[str, Any]:
        """Replay a sequence of memory entries for analysis."""
        try:
            replay_summary = {
                'total_entries': len(entries),
                'agents_involved': set(),
                'action_types': {},
                'timeline': [],
                'decision_chain': [],
                'errors': []
            }

            for entry in entries:
                replay_summary['agents_involved'].add(entry.agent_id)

                # Count action types
                if entry.type in replay_summary['action_types']:
                    replay_summary['action_types'][entry.type] += 1
                else:
                    replay_summary['action_types'][entry.type] = 1

                # Build timeline
                summary = entry.content.get('message', str(entry.content)) if isinstance(entry.content, dict) else str(entry.content)
                timeline_item = {
                    'timestamp': entry.timestamp,
                    'agent_id': entry.agent_id,
                    'type': entry.type,
                    'summary': summary[:100]
                }
                replay_summary['timeline'].append(timeline_item)

                # Track decision chain
                if entry.type == 'decision':
                    decision_item = {
                        'timestamp': entry.timestamp,
                        'agent_id': entry.agent_id,
                        'decision': entry.content,
                        'parent_id': entry.parent_id
                    }
                    replay_summary['decision_chain'].append(decision_item)

                # Track errors
                if entry.type == 'error':
                    error_item = {
                        'timestamp': entry.timestamp,
                        'agent_id': entry.agent_id,
                        'error': entry.content,
                        'context': entry.context
                    }
                    replay_summary['errors'].append(error_item)

            # Convert sets to lists for JSON serialization
            replay_summary['agents_involved'] = list(replay_summary['agents_involved'])

            return replay_summary

        except Exception as e:
            self.logger.error(f"Failed to replay actions: {e}")
            raise SharedMemoryError(f"Failed to replay actions: {e}")

    def archive_old_entries(self, days: int = 90, archive_path: str | None = None) -> dict[str, Any]:
        """Archive old entries to a separate database or file."""
        try:
            cutoff = datetime.now(UTC) - timedelta(days=days)

            # Get entries to archive
            entries_to_archive = []
            tasks_to_archive = []

            with self._lock:
                with self._get_connection() as conn:
                    # Get old memory entries
                    rows = conn.execute(
                        "SELECT * FROM memory_entries WHERE timestamp < ?",
                        (cutoff.isoformat(),)
                    ).fetchall()

                    for row in rows:
                        entry = {
                            'id': row['id'],
                            'agent_id': row['agent_id'],
                            'timestamp': row['timestamp'],
                            'type': row['type'],
                            'content': row['content'],
                            'tags': row['tags'],
                            'parent_id': row['parent_id'],
                            'context': row['context']
                        }
                        entries_to_archive.append(entry)

                    # Get old completed tasks
                    rows = conn.execute(
                        "SELECT * FROM task_status WHERE updated_at < ? AND status IN ('completed', 'failed', 'cancelled')",
                        (cutoff.isoformat(),)
                    ).fetchall()

                    for row in rows:
                        task = {
                            'id': row['id'],
                            'agent_id': row['agent_id'],
                            'status': row['status'],
                            'description': row['description'],
                            'created_at': row['created_at'],
                            'updated_at': row['updated_at'],
                            'metadata': row['metadata'],
                            'dependencies': row['dependencies'],
                            'result': row['result'],
                            'error': row['error']
                        }
                        tasks_to_archive.append(task)

            # Create archive
            archive_data = {
                'archived_at': datetime.now(UTC).isoformat(),
                'cutoff_date': cutoff.isoformat(),
                'memory_entries': entries_to_archive,
                'tasks': tasks_to_archive
            }

            # Save archive if path provided
            if archive_path:
                archive_file = Path(archive_path)
                archive_file.parent.mkdir(parents=True, exist_ok=True)

                with open(archive_file, 'w') as f:
                    json.dump(archive_data, f, indent=2)

            # Remove archived entries from main database
            deleted_counts = self.cleanup_old_entries(days)

            result = {
                'archived_entries': len(entries_to_archive),
                'archived_tasks': len(tasks_to_archive),
                'deleted_counts': deleted_counts,
                'archive_file': str(archive_path) if archive_path else None
            }

            self.logger.info(f"Archived {result['archived_entries']} entries and {result['archived_tasks']} tasks")
            return result

        except Exception as e:
            self.logger.error(f"Failed to archive old entries: {e}")
            raise SharedMemoryError(f"Failed to archive old entries: {e}")

    def get_memory_by_tags(self, tags: set[str], match_all: bool = False, limit: int = 50) -> list[MemoryEntry]:
        """Get memory entries by tags."""
        try:
            with self._lock:
                with self._get_connection() as conn:
                    if match_all:
                        # All tags must be present
                        query = "SELECT * FROM memory_entries WHERE "
                        conditions = []
                        params = []

                        for tag in tags:
                            conditions.append("json_extract(tags, '$') LIKE ?")
                            params.append(f'%"{tag}"%')

                        query += " AND ".join(conditions)
                        query += " ORDER BY timestamp DESC LIMIT ?"
                        params.append(limit)
                    else:
                        # Any tag matches
                        query = "SELECT * FROM memory_entries WHERE "
                        conditions = []
                        params = []

                        for tag in tags:
                            conditions.append("json_extract(tags, '$') LIKE ?")
                            params.append(f'%"{tag}"%')

                        query += " OR ".join(conditions)
                        query += " ORDER BY timestamp DESC LIMIT ?"
                        params.append(limit)

                    rows = conn.execute(query, params).fetchall()

                memories = []
                for row in rows:
                    # Handle new GoosePatch and AST fields (Task 3.3)
                    goose_patch = json.loads(row['goose_patch']) if row['goose_patch'] else None
                    ast_summary = json.loads(row['ast_summary']) if row['ast_summary'] else None

                    entry = MemoryEntry(
                        id=row['id'],
                        agent_id=row['agent_id'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        type=row['type'],
                        content=json.loads(row['content']),
                        tags=set(json.loads(row['tags'] or '[]')),
                        parent_id=row['parent_id'],
                        context=json.loads(row['context'] or '{}'),
                        goose_patch=goose_patch,
                        ast_summary=ast_summary,
                        goose_strategy=row['goose_strategy'],
                        file_path=row['file_path']
                    )
                    memories.append(entry)

                return memories

        except Exception as e:
            self.logger.error(f"Failed to get memories by tags: {e}")
            raise SharedMemoryError(f"Failed to get memories by tags: {e}")

    # Task 3.3: GoosePatch and AST metadata helper methods
    def log_goose_patch_memory(
        self,
        agent_id: str,
        goose_patch: dict[str, Any],
        ast_summary: dict[str, Any] | None = None,
        goose_strategy: str | None = None,
        file_path: str | None = None,
        parent_id: str | None = None
    ) -> str:
        """Log Goose tool outputs and reasoning in shared memory for traceability."""
        content = {
            "action": "goose_patch_applied",
            "patch_metadata": goose_patch,
            "timestamp": datetime.now(UTC).isoformat()
        }

        if ast_summary:
            content["ast_analysis"] = ast_summary

        entry = MemoryEntry(
            agent_id=agent_id,
            type="result",
            content=content,
            tags={"goose", "patch", "refactor"},
            goose_patch=goose_patch,
            ast_summary=ast_summary,
            goose_strategy=goose_strategy,
            file_path=file_path,
            parent_id=parent_id
        )

        return self.add_memory(entry)

    def log_ast_analysis_memory(
        self,
        agent_id: str,
        file_path: str,
        ast_summary: dict[str, Any],
        refactor_strategy: str | None = None,
        parent_id: str | None = None
    ) -> str:
        """Log AST analysis and refactoring strategies for auditability."""
        content = {
            "action": "ast_analysis",
            "file_analyzed": file_path,
            "analysis_results": ast_summary,
            "timestamp": datetime.now(UTC).isoformat()
        }

        if refactor_strategy:
            content["refactor_strategy"] = refactor_strategy

        entry = MemoryEntry(
            agent_id=agent_id,
            type="observation",
            content=content,
            tags={"ast", "analysis", "refactor"},
            goose_patch=None,
            ast_summary=ast_summary,
            goose_strategy=refactor_strategy,
            file_path=file_path,
            parent_id=parent_id
        )

        return self.add_memory(entry)

    def get_goose_patches_for_file(self, file_path: str) -> list[MemoryEntry]:
        """Retrieve all Goose patches applied to a specific file."""
        try:
            with self._lock:
                with self._get_connection() as conn:
                    rows = conn.execute("""
                        SELECT * FROM memory_entries
                        WHERE file_path = ? AND goose_patch IS NOT NULL
                        ORDER BY timestamp DESC
                    """, (file_path,)).fetchall()

                    memories = []
                    for row in rows:
                        goose_patch = json.loads(row['goose_patch']) if row['goose_patch'] else None
                        ast_summary = json.loads(row['ast_summary']) if row['ast_summary'] else None

                        entry = MemoryEntry(
                            id=row['id'],
                            agent_id=row['agent_id'],
                            timestamp=datetime.fromisoformat(row['timestamp']),
                            type=row['type'],
                            content=json.loads(row['content']),
                            tags=set(json.loads(row['tags'] or '[]')),
                            parent_id=row['parent_id'],
                            context=json.loads(row['context'] or '{}'),
                            goose_patch=goose_patch,
                            ast_summary=ast_summary,
                            goose_strategy=row['goose_strategy'],
                            file_path=row['file_path']
                        )
                        memories.append(entry)

                    return memories

        except Exception as e:
            self.logger.error(f"Failed to get Goose patches for file {file_path}: {e}")
            raise SharedMemoryError(f"Failed to get Goose patches: {e}")

    def get_ast_summaries_by_strategy(self, goose_strategy: str) -> list[MemoryEntry]:
        """Retrieve AST summaries that used a specific Goose refactoring strategy."""
        try:
            with self._lock:
                with self._get_connection() as conn:
                    rows = conn.execute("""
                        SELECT * FROM memory_entries
                        WHERE goose_strategy = ? AND ast_summary IS NOT NULL
                        ORDER BY timestamp DESC
                    """, (goose_strategy,)).fetchall()

                    memories = []
                    for row in rows:
                        goose_patch = json.loads(row['goose_patch']) if row['goose_patch'] else None
                        ast_summary = json.loads(row['ast_summary']) if row['ast_summary'] else None

                        entry = MemoryEntry(
                            id=row['id'],
                            agent_id=row['agent_id'],
                            timestamp=datetime.fromisoformat(row['timestamp']),
                            type=row['type'],
                            content=json.loads(row['content']),
                            tags=set(json.loads(row['tags'] or '[]')),
                            parent_id=row['parent_id'],
                            context=json.loads(row['context'] or '{}'),
                            goose_patch=goose_patch,
                            ast_summary=ast_summary,
                            goose_strategy=row['goose_strategy'],
                            file_path=row['file_path']
                        )
                        memories.append(entry)

                    return memories

        except Exception as e:
            self.logger.error(f"Failed to get AST summaries for strategy {goose_strategy}: {e}")
            raise SharedMemoryError(f"Failed to get AST summaries: {e}")
