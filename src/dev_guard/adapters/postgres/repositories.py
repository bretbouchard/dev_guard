from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Iterable, Optional

try:
    import psycopg2  # type: ignore
    from psycopg2.pool import SimpleConnectionPool  # type: ignore
    from psycopg2.extras import RealDictCursor, Json  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "psycopg2-binary not installed. Install with: pip install dev-guard[db]"
    ) from exc

from dev_guard.adapters.repositories import (
    GuardRuleRepository,
    PolicyRepository,
    ViolationRepository,
)
from dev_guard.domain import GuardRule, Policy, RuleSeverity, RuleType, Violation, ViolationStatus


@dataclass
class PostgresConnectionProvider:
    dsn: str
    minconn: int = 1
    maxconn: int = 5

    def __post_init__(self) -> None:
        # Normalize SQLAlchemy-style DSN to a psycopg2-compatible DSN if necessary
        dsn = self.dsn.replace("postgresql+psycopg2://", "postgresql://")
        self._pool = SimpleConnectionPool(self.minconn, self.maxconn, dsn=dsn)

    @contextlib.contextmanager
    def connection(self):
        conn = self._pool.getconn()
        try:
            yield conn
        finally:
            self._pool.putconn(conn)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS policies (
    id UUID PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    version INTEGER NOT NULL,
    active BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS guard_rules (
    id UUID PRIMARY KEY,
    policy_id UUID REFERENCES policies(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    type TEXT NOT NULL,
    severity TEXT NOT NULL,
    criteria JSONB NOT NULL,
    enabled BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS violations (
    id UUID PRIMARY KEY,
    rule_id UUID REFERENCES guard_rules(id) ON DELETE SET NULL,
    policy_id UUID REFERENCES policies(id) ON DELETE SET NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    context JSONB NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    resolved_at TIMESTAMPTZ
);
"""


def ensure_schema(cp: PostgresConnectionProvider) -> None:
    with cp.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
            conn.commit()


class PostgresPolicyRepository(PolicyRepository):
    def __init__(self, cp: PostgresConnectionProvider):
        self._cp = cp

    def get(self, policy_id: str) -> Optional[Policy]:
        with self._cp.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM policies WHERE id = %s", (policy_id,))
                row = cur.fetchone()
                if not row:
                    return None
                policy = self._row_to_policy(row)
                policy.rules = list(self._list_rules(conn, policy.id))
                return policy

    def get_by_name(self, name: str) -> Optional[Policy]:
        with self._cp.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM policies WHERE name = %s", (name,))
                row = cur.fetchone()
                if not row:
                    return None
                policy = self._row_to_policy(row)
                policy.rules = list(self._list_rules(conn, policy.id))
                return policy

    def list(self) -> Iterable[Policy]:
        with self._cp.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM policies ORDER BY created_at DESC")
                for row in cur.fetchall():
                    policy = self._row_to_policy(row)
                    policy.rules = list(self._list_rules(conn, policy.id))
                    yield policy

    def save(self, policy: Policy) -> None:
        with self._cp.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO policies (id, name, description, version, active, created_at, updated_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (id) DO UPDATE SET
                        name = EXCLUDED.name,
                        description = EXCLUDED.description,
                        version = EXCLUDED.version,
                        active = EXCLUDED.active,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (
                        policy.id,
                        policy.name,
                        policy.description,
                        policy.version,
                        policy.active,
                        policy.created_at,
                        policy.updated_at,
                    ),
                )
                for r in policy.rules:
                    r.policy_id = policy.id
                    self._upsert_rule(cur, r)
            conn.commit()

    def delete(self, policy_id: str) -> bool:
        with self._cp.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM policies WHERE id = %s", (policy_id,))
            conn.commit()
            return cur.rowcount > 0

    def _list_rules(self, conn, policy_id: str):
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM guard_rules WHERE policy_id = %s ORDER BY created_at", (policy_id,))
            for row in cur.fetchall():
                yield self._row_to_rule(row)

    def _upsert_rule(self, cur, rule: GuardRule) -> None:
        cur.execute(
            """
            INSERT INTO guard_rules (id, policy_id, name, description, type, severity, criteria, enabled, created_at, updated_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (id) DO UPDATE SET
                policy_id = EXCLUDED.policy_id,
                name = EXCLUDED.name,
                description = EXCLUDED.description,
                type = EXCLUDED.type,
                severity = EXCLUDED.severity,
                criteria = EXCLUDED.criteria,
                enabled = EXCLUDED.enabled,
                updated_at = EXCLUDED.updated_at
            """,
            (
                rule.id,
                rule.policy_id,
                rule.name,
                rule.description,
                rule.type.value,
                rule.severity.value,
                Json(rule.criteria),
                rule.enabled,
                rule.created_at,
                rule.updated_at,
            ),
        )

    def _row_to_policy(self, row: dict) -> Policy:
        return Policy(
            id=str(row["id"]),
            name=row["name"],
            description=row.get("description"),
            version=row["version"],
            active=row["active"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_rule(self, row: dict) -> GuardRule:
        return GuardRule(
            id=str(row["id"]),
            policy_id=str(row["policy_id"]) if row.get("policy_id") else None,
            name=row["name"],
            description=row.get("description"),
            type=RuleType(row["type"]),
            severity=RuleSeverity(row["severity"]),
            criteria=row["criteria"],
            enabled=row["enabled"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class PostgresGuardRuleRepository(GuardRuleRepository):
    def __init__(self, cp: PostgresConnectionProvider):
        self._cp = cp

    def get(self, rule_id: str) -> Optional[GuardRule]:
        with self._cp.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM guard_rules WHERE id = %s", (rule_id,))
                row = cur.fetchone()
                return PostgresPolicyRepository._row_to_rule(self, row) if row else None

    def list_by_policy(self, policy_id: str) -> Iterable[GuardRule]:
        with self._cp.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM guard_rules WHERE policy_id = %s ORDER BY created_at", (policy_id,))
                for row in cur.fetchall():
                    yield PostgresPolicyRepository._row_to_rule(self, row)

    def save(self, rule: GuardRule) -> None:
        with self._cp.connection() as conn:
            with conn.cursor() as cur:
                PostgresPolicyRepository._upsert_rule(self, cur, rule)
            conn.commit()

    def delete(self, rule_id: str) -> bool:
        with self._cp.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM guard_rules WHERE id = %s", (rule_id,))
            conn.commit()
            return cur.rowcount > 0


class PostgresViolationRepository(ViolationRepository):
    def __init__(self, cp: PostgresConnectionProvider):
        self._cp = cp

    def get(self, violation_id: str) -> Optional[Violation]:
        with self._cp.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM violations WHERE id = %s", (violation_id,))
                row = cur.fetchone()
                return self._row_to_violation(row) if row else None

    def list_open(self, policy_id: Optional[str] = None) -> Iterable[Violation]:
        with self._cp.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if policy_id:
                    cur.execute(
                        "SELECT * FROM violations WHERE status = 'open' AND policy_id = %s ORDER BY created_at DESC",
                        (policy_id,),
                    )
                else:
                    cur.execute(
                        "SELECT * FROM violations WHERE status = 'open' ORDER BY created_at DESC"
                    )
                for row in cur.fetchall():
                    yield self._row_to_violation(row)

    def save(self, violation: Violation) -> None:
        with self._cp.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO violations (id, rule_id, policy_id, severity, message, context, status, created_at, resolved_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (id) DO UPDATE SET
                        rule_id = EXCLUDED.rule_id,
                        policy_id = EXCLUDED.policy_id,
                        severity = EXCLUDED.severity,
                        message = EXCLUDED.message,
                        context = EXCLUDED.context,
                        status = EXCLUDED.status,
                        resolved_at = EXCLUDED.resolved_at
                    """,
                    (
                        violation.id,
                        violation.rule_id,
                        violation.policy_id,
                        violation.severity.value,
                        violation.message,
                        Json(violation.context),
                        violation.status.value,
                        violation.created_at,
                        violation.resolved_at,
                    ),
                )
            conn.commit()

    def resolve(self, violation_id: str) -> bool:
        with self._cp.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE violations SET status = 'resolved', resolved_at = NOW() WHERE id = %s AND status != 'resolved'",
                    (violation_id,),
                )
            conn.commit()
            return cur.rowcount > 0

    def _row_to_violation(self, row: dict) -> Violation:
        return Violation(
            id=str(row["id"]),
            rule_id=str(row["rule_id"]) if row.get("rule_id") else "",
            policy_id=str(row["policy_id"]) if row.get("policy_id") else None,
            severity=RuleSeverity(row["severity"]),
            message=row["message"],
            context=row["context"],
            status=ViolationStatus(row["status"]),
            created_at=row["created_at"],
            resolved_at=row.get("resolved_at"),
        )

