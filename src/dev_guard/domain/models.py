from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class RuleSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RuleType(str, Enum):
    CODE_QUALITY = "code_quality"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    PERFORMANCE = "performance"
    CUSTOM = "custom"


class ViolationStatus(str, Enum):
    OPEN = "open"
    SUPPRESSED = "suppressed"
    RESOLVED = "resolved"


class GuardRule(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=2000)
    type: RuleType = RuleType.CUSTOM
    severity: RuleSeverity = RuleSeverity.MEDIUM
    criteria: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    enabled: bool = True
    policy_id: Optional[str] = None

    @field_validator("criteria")
    @classmethod
    def ensure_json_safe(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        return v or {}


class Policy(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=2000)
    rules: List[GuardRule] = Field(default_factory=list)
    version: int = 1
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    active: bool = True

    def add_rule(self, rule: GuardRule) -> None:
        rule.policy_id = self.id
        self.rules.append(rule)
        self.updated_at = datetime.utcnow()

    def remove_rule(self, rule_id: str) -> bool:
        before = len(self.rules)
        self.rules = [r for r in self.rules if r.id != rule_id]
        changed = len(self.rules) != before
        if changed:
            self.updated_at = datetime.utcnow()
        return changed


class Violation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    rule_id: str
    policy_id: Optional[str] = None
    severity: RuleSeverity
    message: str = Field(min_length=1, max_length=2000)
    context: Dict[str, Any] = Field(default_factory=dict)
    status: ViolationStatus = ViolationStatus.OPEN
    created_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None

    @field_validator("resolved_at")
    @classmethod
    def ensure_resolved_consistency(cls, v: Optional[datetime], info):
        status: ViolationStatus = info.data.get("status", ViolationStatus.OPEN)  # type: ignore[assignment]
        if v and status != ViolationStatus.RESOLVED:
            info.data["status"] = ViolationStatus.RESOLVED
        return v

