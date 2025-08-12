from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional

from dev_guard.domain import GuardRule, Policy, Violation


class PolicyRepository(ABC):
    @abstractmethod
    def get(self, policy_id: str) -> Optional[Policy]:
        raise NotImplementedError

    @abstractmethod
    def get_by_name(self, name: str) -> Optional[Policy]:
        raise NotImplementedError

    @abstractmethod
    def list(self) -> Iterable[Policy]:
        raise NotImplementedError

    @abstractmethod
    def save(self, policy: Policy) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, policy_id: str) -> bool:
        raise NotImplementedError


class GuardRuleRepository(ABC):
    @abstractmethod
    def get(self, rule_id: str) -> Optional[GuardRule]:
        raise NotImplementedError

    @abstractmethod
    def list_by_policy(self, policy_id: str) -> Iterable[GuardRule]:
        raise NotImplementedError

    @abstractmethod
    def save(self, rule: GuardRule) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, rule_id: str) -> bool:
        raise NotImplementedError


class ViolationRepository(ABC):
    @abstractmethod
    def get(self, violation_id: str) -> Optional[Violation]:
        raise NotImplementedError

    @abstractmethod
    def list_open(self, policy_id: Optional[str] = None) -> Iterable[Violation]:
        raise NotImplementedError

    @abstractmethod
    def save(self, violation: Violation) -> None:
        raise NotImplementedError

    @abstractmethod
    def resolve(self, violation_id: str) -> bool:
        raise NotImplementedError
