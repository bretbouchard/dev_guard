import pytest

postgres = pytest.importorskip("testcontainers.postgres")

from dev_guard.adapters.postgres.repositories import (
    PostgresConnectionProvider,
    PostgresPolicyRepository,
    PostgresGuardRuleRepository,
    PostgresViolationRepository,
    ensure_schema,
)
from dev_guard.domain import GuardRule, Policy, RuleSeverity, RuleType


@pytest.mark.integration
def test_postgres_repositories_end_to_end():
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as pg:
        dsn = pg.get_connection_url()
        cp = PostgresConnectionProvider(dsn=dsn)
        ensure_schema(cp)

        policy_repo = PostgresPolicyRepository(cp)
        rule_repo = PostgresGuardRuleRepository(cp)
        violation_repo = PostgresViolationRepository(cp)

        p = Policy(name="test-policy", description="integration")
        r1 = GuardRule(name="block-low-quality", type=RuleType.CODE_QUALITY, severity=RuleSeverity.LOW, criteria={"equals": {"a": 1}})
        r2 = GuardRule(name="block-insecure", type=RuleType.SECURITY, severity=RuleSeverity.HIGH, criteria={"equals": {"b": 2}})
        p.add_rule(r1)
        p.add_rule(r2)
        policy_repo.save(p)

        got = policy_repo.get(p.id)
        assert got is not None
        assert got.name == "test-policy"
        assert {r.name for r in got.rules} == {"block-low-quality", "block-insecure"}

        got_by_name = policy_repo.get_by_name("test-policy")
        assert got_by_name is not None and got_by_name.id == p.id

        lst = list(policy_repo.list())
        assert any(pol.id == p.id for pol in lst)

        single = rule_repo.get(r1.id)
        assert single is not None and single.name == r1.name
        only_for_p = list(rule_repo.list_by_policy(p.id))
        assert {r.id for r in only_for_p} == {r1.id, r2.id}

        single.description = "updated"
        rule_repo.save(single)
        again = rule_repo.get(single.id)
        assert again is not None and again.description == "updated"

        from dev_guard.domain import Violation, ViolationStatus

        v = Violation(
            rule_id=r1.id,
            policy_id=p.id,
            severity=RuleSeverity.MEDIUM,
            message="violation happened",
            context={"k": "v"},
        )
        violation_repo.save(v)

        open_now = list(violation_repo.list_open())
        assert any(x.id == v.id and x.status == ViolationStatus.OPEN for x in open_now)

        by_policy = list(violation_repo.list_open(policy_id=p.id))
        assert any(x.id == v.id for x in by_policy)

        assert violation_repo.resolve(v.id) is True
        open_after = list(violation_repo.list_open())
        assert all(x.id != v.id for x in open_after)

        assert policy_repo.delete(p.id) is True
        assert policy_repo.get(p.id) is None

