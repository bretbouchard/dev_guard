import os
import time

import pytest

redis_mod = pytest.importorskip("testcontainers.redis")

from dev_guard.adapters.cache import RedisCache


@pytest.mark.integration
def test_redis_cache_roundtrip():
    from testcontainers.redis import RedisContainer

    with RedisContainer("redis:7-alpine") as rc:
        url = rc.get_connection_url()
        cache = RedisCache(url=url)

        cache.set("foo", "bar")
        assert cache.get("foo") == "bar"

        cache.set("baz", "qux", ttl_seconds=1)
        assert cache.get("baz") == "qux"
        time.sleep(1.2)
        assert cache.get("baz") is None

        cache.delete("foo")
        assert cache.get("foo") is None

