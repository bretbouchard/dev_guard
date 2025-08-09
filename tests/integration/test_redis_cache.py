import os
import time

import pytest

redis_mod = pytest.importorskip("testcontainers.redis")

from dev_guard.adapters.cache import RedisCache


@pytest.mark.integration
def test_redis_cache_roundtrip():
    from testcontainers.redis import RedisContainer

    with RedisContainer("redis:7-alpine") as rc:
        host = rc.get_container_host_ip()
        port = rc.get_exposed_port(6379)
        url = f"redis://{host}:{port}/0"
        cache = RedisCache(url=url)

        cache.set("foo", "bar")
        assert cache.get("foo") == "bar"

        cache.set("baz", "qux", ttl_seconds=1)
        assert cache.get("baz") == "qux"
        time.sleep(1.2)
        assert cache.get("baz") is None

        cache.delete("foo")
        assert cache.get("foo") is None

