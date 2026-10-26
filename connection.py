"""
Redis connection for the streaming modules, with per-role credentials and optional TLS.

    REDIS_URL=rediss://host:6380/0           TLS when the scheme is rediss://
    REDIS_HOST / REDIS_PORT (default 6391) / REDIS_DB / REDIS_PASSWORD / REDIS_USERNAME
    REDIS_INGESTOR_USERNAME / REDIS_INGESTOR_PASSWORD     (also CONSUMER, REPORT, REPLAY)

A role-specific username/password wins over the generic ones, so each process can run with the
minimum Redis permissions (see streaming/README.md for the ACL rules). Credentials are never logged.
"""
from __future__ import annotations

import os
from typing import Optional

import redis


def connect(role: str = "") -> redis.Redis:
    role = role.upper()
    username = os.getenv(f"REDIS_{role}_USERNAME") or os.getenv("REDIS_USERNAME") or None
    password = os.getenv(f"REDIS_{role}_PASSWORD") or os.getenv("REDIS_PASSWORD") or None
    options = dict(decode_responses=True, socket_connect_timeout=5, socket_timeout=15,
                   health_check_interval=30)
    url: Optional[str] = os.getenv("REDIS_URL")
    if url:
        return redis.Redis.from_url(url, username=username, password=password, **options)
    return redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=int(os.getenv("REDIS_PORT", "6391")),
                       db=int(os.getenv("REDIS_DB", "0")), username=username, password=password, **options)