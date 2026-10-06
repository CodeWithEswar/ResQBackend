"""Live session coordination. Supports Redis cluster/standalone with seamless in-memory fallback."""
import asyncio
import hashlib
import logging
import os
import time
from redis.asyncio import Redis
from redis.exceptions import RedisError

log = logging.getLogger('resq.live')


class LiveSessions:
    def __init__(self, url, interval_ms=1000, lease_seconds=45, prefix='resq:live', allow_memory_fallback=None):
        self.url = url
        self.interval_ms = interval_ms
        self.lease_seconds = lease_seconds
        self.prefix = prefix
        if allow_memory_fallback is None:
            self.allow_memory_fallback = os.getenv('LIVE_MEMORY_FALLBACK', '1') == '1'
        else:
            self.allow_memory_fallback = allow_memory_fallback
        self.redis = Redis.from_url(url, decode_responses=True, socket_connect_timeout=2, socket_timeout=2,
                                   health_check_interval=30, max_connections=32)
        self._memory_leases: dict[str, tuple[str, float]] = {}
        self._memory_paces: dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._redis_connected = False

    def keys(self, user_id):
        digest = hashlib.sha256(user_id.encode()).hexdigest()
        root = self.prefix + ':{' + digest + '}'
        return root + ':lease', root + ':pace'

    @property
    def is_redis(self):
        return self._redis_connected

    @property
    def coordination(self):
        return 'redis' if self._redis_connected else 'memory'

    async def is_redis_available(self):
        try:
            ok = bool(await self.redis.ping())
            self._redis_connected = ok
            return ok
        except (RedisError, Exception):
            self._redis_connected = False
            return False

    async def available(self):
        if await self.is_redis_available():
            return True
        return self.allow_memory_fallback

    async def claim(self, user_id, session_id):
        if await self.is_redis_available():
            try:
                return bool(await self.redis.set(self.keys(user_id)[0], session_id, nx=True, ex=self.lease_seconds))
            except RedisError:
                if not self.allow_memory_fallback:
                    raise
                self._redis_connected = False
                log.warning('Redis claim failed; falling back to in-memory coordination.')
        elif not self.allow_memory_fallback:
            raise RedisError('Redis connection unavailable and memory fallback disabled.')

        async with self._lock:
            now = time.monotonic()
            current = self._memory_leases.get(user_id)
            if current and current[1] > now and current[0] != session_id:
                return False
            self._memory_leases[user_id] = (session_id, now + self.lease_seconds)
            return True

    async def touch(self, user_id, session_id, frame=False):
        if self._redis_connected:
            try:
                result = await self.redis.eval('''
                    if redis.call('GET', KEYS[1]) ~= ARGV[1] then return {-1, 0} end
                    redis.call('EXPIRE', KEYS[1], ARGV[2])
                    if ARGV[3] == '0' then return {1, 0} end
                    if redis.call('SET', KEYS[2], ARGV[1], 'NX', 'PX', ARGV[4]) then return {1, 0} end
                    return {0, redis.call('PTTL', KEYS[2])}
                ''', 2, *self.keys(user_id), session_id, self.lease_seconds, int(frame), self.interval_ms)
                return int(result[0]), max(0, int(result[1]))
            except RedisError:
                if not self.allow_memory_fallback:
                    raise
                self._redis_connected = False
                log.warning('Redis touch failed; falling back to in-memory coordination.')
        elif not self.allow_memory_fallback:
            raise RedisError('Redis connection unavailable and memory fallback disabled.')

        async with self._lock:
            now = time.monotonic()
            current = self._memory_leases.get(user_id)
            if not current or current[1] <= now or current[0] != session_id:
                return -1, 0
            self._memory_leases[user_id] = (session_id, now + self.lease_seconds)
            if not frame:
                return 1, 0
            next_allowed = self._memory_paces.get(user_id, 0.0)
            if now < next_allowed:
                retry_ms = max(0, int((next_allowed - now) * 1000))
                return 0, retry_ms
            self._memory_paces[user_id] = now + (self.interval_ms / 1000.0)
            return 1, 0

    async def release(self, user_id, session_id):
        if self._redis_connected:
            try:
                return await self.redis.eval('''
                    if redis.call('GET', KEYS[1]) == ARGV[1] then
                        return redis.call('DEL', KEYS[1], KEYS[2])
                    end
                    return 0
                ''', 2, *self.keys(user_id), session_id)
            except RedisError:
                if not self.allow_memory_fallback:
                    raise
                self._redis_connected = False
        async with self._lock:
            current = self._memory_leases.get(user_id)
            if current and current[0] == session_id:
                self._memory_leases.pop(user_id, None)
                self._memory_paces.pop(user_id, None)
                return 1
            return 0

    async def close(self):
        try:
            await self.redis.aclose()
        except Exception:
            pass
