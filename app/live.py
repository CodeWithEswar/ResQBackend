"""Redis-coordinated live sessions. Redis contains leases, never camera imagery."""
import hashlib
from redis.asyncio import Redis
from redis.exceptions import RedisError


class LiveSessions:
    def __init__(self, url, interval_ms=1000, lease_seconds=45, prefix='resq:live'):
        self.redis = Redis.from_url(url, decode_responses=True, socket_connect_timeout=2, socket_timeout=2,
                                   health_check_interval=30, max_connections=32)
        self.interval_ms = interval_ms
        self.lease_seconds = lease_seconds
        self.prefix = prefix

    def keys(self, user_id):
        digest = hashlib.sha256(user_id.encode()).hexdigest()
        # Common hash tag supports atomic operations on Redis Cluster too.
        root = self.prefix + ':{' + digest + '}'
        return root + ':lease', root + ':pace'

    async def available(self):
        try:
            return bool(await self.redis.ping())
        except RedisError:
            return False

    async def claim(self, user_id, session_id):
        return bool(await self.redis.set(self.keys(user_id)[0], session_id, nx=True, ex=self.lease_seconds))

    async def touch(self, user_id, session_id, frame=False):
        result = await self.redis.eval('''
            if redis.call('GET', KEYS[1]) ~= ARGV[1] then return {-1, 0} end
            redis.call('EXPIRE', KEYS[1], ARGV[2])
            if ARGV[3] == '0' then return {1, 0} end
            if redis.call('SET', KEYS[2], ARGV[1], 'NX', 'PX', ARGV[4]) then return {1, 0} end
            return {0, redis.call('PTTL', KEYS[2])}
        ''', 2, *self.keys(user_id), session_id, self.lease_seconds, int(frame), self.interval_ms)
        return int(result[0]), max(0, int(result[1]))

    async def release(self, user_id, session_id):
        return await self.redis.eval('''
            if redis.call('GET', KEYS[1]) == ARGV[1] then
                return redis.call('DEL', KEYS[1], KEYS[2])
            end
            return 0
        ''', 2, *self.keys(user_id), session_id)

    async def close(self):
        await self.redis.aclose()
