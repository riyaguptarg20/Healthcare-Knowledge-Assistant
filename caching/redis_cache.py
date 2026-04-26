import redis
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

try:
    r = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=False,
        socket_connect_timeout=2,
        socket_timeout=2
    )

    # Test connection
    r.ping()
    logging.info("Redis connected successfully")

except Exception as e:
    logging.error(f"Redis connection failed: {e}")
    r = None


def get_cache(key: str):
    if r is None:
        return None

    try:
        return r.get(key)
    except Exception as e:
        logging.error(f"Redis GET error: {e}")
        return None


def set_cache(key: str, value: str, ttl: int = 3600):
    if r is None:
        return

    try:
        r.set(key, value.encode(), ex=ttl)
    except Exception as e:
        logging.error(f"Redis SET error: {e}")