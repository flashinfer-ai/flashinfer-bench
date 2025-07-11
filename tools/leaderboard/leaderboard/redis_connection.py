import os
import redis


def create_redis_connection(cert_reqs: str | None = None) -> redis.Redis | None:
    """
    Creates a redis connection using application configuration.
    """
    url = os.getenv('REDIS_URL')
    if url is None:
        return None

    kwargs = {}
    if cert_reqs:
        kwargs['ssl_cert_reqs'] = cert_reqs
    
    return redis.from_url(url, **kwargs)
