import functools
import multiprocessing


def mutex(func):
    """排他処理をするためのデコレータ"""
    lock = multiprocessing.Lock()

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not lock.acquire(block=True):
            return
        try:
            return func(*args, **kwargs)
        finally:
            lock.release()

    return wrapper
