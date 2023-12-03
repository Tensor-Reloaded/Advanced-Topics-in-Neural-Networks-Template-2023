#!/usr/bin/env python3
import gc
from functools import wraps
from time import time


def timed(fn: callable):
    @wraps(fn)
    def wrap(*args, **kwargs):
        gc.collect()
        start = time()
        fn(*args, **kwargs)
        end = time()
        return end - start

    return wrap
