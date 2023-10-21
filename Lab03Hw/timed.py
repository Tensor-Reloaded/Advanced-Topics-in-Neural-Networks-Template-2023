import torch
from torch import Tensor
from time import time
import gc
from functools import wraps

# This can be used for everything.
def timedBasic(fn):
    @wraps(fn)
    def wrap(*args, **kwargs):
        gc.collect()
        start = time()
        ret = fn(*args, **kwargs)
        end = time()
        sizes = [f"{x.shape}" for x in args if isinstance(x, Tensor)]
        try:
            name = fn.__name__
        except:
            name =  "traced"
        print(name, f"took {end - start} for input of size", ", ".join(sizes))
        return ret
    return wrap

# Better, but is meant to be used with pytorch and cuda
def timedCuda(fn):
    @wraps(fn)
    def wrap(*args, **kwargs):
        gc.collect()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        ret = fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        sizes = [f"{x.shape}" for x in args if isinstance(x, Tensor)]
        try:
            name = fn.__name__
        except:
            name =  "traced"
        print(name, f"took {start.elapsed_time(end)} for input of size", ", ".join(sizes))
        return ret
    return wrap