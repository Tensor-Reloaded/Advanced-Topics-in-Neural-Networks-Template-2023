import time


class Timer:
    __now: float

    def __init__(self) -> None:
        self.__now = time.time()

    def __call__(self) -> float:
        return time.time() - self.__now
