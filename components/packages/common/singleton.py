import threading


lock = threading.Lock()


class SingletonMeta(type):
    _instance = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instance:
            with lock:
                cls._instance[cls] = super(SingletonMeta, cls).__call__(
                    *args, **kwargs
                )
        return cls._instance[cls]
