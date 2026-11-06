"""Debussy: a System 1 decision model.

    from debussy import Debussy
    d = Debussy()

Heavy imports (llama.cpp) are loaded lazily, so `import debussy.paths` stays light.
"""


def __getattr__(name):
    if name == "Debussy":
        from .model import Debussy
        return Debussy
    if name == "DebussyDuo":
        from .model import DebussyDuo
        return DebussyDuo
    if name == "get_engine":
        from .engine import get_engine
        return get_engine
    raise AttributeError(name)
