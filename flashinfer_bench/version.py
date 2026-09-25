try:
    from flashinfer_bench._version import __version__, __version_tuple__
except Exception:
    __version__ = "0.0.0.dev0"
    __version_tuple__ = (0, 0, 0, "dev0")
