import logging
import sys
from typing import List

import pytest

import flashinfer_bench.logging as fib_logging


@pytest.fixture
def package_logger_state() -> logging.Logger:
    logger = logging.getLogger("flashinfer-bench")
    original_level = logger.level
    original_handlers: List[logging.Handler] = list(logger.handlers)
    original_propagate = logger.propagate

    yield logger

    logger.handlers.clear()
    for handler in original_handlers:
        logger.addHandler(handler)
    logger.setLevel(original_level)
    logger.propagate = original_propagate


def test_get_logger_scopes_name(package_logger_state: logging.Logger) -> None:
    logger = fib_logging.get_logger()
    assert logger.name == "flashinfer-bench"

    scoped = fib_logging.get_logger("bench.tests")
    assert scoped.name == "flashinfer-bench.bench.tests"


def test_configure_logging_with_custom_handler(package_logger_state: logging.Logger) -> None:
    records = []

    class CollectHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = CollectHandler()
    formatter = logging.Formatter("%(levelname)s:%(message)s")

    logger = fib_logging.configure_logging(level="warning", handler=handler, formatter=formatter)

    logger.warning("hello")

    assert logger.level == logging.WARNING
    assert logger.handlers == [handler]
    assert handler.formatter is formatter
    assert records and records[-1].getMessage() == "hello"
    assert not logger.propagate

    handler.close()


def test_configure_logging_rejects_unknown_level(package_logger_state: logging.Logger) -> None:
    with pytest.raises(ValueError):
        fib_logging.configure_logging(level="LOUD")


if __name__ == "__main__":
    pytest.main(sys.argv)
