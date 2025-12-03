import sys

import pytest

from flashinfer_bench.compile.runnable import Runnable, RunnableMetadata


def test_runnable_single_tuple_unpack_and_close_idempotent():
    calls = {"closed": 0}

    def fn(**kw):
        return (42,)

    def closer():
        calls["closed"] += 1

    metadata = RunnableMetadata(
        build_type="python", definition="test", solution="test", misc={"k": 1}
    )

    r = Runnable(callable=fn, cleaner=closer, metadata=metadata)
    assert r() == 42
    # Close twice should not error and closer should be called once
    r.cleanup()
    r.cleanup()
    r.cleanup()
    assert calls["closed"] == 1


if __name__ == "__main__":
    pytest.main(sys.argv)
