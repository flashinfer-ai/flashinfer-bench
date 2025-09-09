import sys

import pytest

from flashinfer_bench.compile.runnable import Runnable


def test_runnable_single_tuple_unpack_and_close_idempotent():
    calls = {"closed": 0}

    def fn(**kw):
        return (42,)

    def closer():
        calls["closed"] += 1

    r = Runnable(fn=fn, closer=closer, meta={"k": 1})
    assert r() == 42
    # Close twice should not error and closer should be called once
    r.close()
    r.close()
    r.close()
    assert calls["closed"] == 1


if __name__ == "__main__":
    pytest.main(sys.argv)
