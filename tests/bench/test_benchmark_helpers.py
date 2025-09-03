import pytest

from flashinfer_bench.bench.benchmark import Benchmark
from flashinfer_bench.data.traceset import TraceSet


def test_benchmark_pick_runners_round_robin(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "flashinfer_bench.bench.benchmark.list_cuda_devices",
        lambda: ["dev0", "dev1", "dev2"],
    )

    ts = TraceSet(root=tmp_path)
    b = Benchmark(ts)

    # Inject known runners (opaque objects are fine for selection logic)
    b._runners = [object(), object(), object()]
    b._curr_device_idx = 0

    sel1 = b._pick_runners(2)
    assert sel1 == [b._runners[0], b._runners[1]]
    sel2 = b._pick_runners(2)
    assert sel2 == [b._runners[2], b._runners[0]]
    sel3 = b._pick_runners(1)
    assert sel3 == [b._runners[1]]
    # K <= 0 returns empty
    assert b._pick_runners(0) == []
