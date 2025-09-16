import importlib

from flashinfer_bench.integration.patch_manager import PatchManager, PatchSpec


def test_patch_manager_function_patch_and_unpatch():
    # Ensure importable
    mod = importlib.import_module("tests.integration.samplemods.pm_dummy")

    pm = PatchManager()

    spec = PatchSpec(
        path="tests.integration.samplemods.pm_dummy.module_function",
        kind="function",
        name="ut_patch",
    )

    called = {"wrapped": False}

    def wf(spec, orig):
        def wrapped(a, b=3):
            called["wrapped"] = True
            return ("wrapped", orig(a, b))

        return wrapped

    assert pm.patch(spec, wf) is True
    # Patched behavior
    assert mod.module_function(4) == ("wrapped", 12)
    assert called["wrapped"] is True

    pm.unpatch_all()
    # Original behavior restored
    assert mod.module_function(4) == 12


def test_patch_manager_method_patch_idempotent():
    mod = importlib.import_module("tests.integration.samplemods.pm_dummy")

    pm = PatchManager()
    spec = PatchSpec(
        path="tests.integration.samplemods.pm_dummy.Foo.instance_method",
        kind="method",
        name="ut_patch_method",
    )

    def wf(spec, orig):
        def wrapped(self, x, y=2):
            return ("meth", orig(self, x, y))

        return wrapped

    # First patch returns True; second should be idempotent and also True
    assert pm.patch(spec, wf) is True
    assert pm.patch(spec, wf) is True

    f = mod.Foo()
    assert f.instance_method(5) == ("meth", 7)

    pm.unpatch_all()
    assert f.instance_method(5) == 7


def test_patch_manager_missing_target_returns_false():
    pm = PatchManager()
    spec = PatchSpec(path="non.existent.module.symbol", kind="function", name="x")
    assert pm.patch(spec, lambda s, o: o) is False
