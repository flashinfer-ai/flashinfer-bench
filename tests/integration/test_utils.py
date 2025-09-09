import inspect

from flashinfer_bench.integration.utils import ArgBinder, ContextStore


def test_arg_binder_bind_with_defaults_and_kwargs():
    def fn(a, b=2, *, c=3):
        return a + b + c

    binder = ArgBinder.from_callable(fn)
    bound = binder.bind(args=(1,), kwargs={"c": 10})
    # Ensure arguments resolved with defaults applied
    assert bound == {"a": 1, "b": 2, "c": 10}


def test_arg_binder_bind_method_like_signature():
    class C:
        def m(self, x, y=5):
            return x + y

    # Use unbound function so signature includes 'self'
    m = C.m
    binder = ArgBinder.from_callable(m)
    # Simulate binding call args including self by passing instance first
    obj = C()
    bound = binder.bind(args=(obj, 7), kwargs={})
    assert bound["x"] == 7 and bound["y"] == 5 and bound.get("self") is obj


def test_context_store_per_instance_isolated_and_mutable():
    store = ContextStore()

    class A:
        pass

    a1, a2 = A(), A()

    d1 = store.get(a1)
    d2 = store.get(a2)
    assert d1 is not d2 and d1 == {} and d2 == {}

    d1["x"] = 42
    assert store.get(a1)["x"] == 42
    assert "x" not in store.get(a2)
