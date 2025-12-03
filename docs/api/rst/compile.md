# flashinfer_bench.compile

`flashinfer_bench.compile` provides infrastructure for building solutions into executable runnables.

The typical workflow is:

1. Get the singleton registry: `registry = BuilderRegistry.get_instance()`
2. Build a solution: `runnable = registry.build(definition, solution)`
3. Execute: `result = runnable(**inputs)`

## Registry

```{eval-rst}
.. currentmodule:: flashinfer_bench.compile

.. autoclass:: BuilderRegistry
   :members:
```

## Builder

```{eval-rst}
.. autoclass:: Builder
   :members:

.. autoexception:: BuildError
```

## Runnable

```{eval-rst}
.. autoclass:: Runnable
   :members:

.. autoclass:: RunnableMetadata
   :members:
```

## Concrete Builders

```{eval-rst}
.. autoclass:: flashinfer_bench.compile.builders.PythonBuilder
   :members:
   :show-inheritance:

.. autoclass:: flashinfer_bench.compile.builders.TritonBuilder
   :members:
   :show-inheritance:

.. autoclass:: flashinfer_bench.compile.builders.TVMFFIBuilder
   :members:
   :show-inheritance:

.. autoclass:: flashinfer_bench.compile.builders.TorchBuilder
   :members:
   :show-inheritance:
```
