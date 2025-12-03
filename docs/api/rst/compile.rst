flashinfer_bench.compile
========================

.. currentmodule:: flashinfer_bench.compile

``flashinfer_bench.compile`` provides infrastructure for building solutions into executable runnables.

The typical workflow is:

1. Get the singleton registry: ``registry = BuilderRegistry.get_instance()``
2. Build a solution: ``runnable = registry.build(definition, solution)``
3. Execute: ``result = runnable(**inputs)``

Registry
--------

.. autoclass:: BuilderRegistry
   :members:

Builder
-------

.. autoclass:: Builder
   :members:

.. autoexception:: BuildError

Runnable
--------

.. autoclass:: Runnable
   :members:

.. autoclass:: RunnableMetadata
   :members:

Concrete Builders
-----------------

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
