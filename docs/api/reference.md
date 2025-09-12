<a id="flashinfer_bench.apply"></a>

# flashinfer\_bench.apply

<a id="flashinfer_bench.apply.apply"></a>

## apply

<a id="flashinfer_bench.apply.disable_apply"></a>

## disable\_apply

<a id="flashinfer_bench.apply.enable_apply"></a>

## enable\_apply

<a id="flashinfer_bench.apply.get_runtime"></a>

## get\_runtime

<a id="flashinfer_bench.apply.__all__"></a>

#### \_\_all\_\_

<a id="flashinfer_bench.apply.api"></a>

# flashinfer\_bench.apply.api

<a id="flashinfer_bench.apply.api.annotations"></a>

## annotations

<a id="flashinfer_bench.apply.api.inspect"></a>

## inspect

<a id="flashinfer_bench.apply.api.os"></a>

## os

<a id="flashinfer_bench.apply.api.Any"></a>

## Any

<a id="flashinfer_bench.apply.api.Callable"></a>

## Callable

<a id="flashinfer_bench.apply.api.Dict"></a>

## Dict

<a id="flashinfer_bench.apply.api.Mapping"></a>

## Mapping

<a id="flashinfer_bench.apply.api.Optional"></a>

## Optional

<a id="flashinfer_bench.apply.api.Union"></a>

## Union

<a id="flashinfer_bench.apply.api.overload"></a>

## overload

<a id="flashinfer_bench.apply.api.ApplyConfig"></a>

## ApplyConfig

<a id="flashinfer_bench.apply.api.ApplyRuntime"></a>

## ApplyRuntime

<a id="flashinfer_bench.apply.api.get_runtime"></a>

## get\_runtime

<a id="flashinfer_bench.apply.api.set_runtime"></a>

## set\_runtime

<a id="flashinfer_bench.apply.api._SENTINEL"></a>

#### \_SENTINEL

<a id="flashinfer_bench.apply.api.apply"></a>

#### apply

```python
@overload
def apply(
    def_name_or_resolver: Union[str, Callable[..., str]]
) -> Callable[[Callable[..., Any]], Callable[..., Any]]
```

<a id="flashinfer_bench.apply.api.apply"></a>

#### apply

```python
@overload
def apply(def_name_or_resolver: Union[str, Callable[..., str]], *,
          runtime_kwargs: Dict[str, Any],
          fallback: Optional[Callable[..., Any]]) -> Any
```

<a id="flashinfer_bench.apply.api.apply"></a>

#### apply

```python
def apply(def_name_or_resolver: Union[str, Callable[..., str]],
          runtime_kwargs: Dict[str, Any] = _SENTINEL,
          fallback: Optional[Callable[..., Any]] = _SENTINEL)
```

Decorator/function for routing to the best-performing kernel recorded in the
FlashInfer Trace database.

This API can be used in two modes:

1) **Decorator mode** (only ``def_name_or_resolver`` provided): returns a decorator
   that wraps a kernel function with a router. The router selects the best-performing
   candidate according to the function's runtime arguments.
2) **Function mode** (``runtime_kwargs`` provided, optionally ``fallback``):
   immediately resolves and calls the best-performing kernel and returns its result.

Parameters
----------
def_name_or_resolver : Union[str, Callable[..., str]]
    The kernel name, or a resolver ``fn(*args, **kwargs) -> str`` that maps runtime
    arguments to a kernel name (definition name).
runtime_kwargs : Dict[str, Any], optional
    Only used in **function mode**. The runtime arguments to feed into the selected
    kernel. Use this to call the kernel immediately instead of returning a decorator.
fallback : Optional[Callable[..., Any]], optional
    Only used in **function mode**. A fallback function to invoke when no matching
    kernel is found in the Trace database.

Returns
-------
Union[Callable[[Callable[..., Any]], Callable[..., Any]], Any]
    - **Decorator mode**: a decorator that transforms the target kernel function into
      a routed version.
    - **Function mode**: the return value produced by the selected (or fallback) kernel.

Examples
--------
Decorator mode with a fixed name
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
>>> @apply("gemm_bf16")
... def gemm_bf16(A, B, bias=None):
...     return torch.nn.functional.linear(A, B, bias)

Decorator mode with a resolver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
>>> @apply(lambda A, B: f"gemm_n_{B.shape[0]}_k_{B.shape[1]}")
... def gemm_bf16(A, B, bias=None):
...     return torch.nn.functional.linear(A, B, bias)

Function mode
~~~~~~~~~~~~~
>>> out = apply(
...     "gemm_bf16",
...     runtime_kwargs={"A": A, "B": B, "bias": None},
...     fallback=lambda **kw: torch.nn.functional.linear(**kw),
... )

<a id="flashinfer_bench.apply.api.enable_apply"></a>

#### enable\_apply

```python
def enable_apply(dataset_path: Optional[str] = None,
                 apply_config: Optional[ApplyConfig] = None) -> _ApplyHandle
```

Immediately enable global apply and return a handle.

The returned handle can be used in two ways:
- As a function for imperative apply calls
- In a with block for contextual availability, exiting restores the original state

Parameters
----------
dataset_path : str, optional
    Path to the dataset/traceset directory
apply_config : ApplyConfig, optional
    Configuration for the apply runtime

Returns
-------
_ApplyHandle
    Handle that can be used for imperative apply or as context manager

Examples
--------
>>> # Direct usage
>>> handle = enable_apply("/path/to/traceset", cfg)
>>> out = apply("rmsnorm_d4096", runtime_kwargs={...}, fallback=ref_fn)

>>> # Context manager usage
>>> with enable_apply("/path/to/traceset", cfg) as apply_fn:
...     out = apply_fn("rmsnorm_d4096", runtime_kwargs={...}, fallback=ref_fn)

<a id="flashinfer_bench.apply.api.disable_apply"></a>

#### disable\_apply

```python
def disable_apply() -> None
```

Disable global apply functionality.

This function silently disables the global apply runtime by setting it to None.
After calling this function, any subsequent calls to apply() will use fallback
functions instead of the apply runtime.

Examples
--------
>>> enable_apply("/path/to/traceset")
>>> # apply is now enabled
>>> disable_apply()
>>> # apply is now disabled, fallback functions will be used

<a id="flashinfer_bench.apply.api._ApplyHandle"></a>

## \_ApplyHandle Objects

```python
class _ApplyHandle()
```

Context manager for enabling apply.

<a id="flashinfer_bench.apply.api._ApplyHandle.__init__"></a>

#### \_\_init\_\_

```python
def __init__(dataset_path: Optional[str] = None,
             config: Optional[ApplyConfig] = None) -> None
```

<a id="flashinfer_bench.apply.api._ApplyHandle.__call__"></a>

#### \_\_call\_\_

```python
def __call__(def_name_or_resolver: Union[str, Callable[..., str]], *,
             runtime_kwargs: Dict[str, Any],
             fallback: Optional[Callable[..., Any]]) -> Any
```

<a id="flashinfer_bench.apply.api._ApplyHandle.__enter__"></a>

#### \_\_enter\_\_

```python
def __enter__() -> Callable[[Union[str, Callable[..., str]]], Any]
```

<a id="flashinfer_bench.apply.api._ApplyHandle.__exit__"></a>

#### \_\_exit\_\_

```python
def __exit__(exc_type, exc, tb) -> bool
```

<a id="flashinfer_bench.apply.api._merge_to_kwargs"></a>

#### \_merge\_to\_kwargs

```python
def _merge_to_kwargs(param_names: tuple[str, ...], args: tuple[Any, ...],
                     kwargs: Mapping[str, Any]) -> Dict[str, Any]
```

<a id="flashinfer_bench.apply.api._resolve_dataset"></a>

#### \_resolve\_dataset

```python
def _resolve_dataset(dataset_path: Optional[str]) -> str
```

<a id="flashinfer_bench.apply.api._resolve_cfg"></a>

#### \_resolve\_cfg

```python
def _resolve_cfg(cfg: Optional[ApplyConfig]) -> ApplyConfig
```

<a id="flashinfer_bench.apply.config"></a>

# flashinfer\_bench.apply.config

<a id="flashinfer_bench.apply.config.dataclass"></a>

## dataclass

<a id="flashinfer_bench.apply.config.Literal"></a>

## Literal

<a id="flashinfer_bench.apply.config.ApplyConfig"></a>

## ApplyConfig Objects

```python
@dataclass(frozen=True)
class ApplyConfig()
```

<a id="flashinfer_bench.apply.config.ApplyConfig.max_atol"></a>

#### max\_atol

<a id="flashinfer_bench.apply.config.ApplyConfig.max_rtol"></a>

#### max\_rtol

<a id="flashinfer_bench.apply.config.ApplyConfig.aot_ratio"></a>

#### aot\_ratio

<a id="flashinfer_bench.apply.config.ApplyConfig.on_miss_policy"></a>

#### on\_miss\_policy

<a id="flashinfer_bench.apply.config.ApplyConfig.__post_init__"></a>

#### \_\_post\_init\_\_

```python
def __post_init__() -> None
```

<a id="flashinfer_bench.apply.hook"></a>

# flashinfer\_bench.apply.hook

<a id="flashinfer_bench.apply.hook.Any"></a>

## Any

<a id="flashinfer_bench.apply.hook.Callable"></a>

## Callable

<a id="flashinfer_bench.apply.hook.Mapping"></a>

## Mapping

<a id="flashinfer_bench.apply.hook.Optional"></a>

## Optional

<a id="flashinfer_bench.apply.hook._hook"></a>

#### \_hook

<a id="flashinfer_bench.apply.hook.set_apply_hook"></a>

#### set\_apply\_hook

```python
def set_apply_hook(
        fn: Optional[Callable[[str, Mapping[str, Any]], None]]) -> None
```

<a id="flashinfer_bench.apply.hook.get_apply_hook"></a>

#### get\_apply\_hook

```python
def get_apply_hook() -> Callable[[str, Mapping[str, Any]], None]
```

<a id="flashinfer_bench.apply.key"></a>

# flashinfer\_bench.apply.key

<a id="flashinfer_bench.apply.key.annotations"></a>

## annotations

<a id="flashinfer_bench.apply.key.json"></a>

## json

<a id="flashinfer_bench.apply.key.ABC"></a>

## ABC

<a id="flashinfer_bench.apply.key.abstractmethod"></a>

## abstractmethod

<a id="flashinfer_bench.apply.key.dataclass"></a>

## dataclass

<a id="flashinfer_bench.apply.key.field"></a>

## field

<a id="flashinfer_bench.apply.key.Any"></a>

## Any

<a id="flashinfer_bench.apply.key.Dict"></a>

## Dict

<a id="flashinfer_bench.apply.key.Tuple"></a>

## Tuple

<a id="flashinfer_bench.apply.key.Type"></a>

## Type

<a id="flashinfer_bench.apply.key.Union"></a>

## Union

<a id="flashinfer_bench.apply.key.AxisVar"></a>

## AxisVar

<a id="flashinfer_bench.apply.key.Definition"></a>

## Definition

<a id="flashinfer_bench.apply.key.Workload"></a>

## Workload

<a id="flashinfer_bench.apply.key.ApplyKey"></a>

## ApplyKey Objects

```python
@dataclass(frozen=True)
class ApplyKey()
```

<a id="flashinfer_bench.apply.key.ApplyKey.axes"></a>

#### axes

<a id="flashinfer_bench.apply.key.ApplyKey.feats"></a>

#### feats

<a id="flashinfer_bench.apply.key.ApplyKey.encode"></a>

#### encode

```python
@property
def encode() -> str
```

<a id="flashinfer_bench.apply.key.ApplyKey.from_encoded"></a>

#### from\_encoded

```python
@classmethod
def from_encoded(cls, s: str) -> "ApplyKey"
```

<a id="flashinfer_bench.apply.key.ApplyKey.__hash__"></a>

#### \_\_hash\_\_

```python
def __hash__() -> int
```

<a id="flashinfer_bench.apply.key.ApplyKey.__eq__"></a>

#### \_\_eq\_\_

```python
def __eq__(other: Any) -> bool
```

<a id="flashinfer_bench.apply.key.ApplyKeyBuilder"></a>

## ApplyKeyBuilder Objects

```python
class ApplyKeyBuilder(ABC)
```

<a id="flashinfer_bench.apply.key.ApplyKeyBuilder.__init__"></a>

#### \_\_init\_\_

```python
def __init__(defn: Definition) -> None
```

<a id="flashinfer_bench.apply.key.ApplyKeyBuilder.build_from_runtime"></a>

#### build\_from\_runtime

```python
@abstractmethod
def build_from_runtime(runtime_kwargs: Dict[str, Any]) -> ApplyKey
```

Build a key from runtime kwargs

<a id="flashinfer_bench.apply.key.ApplyKeyBuilder.build_from_workload"></a>

#### build\_from\_workload

```python
@abstractmethod
def build_from_workload(workload: Workload) -> ApplyKey
```

Build a key from offline workload trace

<a id="flashinfer_bench.apply.key.ApplyKeyBuilder.features"></a>

#### features

```python
@abstractmethod
def features(runtime_kwargs: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]
```

Lightweight feature extraction

<a id="flashinfer_bench.apply.key.ApplyKeyBuilder._collect_var_axis_projections"></a>

#### \_collect\_var\_axis\_projections

```python
def _collect_var_axis_projections(
        defn: Definition) -> Dict[str, Tuple[str, int]]
```

Iterate over the shape of inputs, find the first occurrence of each var axis:
  axis_name -> (input_name, dim_idx)

<a id="flashinfer_bench.apply.key.ApplyKeyBuilder._materialize_axes"></a>

#### \_materialize\_axes

```python
def _materialize_axes(runtime_kwargs: Dict[str, Any]) -> Dict[str, int]
```

<a id="flashinfer_bench.apply.key.AxesOnlyKeyBuilder"></a>

## AxesOnlyKeyBuilder Objects

```python
class AxesOnlyKeyBuilder(ApplyKeyBuilder)
```

<a id="flashinfer_bench.apply.key.AxesOnlyKeyBuilder.build_from_runtime"></a>

#### build\_from\_runtime

```python
def build_from_runtime(runtime_kwargs: Dict[str, Any]) -> ApplyKey
```

<a id="flashinfer_bench.apply.key.AxesOnlyKeyBuilder.build_from_workload"></a>

#### build\_from\_workload

```python
def build_from_workload(workload: Workload) -> ApplyKey
```

<a id="flashinfer_bench.apply.key.AxesOnlyKeyBuilder.features"></a>

#### features

```python
def features(runtime_kwargs: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]
```

<a id="flashinfer_bench.apply.key.GEMMKeyBuilder"></a>

## GEMMKeyBuilder Objects

```python
class GEMMKeyBuilder(AxesOnlyKeyBuilder)
```

<a id="flashinfer_bench.apply.key.GQAKeyBuilder"></a>

## GQAKeyBuilder Objects

```python
class GQAKeyBuilder(AxesOnlyKeyBuilder)
```

<a id="flashinfer_bench.apply.key.MLAKeyBuilder"></a>

## MLAKeyBuilder Objects

```python
class MLAKeyBuilder(AxesOnlyKeyBuilder)
```

<a id="flashinfer_bench.apply.key.ApplyKeyFactory"></a>

## ApplyKeyFactory Objects

```python
class ApplyKeyFactory()
```

<a id="flashinfer_bench.apply.key.ApplyKeyFactory._REGISTRY"></a>

#### \_REGISTRY

<a id="flashinfer_bench.apply.key.ApplyKeyFactory.register"></a>

#### register

```python
@classmethod
def register(cls, type_name: str, builder_cls: Type[ApplyKeyBuilder]) -> None
```

<a id="flashinfer_bench.apply.key.ApplyKeyFactory.for_type"></a>

#### for\_type

```python
@classmethod
def for_type(cls, type_name: str) -> Type[ApplyKeyBuilder]
```

<a id="flashinfer_bench.apply.key.ApplyKeyFactory.specialize"></a>

#### specialize

```python
@classmethod
def specialize(cls, defn: Definition) -> ApplyKeyBuilder
```

<a id="flashinfer_bench.apply.runtime"></a>

# flashinfer\_bench.apply.runtime

<a id="flashinfer_bench.apply.runtime.annotations"></a>

## annotations

<a id="flashinfer_bench.apply.runtime.os"></a>

## os

<a id="flashinfer_bench.apply.runtime.Any"></a>

## Any

<a id="flashinfer_bench.apply.runtime.Callable"></a>

## Callable

<a id="flashinfer_bench.apply.runtime.Dict"></a>

## Dict

<a id="flashinfer_bench.apply.runtime.Mapping"></a>

## Mapping

<a id="flashinfer_bench.apply.runtime.Optional"></a>

## Optional

<a id="flashinfer_bench.apply.runtime.Union"></a>

## Union

<a id="flashinfer_bench.apply.runtime.get_registry"></a>

## get\_registry

<a id="flashinfer_bench.apply.runtime.TraceSet"></a>

## TraceSet

<a id="flashinfer_bench.apply.runtime.ApplyConfig"></a>

## ApplyConfig

<a id="flashinfer_bench.apply.runtime.get_apply_hook"></a>

## get\_apply\_hook

<a id="flashinfer_bench.apply.runtime.ApplyKeyBuilder"></a>

## ApplyKeyBuilder

<a id="flashinfer_bench.apply.runtime.ApplyKeyFactory"></a>

## ApplyKeyFactory

<a id="flashinfer_bench.apply.runtime.ApplyTable"></a>

## ApplyTable

<a id="flashinfer_bench.apply.runtime._runtime"></a>

#### \_runtime

<a id="flashinfer_bench.apply.runtime.get_runtime"></a>

#### get\_runtime

```python
def get_runtime() -> "ApplyRuntime"
```

<a id="flashinfer_bench.apply.runtime.set_runtime"></a>

#### set\_runtime

```python
def set_runtime(rt: Optional["ApplyRuntime"]) -> None
```

<a id="flashinfer_bench.apply.runtime._maybe_init_from_env"></a>

#### \_maybe\_init\_from\_env

```python
def _maybe_init_from_env() -> None
```

<a id="flashinfer_bench.apply.runtime.ApplyRuntime"></a>

## ApplyRuntime Objects

```python
class ApplyRuntime()
```

<a id="flashinfer_bench.apply.runtime.ApplyRuntime.__init__"></a>

#### \_\_init\_\_

```python
def __init__(traceset: Union[TraceSet, str], config: ApplyConfig) -> None
```

<a id="flashinfer_bench.apply.runtime.ApplyRuntime.rebuild"></a>

#### rebuild

```python
def rebuild(traceset: Optional[Union[TraceSet, str]] = None,
            config: Optional[ApplyConfig] = None) -> None
```

<a id="flashinfer_bench.apply.runtime.ApplyRuntime.dispatch"></a>

#### dispatch

```python
def dispatch(def_name: str, runtime_kwargs: Mapping[str, Any],
             fallback: Optional[Callable[..., Any]]) -> Any
```

<a id="flashinfer_bench.apply.table"></a>

# flashinfer\_bench.apply.table

<a id="flashinfer_bench.apply.table.annotations"></a>

## annotations

<a id="flashinfer_bench.apply.table.hashlib"></a>

## hashlib

<a id="flashinfer_bench.apply.table.json"></a>

## json

<a id="flashinfer_bench.apply.table.os"></a>

## os

<a id="flashinfer_bench.apply.table.Counter"></a>

## Counter

<a id="flashinfer_bench.apply.table.dataclass"></a>

## dataclass

<a id="flashinfer_bench.apply.table.field"></a>

## field

<a id="flashinfer_bench.apply.table.Path"></a>

## Path

<a id="flashinfer_bench.apply.table.Any"></a>

## Any

<a id="flashinfer_bench.apply.table.Dict"></a>

## Dict

<a id="flashinfer_bench.apply.table.List"></a>

## List

<a id="flashinfer_bench.apply.table.Optional"></a>

## Optional

<a id="flashinfer_bench.apply.table.Tuple"></a>

## Tuple

<a id="flashinfer_bench.apply.table.get_registry"></a>

## get\_registry

<a id="flashinfer_bench.apply.table.Runnable"></a>

## Runnable

<a id="flashinfer_bench.apply.table.dataclass_to_dict"></a>

## dataclass\_to\_dict

<a id="flashinfer_bench.apply.table.Trace"></a>

## Trace

<a id="flashinfer_bench.apply.table.TraceSet"></a>

## TraceSet

<a id="flashinfer_bench.apply.table.ApplyConfig"></a>

## ApplyConfig

<a id="flashinfer_bench.apply.table.ApplyKey"></a>

## ApplyKey

<a id="flashinfer_bench.apply.table.ApplyKeyFactory"></a>

## ApplyKeyFactory

<a id="flashinfer_bench.apply.table._cache_root"></a>

#### \_cache\_root

```python
def _cache_root() -> Path
```

<a id="flashinfer_bench.apply.table._apply_table_dir"></a>

#### \_apply\_table\_dir

```python
def _apply_table_dir() -> Path
```

<a id="flashinfer_bench.apply.table.ApplyTable"></a>

## ApplyTable Objects

```python
@dataclass
class ApplyTable()
```

<a id="flashinfer_bench.apply.table.ApplyTable.digest"></a>

#### digest

<a id="flashinfer_bench.apply.table.ApplyTable.index"></a>

#### index

<a id="flashinfer_bench.apply.table.ApplyTable.def_best"></a>

#### def\_best

<a id="flashinfer_bench.apply.table.ApplyTable.load_or_build"></a>

#### load\_or\_build

```python
@classmethod
def load_or_build(cls, ts: TraceSet, config: ApplyConfig) -> "ApplyTable"
```

<a id="flashinfer_bench.apply.table.ApplyTable._build"></a>

#### \_build

```python
@classmethod
def _build(cls, ts: TraceSet, config: ApplyConfig) -> "ApplyTable"
```

<a id="flashinfer_bench.apply.table.ApplyTable._sweep_def"></a>

#### \_sweep\_def

```python
@classmethod
def _sweep_def(
        cls, ts: TraceSet, def_name: str, max_atol: float, max_rtol: float
) -> Tuple[Dict[ApplyKey, Trace], List[Tuple[str, int]]]
```

<a id="flashinfer_bench.apply.table.ApplyTable._prewarm_aot"></a>

#### \_prewarm\_aot

```python
@classmethod
def _prewarm_aot(cls, ts: TraceSet, config: ApplyConfig,
                 table: "ApplyTable") -> None
```

<a id="flashinfer_bench.apply.table.ApplyTable._digest"></a>

#### \_digest

```python
@classmethod
def _digest(cls, ts: TraceSet, config: ApplyConfig) -> str
```

<a id="flashinfer_bench.apply.table.ApplyTable.match_solution"></a>

#### match\_solution

```python
def match_solution(def_name: str, key: ApplyKey) -> Optional[str]
```

<a id="flashinfer_bench.bench"></a>

# flashinfer\_bench.bench

<a id="flashinfer_bench.bench.annotations"></a>

## annotations

<a id="flashinfer_bench.bench.Benchmark"></a>

## Benchmark

<a id="flashinfer_bench.bench.BenchmarkConfig"></a>

## BenchmarkConfig

<a id="flashinfer_bench.bench.__all__"></a>

#### \_\_all\_\_

<a id="flashinfer_bench.bench.benchmark"></a>

# flashinfer\_bench.bench.benchmark

<a id="flashinfer_bench.bench.benchmark.annotations"></a>

## annotations

<a id="flashinfer_bench.bench.benchmark.logging"></a>

## logging

<a id="flashinfer_bench.bench.benchmark.shutil"></a>

## shutil

<a id="flashinfer_bench.bench.benchmark.defaultdict"></a>

## defaultdict

<a id="flashinfer_bench.bench.benchmark.ThreadPoolExecutor"></a>

## ThreadPoolExecutor

<a id="flashinfer_bench.bench.benchmark.datetime"></a>

## datetime

<a id="flashinfer_bench.bench.benchmark.Dict"></a>

## Dict

<a id="flashinfer_bench.bench.benchmark.List"></a>

## List

<a id="flashinfer_bench.bench.benchmark.get_registry"></a>

## get\_registry

<a id="flashinfer_bench.bench.benchmark.append_jsonl_lines"></a>

## append\_jsonl\_lines

<a id="flashinfer_bench.bench.benchmark.Evaluation"></a>

## Evaluation

<a id="flashinfer_bench.bench.benchmark.EvaluationStatus"></a>

## EvaluationStatus

<a id="flashinfer_bench.bench.benchmark.Trace"></a>

## Trace

<a id="flashinfer_bench.bench.benchmark.TraceSet"></a>

## TraceSet

<a id="flashinfer_bench.bench.benchmark.list_cuda_devices"></a>

## list\_cuda\_devices

<a id="flashinfer_bench.bench.benchmark.BenchmarkConfig"></a>

## BenchmarkConfig

<a id="flashinfer_bench.bench.benchmark.BaselineHandle"></a>

## BaselineHandle

<a id="flashinfer_bench.bench.benchmark.Runner"></a>

## Runner

<a id="flashinfer_bench.bench.benchmark.MultiProcessRunner"></a>

## MultiProcessRunner

<a id="flashinfer_bench.bench.benchmark.Benchmark"></a>

## Benchmark Objects

```python
class Benchmark()
```

<a id="flashinfer_bench.bench.benchmark.Benchmark.__init__"></a>

#### \_\_init\_\_

```python
def __init__(trace_set: TraceSet, log_level: str = "INFO") -> None
```

<a id="flashinfer_bench.bench.benchmark.Benchmark._pick_runners"></a>

#### \_pick\_runners

```python
def _pick_runners(K: int) -> list[Runner]
```

<a id="flashinfer_bench.bench.benchmark.Benchmark._relaunch_runner"></a>

#### \_relaunch\_runner

```python
def _relaunch_runner(device: str) -> Runner
```

<a id="flashinfer_bench.bench.benchmark.Benchmark._handle_failed_runners"></a>

#### \_handle\_failed\_runners

```python
def _handle_failed_runners(failed_runners: List[Runner]) -> None
```

<a id="flashinfer_bench.bench.benchmark.Benchmark.run"></a>

#### run

```python
def run(config: BenchmarkConfig = BenchmarkConfig()) -> None
```

<a id="flashinfer_bench.bench.benchmark.Benchmark.evaluate"></a>

#### evaluate

```python
def evaluate(config: BenchmarkConfig = BenchmarkConfig()) -> TraceSet
```

Evaluate solutions and return a TraceSet with results immediately.
Used for small TraceSets that need immediate feedback.

<a id="flashinfer_bench.bench.benchmark.Benchmark._ensure_archive"></a>

#### \_ensure\_archive

```python
def _ensure_archive() -> None
```

<a id="flashinfer_bench.bench.benchmark.Benchmark.flush"></a>

#### flush

```python
def flush() -> None
```

<a id="flashinfer_bench.bench.config"></a>

# flashinfer\_bench.bench.config

<a id="flashinfer_bench.bench.config.annotations"></a>

## annotations

<a id="flashinfer_bench.bench.config.dataclass"></a>

## dataclass

<a id="flashinfer_bench.bench.config.field"></a>

## field

<a id="flashinfer_bench.bench.config.Literal"></a>

## Literal

<a id="flashinfer_bench.bench.config.BenchmarkConfig"></a>

## BenchmarkConfig Objects

```python
@dataclass
class BenchmarkConfig()
```

Configuration for benchmark runs.

All fields have default values to make configuration optional.

<a id="flashinfer_bench.bench.config.BenchmarkConfig.warmup_runs"></a>

#### warmup\_runs

<a id="flashinfer_bench.bench.config.BenchmarkConfig.iterations"></a>

#### iterations

<a id="flashinfer_bench.bench.config.BenchmarkConfig.num_trials"></a>

#### num\_trials

<a id="flashinfer_bench.bench.config.BenchmarkConfig.rtol"></a>

#### rtol

<a id="flashinfer_bench.bench.config.BenchmarkConfig.atol"></a>

#### atol

<a id="flashinfer_bench.bench.config.BenchmarkConfig.log_level"></a>

#### log\_level

<a id="flashinfer_bench.bench.config.BenchmarkConfig.__post_init__"></a>

#### \_\_post\_init\_\_

```python
def __post_init__()
```

<a id="flashinfer_bench.bench.runner"></a>

# flashinfer\_bench.bench.runner

<a id="flashinfer_bench.bench.runner.annotations"></a>

## annotations

<a id="flashinfer_bench.bench.runner.ABC"></a>

## ABC

<a id="flashinfer_bench.bench.runner.abstractmethod"></a>

## abstractmethod

<a id="flashinfer_bench.bench.runner.dataclass"></a>

## dataclass

<a id="flashinfer_bench.bench.runner.Path"></a>

## Path

<a id="flashinfer_bench.bench.runner.Any"></a>

## Any

<a id="flashinfer_bench.bench.runner.Dict"></a>

## Dict

<a id="flashinfer_bench.bench.runner.List"></a>

## List

<a id="flashinfer_bench.bench.runner.Optional"></a>

## Optional

<a id="flashinfer_bench.bench.runner.torch"></a>

## torch

<a id="flashinfer_bench.bench.runner.Definition"></a>

## Definition

<a id="flashinfer_bench.bench.runner.Solution"></a>

## Solution

<a id="flashinfer_bench.bench.runner.Evaluation"></a>

## Evaluation

<a id="flashinfer_bench.bench.runner.Workload"></a>

## Workload

<a id="flashinfer_bench.bench.runner.BenchmarkConfig"></a>

## BenchmarkConfig

<a id="flashinfer_bench.bench.runner.RunnerError"></a>

## RunnerError Objects

```python
class RunnerError(RuntimeError)
```

<a id="flashinfer_bench.bench.runner.RunnerFatalError"></a>

## RunnerFatalError Objects

```python
class RunnerFatalError(RunnerError)
```

<a id="flashinfer_bench.bench.runner.BaselineHandle"></a>

## BaselineHandle Objects

```python
class BaselineHandle(str)
```

<a id="flashinfer_bench.bench.runner.DeviceBaseline"></a>

## DeviceBaseline Objects

```python
@dataclass
class DeviceBaseline()
```

<a id="flashinfer_bench.bench.runner.DeviceBaseline.handle"></a>

#### handle

<a id="flashinfer_bench.bench.runner.DeviceBaseline.defn"></a>

#### defn

<a id="flashinfer_bench.bench.runner.DeviceBaseline.device"></a>

#### device

<a id="flashinfer_bench.bench.runner.DeviceBaseline.inputs_dev"></a>

#### inputs\_dev

<a id="flashinfer_bench.bench.runner.DeviceBaseline.ref_outputs_dev"></a>

#### ref\_outputs\_dev

<a id="flashinfer_bench.bench.runner.DeviceBaseline.ref_mean_latency_ms"></a>

#### ref\_mean\_latency\_ms

<a id="flashinfer_bench.bench.runner.Runner"></a>

## Runner Objects

```python
class Runner(ABC)
```

Single-device runner interface.

<a id="flashinfer_bench.bench.runner.Runner.__init__"></a>

#### \_\_init\_\_

```python
def __init__(device: str, log_dir: str = "/tmp/flashinfer_bench") -> None
```

<a id="flashinfer_bench.bench.runner.Runner.run_ref"></a>

#### run\_ref

```python
@abstractmethod
def run_ref(defn: Definition,
            workload: Workload,
            cfg: BenchmarkConfig,
            traceset_root: Optional[Path] = None) -> BaselineHandle
```

Build a baseline for the given definition and workload.

<a id="flashinfer_bench.bench.runner.Runner.run_solution"></a>

#### run\_solution

```python
@abstractmethod
def run_solution(sol: Solution, baseline: BaselineHandle,
                 cfg: BenchmarkConfig) -> Evaluation
```

Run a solution against the given baseline.

<a id="flashinfer_bench.bench.runner.Runner.close"></a>

#### close

```python
@abstractmethod
def close() -> None
```

Release all resources.

<a id="flashinfer_bench.bench.runner.Runner.release"></a>

#### release

```python
@abstractmethod
def release(baseline: BaselineHandle) -> None
```

Release a baseline.

<a id="flashinfer_bench.bench.runners.mp_runner"></a>

# flashinfer\_bench.bench.runners.mp\_runner

<a id="flashinfer_bench.bench.runners.mp_runner.annotations"></a>

## annotations

<a id="flashinfer_bench.bench.runners.mp_runner.os"></a>

## os

<a id="flashinfer_bench.bench.runners.mp_runner.time"></a>

## time

<a id="flashinfer_bench.bench.runners.mp_runner.uuid"></a>

## uuid

<a id="flashinfer_bench.bench.runners.mp_runner.datetime"></a>

## datetime

<a id="flashinfer_bench.bench.runners.mp_runner.Path"></a>

## Path

<a id="flashinfer_bench.bench.runners.mp_runner.Any"></a>

## Any

<a id="flashinfer_bench.bench.runners.mp_runner.Dict"></a>

## Dict

<a id="flashinfer_bench.bench.runners.mp_runner.List"></a>

## List

<a id="flashinfer_bench.bench.runners.mp_runner.Optional"></a>

## Optional

<a id="flashinfer_bench.bench.runners.mp_runner.torch"></a>

## torch

<a id="flashinfer_bench.bench.runners.mp_runner.mp"></a>

## mp

<a id="flashinfer_bench.bench.runners.mp_runner.get_registry"></a>

## get\_registry

<a id="flashinfer_bench.bench.runners.mp_runner.Runnable"></a>

## Runnable

<a id="flashinfer_bench.bench.runners.mp_runner.Definition"></a>

## Definition

<a id="flashinfer_bench.bench.runners.mp_runner.Solution"></a>

## Solution

<a id="flashinfer_bench.bench.runners.mp_runner.Correctness"></a>

## Correctness

<a id="flashinfer_bench.bench.runners.mp_runner.Evaluation"></a>

## Evaluation

<a id="flashinfer_bench.bench.runners.mp_runner.EvaluationStatus"></a>

## EvaluationStatus

<a id="flashinfer_bench.bench.runners.mp_runner.Performance"></a>

## Performance

<a id="flashinfer_bench.bench.runners.mp_runner.Workload"></a>

## Workload

<a id="flashinfer_bench.bench.runners.mp_runner.env_snapshot"></a>

## env\_snapshot

<a id="flashinfer_bench.bench.runners.mp_runner.redirect_stdio_to_file"></a>

## redirect\_stdio\_to\_file

<a id="flashinfer_bench.bench.runners.mp_runner.torch_dtype_from_def"></a>

## torch\_dtype\_from\_def

<a id="flashinfer_bench.bench.runners.mp_runner.BenchmarkConfig"></a>

## BenchmarkConfig

<a id="flashinfer_bench.bench.runners.mp_runner.BaselineHandle"></a>

## BaselineHandle

<a id="flashinfer_bench.bench.runners.mp_runner.DeviceBaseline"></a>

## DeviceBaseline

<a id="flashinfer_bench.bench.runners.mp_runner.Runner"></a>

## Runner

<a id="flashinfer_bench.bench.runners.mp_runner.RunnerError"></a>

## RunnerError

<a id="flashinfer_bench.bench.runners.mp_runner.RunnerFatalError"></a>

## RunnerFatalError

<a id="flashinfer_bench.bench.runners.mp_runner.time_runnable"></a>

## time\_runnable

<a id="flashinfer_bench.bench.runners.mp_runner._rand_tensor"></a>

#### \_rand\_tensor

```python
def _rand_tensor(shape: List[int], dtype: torch.dtype,
                 device: torch.device) -> torch.Tensor
```

<a id="flashinfer_bench.bench.runners.mp_runner._normalize_outputs"></a>

#### \_normalize\_outputs

```python
def _normalize_outputs(
        out: Any, *, device: torch.device, output_names: List[str],
        output_dtypes: Dict[str, torch.dtype]) -> Dict[str, torch.Tensor]
```

<a id="flashinfer_bench.bench.runners.mp_runner._load_safetensors"></a>

#### \_load\_safetensors

```python
def _load_safetensors(
        defn: Definition,
        wl: Workload,
        traceset_root: Optional[Path] = None) -> Dict[str, torch.Tensor]
```

<a id="flashinfer_bench.bench.runners.mp_runner._gen_inputs"></a>

#### \_gen\_inputs

```python
def _gen_inputs(
        defn: Definition,
        wl: Workload,
        device: str,
        stensors: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]
```

<a id="flashinfer_bench.bench.runners.mp_runner.MultiProcessRunner"></a>

## MultiProcessRunner Objects

```python
class MultiProcessRunner(Runner)
```

Each instance binds to a CUDA device; the baseline resides in the main process; each Solution starts an independent Worker process for strong isolation.

<a id="flashinfer_bench.bench.runners.mp_runner.MultiProcessRunner.__init__"></a>

#### \_\_init\_\_

```python
def __init__(device: str) -> None
```

<a id="flashinfer_bench.bench.runners.mp_runner.MultiProcessRunner.run_ref"></a>

#### run\_ref

```python
def run_ref(defn: Definition,
            workload: Workload,
            cfg: BenchmarkConfig,
            traceset_root: Optional[Path] = None) -> BaselineHandle
```

<a id="flashinfer_bench.bench.runners.mp_runner.MultiProcessRunner.run_solution"></a>

#### run\_solution

```python
def run_solution(sol: Solution, baseline: BaselineHandle,
                 cfg: BenchmarkConfig) -> Evaluation
```

<a id="flashinfer_bench.bench.runners.mp_runner.MultiProcessRunner.release"></a>

#### release

```python
def release(baseline: BaselineHandle) -> None
```

<a id="flashinfer_bench.bench.runners.mp_runner.MultiProcessRunner.close"></a>

#### close

```python
def close() -> None
```

<a id="flashinfer_bench.bench.runners.mp_runner._solution_worker_main"></a>

#### \_solution\_worker\_main

```python
def _solution_worker_main(conn: mp.connection.Connection, device: str,
                          defn: Definition, sol: Solution,
                          cfg: BenchmarkConfig, log_path: str) -> None
```

Worker process: strong isolation for single Solution. Borrow/return trial data via Pipe and send Evaluation back to parent process.

<a id="flashinfer_bench.bench.runners.mp_runner._make_eval"></a>

#### \_make\_eval

```python
def _make_eval(status: EvaluationStatus,
               device: str,
               log_file: str,
               correctness: Optional[Correctness] = None,
               performance: Optional[Performance] = None,
               error: Optional[str] = None) -> Evaluation
```

<a id="flashinfer_bench.bench.timing"></a>

# flashinfer\_bench.bench.timing

<a id="flashinfer_bench.bench.timing.annotations"></a>

## annotations

<a id="flashinfer_bench.bench.timing.threading"></a>

## threading

<a id="flashinfer_bench.bench.timing.torch"></a>

## torch

<a id="flashinfer_bench.bench.timing.do_bench"></a>

## do\_bench

<a id="flashinfer_bench.bench.timing.Runnable"></a>

## Runnable

<a id="flashinfer_bench.bench.timing._device_locks"></a>

#### \_device\_locks

<a id="flashinfer_bench.bench.timing._registry_lock"></a>

#### \_registry\_lock

<a id="flashinfer_bench.bench.timing._device_lock"></a>

#### \_device\_lock

```python
def _device_lock(device: str) -> threading.Lock
```

<a id="flashinfer_bench.bench.timing.time_runnable"></a>

#### time\_runnable

```python
def time_runnable(fn: Runnable, inputs: dict, warmup: int, iters: int,
                  device: str) -> float
```

<a id="flashinfer_bench.cli"></a>

# flashinfer\_bench.cli

<a id="flashinfer_bench.cli.cli"></a>

## cli

<a id="flashinfer_bench.cli.__all__"></a>

#### \_\_all\_\_

<a id="flashinfer_bench.cli.main"></a>

# flashinfer\_bench.cli.main

<a id="flashinfer_bench.cli.main.argparse"></a>

## argparse

<a id="flashinfer_bench.cli.main.Path"></a>

## Path

<a id="flashinfer_bench.cli.main.List"></a>

## List

<a id="flashinfer_bench.cli.main.Benchmark"></a>

## Benchmark

<a id="flashinfer_bench.cli.main.BenchmarkConfig"></a>

## BenchmarkConfig

<a id="flashinfer_bench.cli.main.TraceSet"></a>

## TraceSet

<a id="flashinfer_bench.cli.main.save_json_file"></a>

## save\_json\_file

<a id="flashinfer_bench.cli.main.save_jsonl_file"></a>

## save\_jsonl\_file

<a id="flashinfer_bench.cli.main.best"></a>

#### best

```python
def best(args: argparse.Namespace)
```

<a id="flashinfer_bench.cli.main.summary"></a>

#### summary

```python
def summary(args: argparse.Namespace)
```

<a id="flashinfer_bench.cli.main.merge_tracesets"></a>

#### merge\_tracesets

```python
def merge_tracesets(trace_sets)
```

Merge multiple TraceSets into one, raising on definition conflicts.

<a id="flashinfer_bench.cli.main.export_traceset"></a>

#### export\_traceset

```python
def export_traceset(trace_set, output_dir)
```

Export a TraceSet to a directory in the expected structure.

<a id="flashinfer_bench.cli.main.merge"></a>

#### merge

```python
def merge(args: argparse.Namespace)
```

Merge multiple TraceSets into a single one and export to output directory.

<a id="flashinfer_bench.cli.main.visualize"></a>

#### visualize

```python
def visualize(args: argparse.Namespace)
```

Visualize benchmark results as a console table.

<a id="flashinfer_bench.cli.main.run"></a>

#### run

```python
def run(args: argparse.Namespace)
```

Benchmark run: executes benchmarks and writes results.

<a id="flashinfer_bench.cli.main._load_traces"></a>

#### \_load\_traces

```python
def _load_traces(args: argparse.Namespace) -> List[TraceSet]
```

<a id="flashinfer_bench.cli.main.cli"></a>

#### cli

```python
def cli()
```

<a id="flashinfer_bench.compile"></a>

# flashinfer\_bench.compile

Compiler subsystem package.

Exports common builder types for convenience.

<a id="flashinfer_bench.compile.Builder"></a>

## Builder

<a id="flashinfer_bench.compile.BuildError"></a>

## BuildError

<a id="flashinfer_bench.compile.Runnable"></a>

## Runnable

<a id="flashinfer_bench.compile.__all__"></a>

#### \_\_all\_\_

<a id="flashinfer_bench.compile.builder"></a>

# flashinfer\_bench.compile.builder

<a id="flashinfer_bench.compile.builder.annotations"></a>

## annotations

<a id="flashinfer_bench.compile.builder.hashlib"></a>

## hashlib

<a id="flashinfer_bench.compile.builder.os"></a>

## os

<a id="flashinfer_bench.compile.builder.re"></a>

## re

<a id="flashinfer_bench.compile.builder.tempfile"></a>

## tempfile

<a id="flashinfer_bench.compile.builder.ABC"></a>

## ABC

<a id="flashinfer_bench.compile.builder.abstractmethod"></a>

## abstractmethod

<a id="flashinfer_bench.compile.builder.Callable"></a>

## Callable

<a id="flashinfer_bench.compile.builder.Dict"></a>

## Dict

<a id="flashinfer_bench.compile.builder.Optional"></a>

## Optional

<a id="flashinfer_bench.compile.builder.Runnable"></a>

## Runnable

<a id="flashinfer_bench.compile.builder.Definition"></a>

## Definition

<a id="flashinfer_bench.compile.builder.Solution"></a>

## Solution

<a id="flashinfer_bench.compile.builder.SourceFile"></a>

## SourceFile

<a id="flashinfer_bench.compile.builder.write_sources_to_dir"></a>

#### write\_sources\_to\_dir

```python
def write_sources_to_dir(dir: str, sources: list[SourceFile]) -> None
```

<a id="flashinfer_bench.compile.builder.write_sources_to_temp"></a>

#### write\_sources\_to\_temp

```python
def write_sources_to_temp(base: str,
                          sources: list[SourceFile],
                          pkg: Optional[str] = None) -> str
```

<a id="flashinfer_bench.compile.builder.create_pkg_name"></a>

#### create\_pkg\_name

```python
def create_pkg_name(sol: Solution, prefix: str = "") -> str
```

<a id="flashinfer_bench.compile.builder.BuildError"></a>

## BuildError Objects

```python
class BuildError(RuntimeError)
```

Raised when a builder fails to construct a runnable implementation.

<a id="flashinfer_bench.compile.builder.Builder"></a>

## Builder Objects

```python
class Builder(ABC)
```

Builder abstraction: (Definition, Solution) -> Runnable with hidden cache.

<a id="flashinfer_bench.compile.builder.Builder.__init__"></a>

#### \_\_init\_\_

```python
def __init__() -> None
```

<a id="flashinfer_bench.compile.builder.Builder.can_build"></a>

#### can\_build

```python
@abstractmethod
def can_build(solution: Solution) -> bool
```

Build guard to check if this builder can handle the given solution.

<a id="flashinfer_bench.compile.builder.Builder._build"></a>

#### \_build

```python
@abstractmethod
def _build(definition: Definition, solution: Solution) -> Runnable
```

Perform a real build and return a Runnable; raise BuildError on failure.

<a id="flashinfer_bench.compile.builder.Builder._make_closer"></a>

#### \_make\_closer

```python
@abstractmethod
def _make_closer(*args, **kwargs) -> Callable[[], None]
```

Factory for a resource closer used by the concrete builder.

<a id="flashinfer_bench.compile.builder.Builder._make_key"></a>

#### \_make\_key

```python
@abstractmethod
def _make_key(solution: Solution) -> str
```

Cache key for a solution.

<a id="flashinfer_bench.compile.builder.Builder.build"></a>

#### build

```python
def build(definition: Definition, solution: Solution) -> Runnable
```

Public entry with per-solution cache keyed by solution.name.

<a id="flashinfer_bench.compile.builder.Builder.clear_cache"></a>

#### clear\_cache

```python
def clear_cache() -> None
```

Close all cached runnables and clear the cache.

<a id="flashinfer_bench.compile.builders"></a>

# flashinfer\_bench.compile.builders

<a id="flashinfer_bench.compile.builders.annotations"></a>

## annotations

<a id="flashinfer_bench.compile.builders.CUDABuilder"></a>

## CUDABuilder

<a id="flashinfer_bench.compile.builders.PythonBuilder"></a>

## PythonBuilder

<a id="flashinfer_bench.compile.builders.TritonBuilder"></a>

## TritonBuilder

<a id="flashinfer_bench.compile.builders.__all__"></a>

#### \_\_all\_\_

<a id="flashinfer_bench.compile.builders.cuda_builder"></a>

# flashinfer\_bench.compile.builders.cuda\_builder

<a id="flashinfer_bench.compile.builders.cuda_builder.annotations"></a>

## annotations

<a id="flashinfer_bench.compile.builders.cuda_builder.os"></a>

## os

<a id="flashinfer_bench.compile.builders.cuda_builder.re"></a>

## re

<a id="flashinfer_bench.compile.builders.cuda_builder.shutil"></a>

## shutil

<a id="flashinfer_bench.compile.builders.cuda_builder.sys"></a>

## sys

<a id="flashinfer_bench.compile.builders.cuda_builder.resources"></a>

## resources

<a id="flashinfer_bench.compile.builders.cuda_builder.Path"></a>

## Path

<a id="flashinfer_bench.compile.builders.cuda_builder.Dict"></a>

## Dict

<a id="flashinfer_bench.compile.builders.cuda_builder.List"></a>

## List

<a id="flashinfer_bench.compile.builders.cuda_builder.Definition"></a>

## Definition

<a id="flashinfer_bench.compile.builders.cuda_builder.Solution"></a>

## Solution

<a id="flashinfer_bench.compile.builders.cuda_builder.SourceFile"></a>

## SourceFile

<a id="flashinfer_bench.compile.builders.cuda_builder.SupportedLanguages"></a>

## SupportedLanguages

<a id="flashinfer_bench.compile.builders.cuda_builder.Builder"></a>

## Builder

<a id="flashinfer_bench.compile.builders.cuda_builder.BuildError"></a>

## BuildError

<a id="flashinfer_bench.compile.builders.cuda_builder.create_pkg_name"></a>

## create\_pkg\_name

<a id="flashinfer_bench.compile.builders.cuda_builder.write_sources_to_dir"></a>

## write\_sources\_to\_dir

<a id="flashinfer_bench.compile.builders.cuda_builder.Runnable"></a>

## Runnable

<a id="flashinfer_bench.compile.builders.cuda_builder.CUDA_ALLOWED_EXTS"></a>

#### CUDA\_ALLOWED\_EXTS

<a id="flashinfer_bench.compile.builders.cuda_builder._verify_cuda"></a>

#### \_verify\_cuda

```python
def _verify_cuda() -> bool
```

<a id="flashinfer_bench.compile.builders.cuda_builder._get_package_paths"></a>

#### \_get\_package\_paths

```python
def _get_package_paths(pkg_name: str, lib_names: List[str] = None)
```

<a id="flashinfer_bench.compile.builders.cuda_builder.CUDA_DEPS"></a>

#### CUDA\_DEPS

<a id="flashinfer_bench.compile.builders.cuda_builder._discover_cuda_deps"></a>

#### \_discover\_cuda\_deps

```python
def _discover_cuda_deps(extra_include_paths: Dict[str, str],
                        extra_ldflags: Dict[str, List[str]])
```

<a id="flashinfer_bench.compile.builders.cuda_builder.CUDA_DEPS_INCLUDE_PATTERNS"></a>

#### CUDA\_DEPS\_INCLUDE\_PATTERNS

<a id="flashinfer_bench.compile.builders.cuda_builder._check_dependency"></a>

#### \_check\_dependency

```python
def _check_dependency(sources: List[SourceFile], dep_name: str) -> bool
```

<a id="flashinfer_bench.compile.builders.cuda_builder.CUDABuilder"></a>

## CUDABuilder Objects

```python
class CUDABuilder(Builder)
```

<a id="flashinfer_bench.compile.builders.cuda_builder.CUDABuilder._cuda_available"></a>

#### \_cuda\_available

<a id="flashinfer_bench.compile.builders.cuda_builder.CUDABuilder._get_cuda_available"></a>

#### \_get\_cuda\_available

```python
@classmethod
def _get_cuda_available(cls) -> bool
```

<a id="flashinfer_bench.compile.builders.cuda_builder.CUDABuilder.__init__"></a>

#### \_\_init\_\_

```python
def __init__() -> None
```

<a id="flashinfer_bench.compile.builders.cuda_builder.CUDABuilder.can_build"></a>

#### can\_build

```python
def can_build(sol: Solution) -> bool
```

<a id="flashinfer_bench.compile.builders.cuda_builder.CUDABuilder._make_key"></a>

#### \_make\_key

```python
def _make_key(solution: Solution) -> str
```

<a id="flashinfer_bench.compile.builders.cuda_builder.CUDABuilder._make_closer"></a>

#### \_make\_closer

```python
def _make_closer()
```

<a id="flashinfer_bench.compile.builders.cuda_builder.CUDABuilder._build"></a>

#### \_build

```python
def _build(defn: Definition, sol: Solution) -> Runnable
```

<a id="flashinfer_bench.compile.builders.cuda_builder.CUDABuilder.clear_cache"></a>

#### clear\_cache

```python
def clear_cache() -> None
```

<a id="flashinfer_bench.compile.builders.python_builder"></a>

# flashinfer\_bench.compile.builders.python\_builder

<a id="flashinfer_bench.compile.builders.python_builder.annotations"></a>

## annotations

<a id="flashinfer_bench.compile.builders.python_builder.importlib"></a>

## importlib

<a id="flashinfer_bench.compile.builders.python_builder.os"></a>

## os

<a id="flashinfer_bench.compile.builders.python_builder.shutil"></a>

## shutil

<a id="flashinfer_bench.compile.builders.python_builder.sys"></a>

## sys

<a id="flashinfer_bench.compile.builders.python_builder.Path"></a>

## Path

<a id="flashinfer_bench.compile.builders.python_builder.Any"></a>

## Any

<a id="flashinfer_bench.compile.builders.python_builder.Callable"></a>

## Callable

<a id="flashinfer_bench.compile.builders.python_builder.Definition"></a>

## Definition

<a id="flashinfer_bench.compile.builders.python_builder.Solution"></a>

## Solution

<a id="flashinfer_bench.compile.builders.python_builder.SupportedLanguages"></a>

## SupportedLanguages

<a id="flashinfer_bench.compile.builders.python_builder.Builder"></a>

## Builder

<a id="flashinfer_bench.compile.builders.python_builder.BuildError"></a>

## BuildError

<a id="flashinfer_bench.compile.builders.python_builder.create_pkg_name"></a>

## create\_pkg\_name

<a id="flashinfer_bench.compile.builders.python_builder.write_sources_to_temp"></a>

## write\_sources\_to\_temp

<a id="flashinfer_bench.compile.builders.python_builder.Runnable"></a>

## Runnable

<a id="flashinfer_bench.compile.builders.python_builder.PythonBuilder"></a>

## PythonBuilder Objects

```python
class PythonBuilder(Builder)
```

Load a Python entry point from provided sources into a temporary module.

<a id="flashinfer_bench.compile.builders.python_builder.PythonBuilder.can_build"></a>

#### can\_build

```python
def can_build(sol: Solution) -> bool
```

<a id="flashinfer_bench.compile.builders.python_builder.PythonBuilder._make_key"></a>

#### \_make\_key

```python
def _make_key(solution: Solution) -> str
```

<a id="flashinfer_bench.compile.builders.python_builder.PythonBuilder._make_closer"></a>

#### \_make\_closer

```python
def _make_closer(pkg: str, tmpdir: str) -> Callable[[], None]
```

<a id="flashinfer_bench.compile.builders.python_builder.PythonBuilder._build"></a>

#### \_build

```python
def _build(defn: Definition, sol: Solution) -> Runnable
```

<a id="flashinfer_bench.compile.builders.triton_builder"></a>

# flashinfer\_bench.compile.builders.triton\_builder

<a id="flashinfer_bench.compile.builders.triton_builder.annotations"></a>

## annotations

<a id="flashinfer_bench.compile.builders.triton_builder.Definition"></a>

## Definition

<a id="flashinfer_bench.compile.builders.triton_builder.Solution"></a>

## Solution

<a id="flashinfer_bench.compile.builders.triton_builder.SupportedLanguages"></a>

## SupportedLanguages

<a id="flashinfer_bench.compile.builders.triton_builder.Builder"></a>

## Builder

<a id="flashinfer_bench.compile.builders.triton_builder.BuildError"></a>

## BuildError

<a id="flashinfer_bench.compile.builders.triton_builder.create_pkg_name"></a>

## create\_pkg\_name

<a id="flashinfer_bench.compile.builders.triton_builder.Runnable"></a>

## Runnable

<a id="flashinfer_bench.compile.builders.triton_builder.PythonBuilder"></a>

## PythonBuilder

<a id="flashinfer_bench.compile.builders.triton_builder._verify_triton"></a>

#### \_verify\_triton

```python
def _verify_triton() -> bool
```

<a id="flashinfer_bench.compile.builders.triton_builder.TritonBuilder"></a>

## TritonBuilder Objects

```python
class TritonBuilder(Builder)
```

<a id="flashinfer_bench.compile.builders.triton_builder.TritonBuilder._triton_available"></a>

#### \_triton\_available

<a id="flashinfer_bench.compile.builders.triton_builder.TritonBuilder._get_triton_available"></a>

#### \_get\_triton\_available

```python
@classmethod
def _get_triton_available(cls) -> bool
```

<a id="flashinfer_bench.compile.builders.triton_builder.TritonBuilder.__init__"></a>

#### \_\_init\_\_

```python
def __init__(py_builder: PythonBuilder) -> None
```

<a id="flashinfer_bench.compile.builders.triton_builder.TritonBuilder.can_build"></a>

#### can\_build

```python
def can_build(sol: Solution) -> bool
```

<a id="flashinfer_bench.compile.builders.triton_builder.TritonBuilder._make_key"></a>

#### \_make\_key

```python
def _make_key(solution: Solution) -> str
```

<a id="flashinfer_bench.compile.builders.triton_builder.TritonBuilder._make_closer"></a>

#### \_make\_closer

```python
def _make_closer(*args, **kwargs)
```

<a id="flashinfer_bench.compile.builders.triton_builder.TritonBuilder._build"></a>

#### \_build

```python
def _build(defn: Definition, sol: Solution) -> Runnable
```

<a id="flashinfer_bench.compile.registry"></a>

# flashinfer\_bench.compile.registry

<a id="flashinfer_bench.compile.registry.annotations"></a>

## annotations

<a id="flashinfer_bench.compile.registry.Tuple"></a>

## Tuple

<a id="flashinfer_bench.compile.registry.Definition"></a>

## Definition

<a id="flashinfer_bench.compile.registry.BuildSpec"></a>

## BuildSpec

<a id="flashinfer_bench.compile.registry.Solution"></a>

## Solution

<a id="flashinfer_bench.compile.registry.SourceFile"></a>

## SourceFile

<a id="flashinfer_bench.compile.registry.SupportedLanguages"></a>

## SupportedLanguages

<a id="flashinfer_bench.compile.registry.Builder"></a>

## Builder

<a id="flashinfer_bench.compile.registry.BuildError"></a>

## BuildError

<a id="flashinfer_bench.compile.registry.Runnable"></a>

## Runnable

<a id="flashinfer_bench.compile.registry.BuilderRegistry"></a>

## BuilderRegistry Objects

```python
class BuilderRegistry()
```

Registry that dispatches to the first capable builder.

<a id="flashinfer_bench.compile.registry.BuilderRegistry.__init__"></a>

#### \_\_init\_\_

```python
def __init__(builders: Tuple[Builder, ...]) -> None
```

<a id="flashinfer_bench.compile.registry.BuilderRegistry.clear"></a>

#### clear

```python
def clear() -> None
```

<a id="flashinfer_bench.compile.registry.BuilderRegistry.build"></a>

#### build

```python
def build(defn: Definition, sol: Solution) -> Runnable
```

<a id="flashinfer_bench.compile.registry.BuilderRegistry.build_reference"></a>

#### build\_reference

```python
def build_reference(defn: Definition) -> Runnable
```

<a id="flashinfer_bench.compile.registry._registry"></a>

#### \_registry

<a id="flashinfer_bench.compile.registry.get_registry"></a>

#### get\_registry

```python
def get_registry() -> BuilderRegistry
```

<a id="flashinfer_bench.compile.runnable"></a>

# flashinfer\_bench.compile.runnable

<a id="flashinfer_bench.compile.runnable.annotations"></a>

## annotations

<a id="flashinfer_bench.compile.runnable.Any"></a>

## Any

<a id="flashinfer_bench.compile.runnable.Callable"></a>

## Callable

<a id="flashinfer_bench.compile.runnable.Dict"></a>

## Dict

<a id="flashinfer_bench.compile.runnable.Optional"></a>

## Optional

<a id="flashinfer_bench.compile.runnable.Runnable"></a>

## Runnable Objects

```python
class Runnable()
```

<a id="flashinfer_bench.compile.runnable.Runnable.__init__"></a>

#### \_\_init\_\_

```python
def __init__(fn: Callable[..., Any],
             closer: Callable[[], None],
             meta: Optional[Dict[str, Any]] = None) -> None
```

A runnable callable with a required resource closer.

closer: must be provided by the builder and be idempotent.

<a id="flashinfer_bench.compile.runnable.Runnable.__call__"></a>

#### \_\_call\_\_

```python
def __call__(**kwargs: Any) -> Any
```

- Accept kwargs only (aligns with Definition.inputs naming)
- Unpack a single-element tuple to a scalar value
- No type/shape/count validation; errors surface naturally

<a id="flashinfer_bench.compile.runnable.Runnable.close"></a>

#### close

```python
def close() -> None
```

Release build artifacts/resources; must be idempotent.

<a id="flashinfer_bench.data"></a>

# flashinfer\_bench.data

Data layer with strongly-typed dataclasses for FlashInfer Bench.

<a id="flashinfer_bench.data.AxisConst"></a>

## AxisConst

<a id="flashinfer_bench.data.AxisVar"></a>

## AxisVar

<a id="flashinfer_bench.data.Definition"></a>

## Definition

<a id="flashinfer_bench.data.TensorSpec"></a>

## TensorSpec

<a id="flashinfer_bench.data.from_json"></a>

## from\_json

<a id="flashinfer_bench.data.load_json_file"></a>

## load\_json\_file

<a id="flashinfer_bench.data.load_jsonl_file"></a>

## load\_jsonl\_file

<a id="flashinfer_bench.data.save_json_file"></a>

## save\_json\_file

<a id="flashinfer_bench.data.save_jsonl_file"></a>

## save\_jsonl\_file

<a id="flashinfer_bench.data.to_json"></a>

## to\_json

<a id="flashinfer_bench.data.BuildSpec"></a>

## BuildSpec

<a id="flashinfer_bench.data.Solution"></a>

## Solution

<a id="flashinfer_bench.data.SourceFile"></a>

## SourceFile

<a id="flashinfer_bench.data.SupportedLanguages"></a>

## SupportedLanguages

<a id="flashinfer_bench.data.Correctness"></a>

## Correctness

<a id="flashinfer_bench.data.Environment"></a>

## Environment

<a id="flashinfer_bench.data.Evaluation"></a>

## Evaluation

<a id="flashinfer_bench.data.EvaluationStatus"></a>

## EvaluationStatus

<a id="flashinfer_bench.data.InputDesc"></a>

## InputDesc

<a id="flashinfer_bench.data.Performance"></a>

## Performance

<a id="flashinfer_bench.data.RandomInput"></a>

## RandomInput

<a id="flashinfer_bench.data.SafetensorsInput"></a>

## SafetensorsInput

<a id="flashinfer_bench.data.ScalarInput"></a>

## ScalarInput

<a id="flashinfer_bench.data.Trace"></a>

## Trace

<a id="flashinfer_bench.data.Workload"></a>

## Workload

<a id="flashinfer_bench.data.TraceSet"></a>

## TraceSet

<a id="flashinfer_bench.data.__all__"></a>

#### \_\_all\_\_

<a id="flashinfer_bench.data.definition"></a>

# flashinfer\_bench.data.definition

Strong-typed data definitions for workload specifications.

<a id="flashinfer_bench.data.definition.ast"></a>

## ast

<a id="flashinfer_bench.data.definition.dataclass"></a>

## dataclass

<a id="flashinfer_bench.data.definition.cached_property"></a>

## cached\_property

<a id="flashinfer_bench.data.definition.Dict"></a>

## Dict

<a id="flashinfer_bench.data.definition.List"></a>

## List

<a id="flashinfer_bench.data.definition.Literal"></a>

## Literal

<a id="flashinfer_bench.data.definition.Optional"></a>

## Optional

<a id="flashinfer_bench.data.definition.Tuple"></a>

## Tuple

<a id="flashinfer_bench.data.definition.Union"></a>

## Union

<a id="flashinfer_bench.data.definition.AxisConst"></a>

## AxisConst Objects

```python
@dataclass
class AxisConst()
```

Constant axis with a fixed value.

<a id="flashinfer_bench.data.definition.AxisConst.type"></a>

#### type

<a id="flashinfer_bench.data.definition.AxisConst.value"></a>

#### value

<a id="flashinfer_bench.data.definition.AxisConst.description"></a>

#### description

<a id="flashinfer_bench.data.definition.AxisConst.__post_init__"></a>

#### \_\_post\_init\_\_

```python
def __post_init__()
```

<a id="flashinfer_bench.data.definition.AxisVar"></a>

## AxisVar Objects

```python
@dataclass
class AxisVar()
```

Variable axis that can be specified at runtime.

<a id="flashinfer_bench.data.definition.AxisVar.type"></a>

#### type

<a id="flashinfer_bench.data.definition.AxisVar.parent"></a>

#### parent

<a id="flashinfer_bench.data.definition.AxisVar.description"></a>

#### description

<a id="flashinfer_bench.data.definition.ALLOWED_DTYPES"></a>

#### ALLOWED\_DTYPES

<a id="flashinfer_bench.data.definition.TensorSpec"></a>

## TensorSpec Objects

```python
@dataclass
class TensorSpec()
```

Specification for a tensor including shape and data type.

<a id="flashinfer_bench.data.definition.TensorSpec.shape"></a>

#### shape

<a id="flashinfer_bench.data.definition.TensorSpec.dtype"></a>

#### dtype

<a id="flashinfer_bench.data.definition.TensorSpec.description"></a>

#### description

<a id="flashinfer_bench.data.definition.TensorSpec.__post_init__"></a>

#### \_\_post\_init\_\_

```python
def __post_init__()
```

<a id="flashinfer_bench.data.definition.Definition"></a>

## Definition Objects

```python
@dataclass
class Definition()
```

Complete definition of a computational workload.

<a id="flashinfer_bench.data.definition.Definition.name"></a>

#### name

<a id="flashinfer_bench.data.definition.Definition.type"></a>

#### type

<a id="flashinfer_bench.data.definition.Definition.axes"></a>

#### axes

<a id="flashinfer_bench.data.definition.Definition.inputs"></a>

#### inputs

<a id="flashinfer_bench.data.definition.Definition.outputs"></a>

#### outputs

<a id="flashinfer_bench.data.definition.Definition.reference"></a>

#### reference

<a id="flashinfer_bench.data.definition.Definition.tags"></a>

#### tags

<a id="flashinfer_bench.data.definition.Definition.description"></a>

#### description

<a id="flashinfer_bench.data.definition.Definition.constraints"></a>

#### constraints

<a id="flashinfer_bench.data.definition.Definition.__post_init__"></a>

#### \_\_post\_init\_\_

```python
def __post_init__()
```

<a id="flashinfer_bench.data.definition.Definition.get_const_axes"></a>

#### get\_const\_axes

```python
def get_const_axes() -> Dict[str, int]
```

Get all constant axes and their values.

<a id="flashinfer_bench.data.definition.Definition.get_var_axes"></a>

#### get\_var\_axes

```python
def get_var_axes() -> List[str]
```

Get all variable axis names.

<a id="flashinfer_bench.data.definition.Definition.get_var_axes_bindings"></a>

#### get\_var\_axes\_bindings

```python
@cached_property
def get_var_axes_bindings() -> Dict[str, Tuple[str, int]]
```

Get the bindings of variable axes to input tensors dimensions.

**Returns**:

  Dict[str, Tuple[str, int]]: axis_name -> (input_name, dim_idx)

<a id="flashinfer_bench.data.definition.Definition._get_shapes"></a>

#### \_get\_shapes

```python
def _get_shapes(
        tensors: Dict[str, TensorSpec],
        var_values: Optional[Dict[str, int]] = None) -> Dict[str, List[int]]
```

Get concrete tensor shapes given variable axis values.

<a id="flashinfer_bench.data.definition.Definition.get_input_shapes"></a>

#### get\_input\_shapes

```python
def get_input_shapes(
        var_values: Optional[Dict[str, int]] = None) -> Dict[str, List[int]]
```

Get concrete input shapes given variable axis values.

<a id="flashinfer_bench.data.definition.Definition.get_output_shapes"></a>

#### get\_output\_shapes

```python
def get_output_shapes(
        var_values: Optional[Dict[str, int]] = None) -> Dict[str, List[int]]
```

Get concrete output shapes given variable axis values.

<a id="flashinfer_bench.data.json_codec"></a>

# flashinfer\_bench.data.json\_codec

Unified JSON encoding/decoding for all dataclasses.

<a id="flashinfer_bench.data.json_codec.json"></a>

## json

<a id="flashinfer_bench.data.json_codec.types"></a>

## types

<a id="flashinfer_bench.data.json_codec.fields"></a>

## fields

<a id="flashinfer_bench.data.json_codec.is_dataclass"></a>

## is\_dataclass

<a id="flashinfer_bench.data.json_codec.Path"></a>

## Path

<a id="flashinfer_bench.data.json_codec.Any"></a>

## Any

<a id="flashinfer_bench.data.json_codec.Dict"></a>

## Dict

<a id="flashinfer_bench.data.json_codec.List"></a>

## List

<a id="flashinfer_bench.data.json_codec.Type"></a>

## Type

<a id="flashinfer_bench.data.json_codec.TypeVar"></a>

## TypeVar

<a id="flashinfer_bench.data.json_codec.Union"></a>

## Union

<a id="flashinfer_bench.data.json_codec.get_args"></a>

## get\_args

<a id="flashinfer_bench.data.json_codec.get_origin"></a>

## get\_origin

<a id="flashinfer_bench.data.json_codec.AxisConst"></a>

## AxisConst

<a id="flashinfer_bench.data.json_codec.AxisVar"></a>

## AxisVar

<a id="flashinfer_bench.data.json_codec.Definition"></a>

## Definition

<a id="flashinfer_bench.data.json_codec.TensorSpec"></a>

## TensorSpec

<a id="flashinfer_bench.data.json_codec.BuildSpec"></a>

## BuildSpec

<a id="flashinfer_bench.data.json_codec.Solution"></a>

## Solution

<a id="flashinfer_bench.data.json_codec.SourceFile"></a>

## SourceFile

<a id="flashinfer_bench.data.json_codec.SupportedLanguages"></a>

## SupportedLanguages

<a id="flashinfer_bench.data.json_codec.Correctness"></a>

## Correctness

<a id="flashinfer_bench.data.json_codec.Environment"></a>

## Environment

<a id="flashinfer_bench.data.json_codec.Evaluation"></a>

## Evaluation

<a id="flashinfer_bench.data.json_codec.EvaluationStatus"></a>

## EvaluationStatus

<a id="flashinfer_bench.data.json_codec.Performance"></a>

## Performance

<a id="flashinfer_bench.data.json_codec.RandomInput"></a>

## RandomInput

<a id="flashinfer_bench.data.json_codec.SafetensorsInput"></a>

## SafetensorsInput

<a id="flashinfer_bench.data.json_codec.ScalarInput"></a>

## ScalarInput

<a id="flashinfer_bench.data.json_codec.Trace"></a>

## Trace

<a id="flashinfer_bench.data.json_codec.Workload"></a>

## Workload

<a id="flashinfer_bench.data.json_codec.T"></a>

#### T

<a id="flashinfer_bench.data.json_codec._PRESERVE_NULL_FIELDS"></a>

#### \_PRESERVE\_NULL\_FIELDS

<a id="flashinfer_bench.data.json_codec._FIELD_ORDER"></a>

#### \_FIELD\_ORDER

<a id="flashinfer_bench.data.json_codec._decode_axes"></a>

#### \_decode\_axes

```python
def _decode_axes(v)
```

Decode Definition.axes.

<a id="flashinfer_bench.data.json_codec._decode_tensor_specs"></a>

#### \_decode\_tensor\_specs

```python
def _decode_tensor_specs(v)
```

Decode Definition.inputs/outputs.

<a id="flashinfer_bench.data.json_codec._decode_workload_inputs"></a>

#### \_decode\_workload\_inputs

```python
def _decode_workload_inputs(v)
```

Decode Workload.inputs.

<a id="flashinfer_bench.data.json_codec._decode_sources"></a>

#### \_decode\_sources

```python
def _decode_sources(v)
```

Decode Solution.sources.

<a id="flashinfer_bench.data.json_codec._decode_spec"></a>

#### \_decode\_spec

```python
def _decode_spec(v)
```

Decode Solution.spec.

<a id="flashinfer_bench.data.json_codec._decode_evaluation"></a>

#### \_decode\_evaluation

```python
def _decode_evaluation(v)
```

Decode Trace.evaluation.

<a id="flashinfer_bench.data.json_codec._decode_workload"></a>

#### \_decode\_workload

```python
def _decode_workload(v)
```

Decode Trace.workload.

<a id="flashinfer_bench.data.json_codec._decode_correctness"></a>

#### \_decode\_correctness

```python
def _decode_correctness(v)
```

Decode Evaluation.correctness.

<a id="flashinfer_bench.data.json_codec._decode_performance"></a>

#### \_decode\_performance

```python
def _decode_performance(v)
```

Decode Evaluation.performance.

<a id="flashinfer_bench.data.json_codec._decode_environment"></a>

#### \_decode\_environment

```python
def _decode_environment(v)
```

Decode Evaluation.environment.

<a id="flashinfer_bench.data.json_codec._decode_language"></a>

#### \_decode\_language

```python
def _decode_language(v)
```

Decode BuildSpec.language.

<a id="flashinfer_bench.data.json_codec._decode_status"></a>

#### \_decode\_status

```python
def _decode_status(v)
```

Decode Evaluation.status.

<a id="flashinfer_bench.data.json_codec._FIELD_DECODERS"></a>

#### \_FIELD\_DECODERS

<a id="flashinfer_bench.data.json_codec.dataclass_to_dict"></a>

#### dataclass\_to\_dict

```python
def dataclass_to_dict(obj: Any) -> Any
```

Convert a dataclass instance to a dictionary, handling nested dataclasses.

<a id="flashinfer_bench.data.json_codec.dict_to_dataclass"></a>

#### dict\_to\_dataclass

```python
def dict_to_dataclass(data: Dict[str, Any], cls: Type[T]) -> T
```

Convert a dictionary to a dataclass instance.

<a id="flashinfer_bench.data.json_codec.to_json"></a>

#### to\_json

```python
def to_json(obj: Any, indent: int = 2) -> str
```

Convert a dataclass to JSON string.

<a id="flashinfer_bench.data.json_codec.from_json"></a>

#### from\_json

```python
def from_json(json_str: str, cls: Type[T]) -> T
```

Parse JSON string to dataclass.

<a id="flashinfer_bench.data.json_codec.save_json_file"></a>

#### save\_json\_file

```python
def save_json_file(obj: Any, path: Union[str, Path]) -> None
```

Save a dataclass to a JSON file.

<a id="flashinfer_bench.data.json_codec.load_json_file"></a>

#### load\_json\_file

```python
def load_json_file(path: Union[str, Path], cls: Type[T]) -> T
```

Load a dataclass from a JSON file.

<a id="flashinfer_bench.data.json_codec.save_jsonl_file"></a>

#### save\_jsonl\_file

```python
def save_jsonl_file(objects: List[Any], path: Union[str, Path]) -> None
```

Save a list of dataclasses to a JSONL file.

<a id="flashinfer_bench.data.json_codec.load_jsonl_file"></a>

#### load\_jsonl\_file

```python
def load_jsonl_file(path: Union[str, Path], cls: Type[T]) -> List[T]
```

Load a list of dataclasses from a JSONL file.

<a id="flashinfer_bench.data.json_codec.append_jsonl_line"></a>

#### append\_jsonl\_line

```python
def append_jsonl_line(path: Union[str, Path], obj: Any) -> None
```

Append a dataclass to a JSONL file.

<a id="flashinfer_bench.data.json_codec.append_jsonl_lines"></a>

#### append\_jsonl\_lines

```python
def append_jsonl_lines(path: Union[str, Path], objs: List[Any]) -> None
```

Append a list of dataclasses to a JSONL file.

<a id="flashinfer_bench.data.solution"></a>

# flashinfer\_bench.data.solution

Strong-typed data definitions for solution implementations.

<a id="flashinfer_bench.data.solution.ast"></a>

## ast

<a id="flashinfer_bench.data.solution.dataclass"></a>

## dataclass

<a id="flashinfer_bench.data.solution.Enum"></a>

## Enum

<a id="flashinfer_bench.data.solution.List"></a>

## List

<a id="flashinfer_bench.data.solution.Optional"></a>

## Optional

<a id="flashinfer_bench.data.solution.SupportedLanguages"></a>

## SupportedLanguages Objects

```python
class SupportedLanguages(Enum)
```

<a id="flashinfer_bench.data.solution.SupportedLanguages.PYTHON"></a>

#### PYTHON

<a id="flashinfer_bench.data.solution.SupportedLanguages.TRITON"></a>

#### TRITON

<a id="flashinfer_bench.data.solution.SupportedLanguages.CUDA"></a>

#### CUDA

<a id="flashinfer_bench.data.solution.SourceFile"></a>

## SourceFile Objects

```python
@dataclass
class SourceFile()
```

A single source code file.

<a id="flashinfer_bench.data.solution.SourceFile.path"></a>

#### path

<a id="flashinfer_bench.data.solution.SourceFile.content"></a>

#### content

<a id="flashinfer_bench.data.solution.SourceFile.__post_init__"></a>

#### \_\_post\_init\_\_

```python
def __post_init__()
```

<a id="flashinfer_bench.data.solution.BuildSpec"></a>

## BuildSpec Objects

```python
@dataclass
class BuildSpec()
```

Build specification for a solution.

<a id="flashinfer_bench.data.solution.BuildSpec.language"></a>

#### language

<a id="flashinfer_bench.data.solution.BuildSpec.target_hardware"></a>

#### target\_hardware

<a id="flashinfer_bench.data.solution.BuildSpec.entry_point"></a>

#### entry\_point

<a id="flashinfer_bench.data.solution.BuildSpec.dependencies"></a>

#### dependencies

<a id="flashinfer_bench.data.solution.BuildSpec.build_commands"></a>

#### build\_commands

<a id="flashinfer_bench.data.solution.BuildSpec.__post_init__"></a>

#### \_\_post\_init\_\_

```python
def __post_init__()
```

<a id="flashinfer_bench.data.solution.Solution"></a>

## Solution Objects

```python
@dataclass
class Solution()
```

A concrete implementation for a given Definition.

<a id="flashinfer_bench.data.solution.Solution.name"></a>

#### name

<a id="flashinfer_bench.data.solution.Solution.definition"></a>

#### definition

Name of the Definition this solves

<a id="flashinfer_bench.data.solution.Solution.author"></a>

#### author

<a id="flashinfer_bench.data.solution.Solution.spec"></a>

#### spec

<a id="flashinfer_bench.data.solution.Solution.sources"></a>

#### sources

<a id="flashinfer_bench.data.solution.Solution.description"></a>

#### description

<a id="flashinfer_bench.data.solution.Solution.__post_init__"></a>

#### \_\_post\_init\_\_

```python
def __post_init__()
```

<a id="flashinfer_bench.data.solution.Solution.get_entry_source"></a>

#### get\_entry\_source

```python
def get_entry_source() -> Optional[SourceFile]
```

Get the entry source file.

<a id="flashinfer_bench.data.solution.Solution.requires_build"></a>

#### requires\_build

```python
def requires_build() -> bool
```

Check if the solution requires a build step.

<a id="flashinfer_bench.data.solution.Solution.to_json"></a>

#### to\_json

```python
def to_json() -> str
```

Serialize the Solution to JSON string.

<a id="flashinfer_bench.data.trace"></a>

# flashinfer\_bench.data.trace

Strong-typed data definitions for traces and evaluations.

<a id="flashinfer_bench.data.trace.dataclass"></a>

## dataclass

<a id="flashinfer_bench.data.trace.field"></a>

## field

<a id="flashinfer_bench.data.trace.Enum"></a>

## Enum

<a id="flashinfer_bench.data.trace.Dict"></a>

## Dict

<a id="flashinfer_bench.data.trace.Literal"></a>

## Literal

<a id="flashinfer_bench.data.trace.Optional"></a>

## Optional

<a id="flashinfer_bench.data.trace.Union"></a>

## Union

<a id="flashinfer_bench.data.trace.RandomInput"></a>

## RandomInput Objects

```python
@dataclass
class RandomInput()
```

Random input generation descriptor.

<a id="flashinfer_bench.data.trace.RandomInput.type"></a>

#### type

<a id="flashinfer_bench.data.trace.ScalarValue"></a>

#### ScalarValue

<a id="flashinfer_bench.data.trace.ScalarInput"></a>

## ScalarInput Objects

```python
@dataclass
class ScalarInput()
```

Scalar literal.

<a id="flashinfer_bench.data.trace.ScalarInput.type"></a>

#### type

<a id="flashinfer_bench.data.trace.ScalarInput.value"></a>

#### value

<a id="flashinfer_bench.data.trace.ScalarInput.__post_init__"></a>

#### \_\_post\_init\_\_

```python
def __post_init__()
```

<a id="flashinfer_bench.data.trace.SafetensorsInput"></a>

## SafetensorsInput Objects

```python
@dataclass
class SafetensorsInput()
```

Input loaded from a safetensors file.

<a id="flashinfer_bench.data.trace.SafetensorsInput.type"></a>

#### type

<a id="flashinfer_bench.data.trace.SafetensorsInput.path"></a>

#### path

<a id="flashinfer_bench.data.trace.SafetensorsInput.tensor_key"></a>

#### tensor\_key

<a id="flashinfer_bench.data.trace.SafetensorsInput.__post_init__"></a>

#### \_\_post\_init\_\_

```python
def __post_init__()
```

<a id="flashinfer_bench.data.trace.InputDesc"></a>

#### InputDesc

<a id="flashinfer_bench.data.trace.Workload"></a>

## Workload Objects

```python
@dataclass
class Workload()
```

Concrete workload configuration.

<a id="flashinfer_bench.data.trace.Workload.axes"></a>

#### axes

<a id="flashinfer_bench.data.trace.Workload.inputs"></a>

#### inputs

<a id="flashinfer_bench.data.trace.Workload.uuid"></a>

#### uuid

<a id="flashinfer_bench.data.trace.Workload.__post_init__"></a>

#### \_\_post\_init\_\_

```python
def __post_init__()
```

<a id="flashinfer_bench.data.trace.Correctness"></a>

## Correctness Objects

```python
@dataclass
class Correctness()
```

Correctness metrics from evaluation.

<a id="flashinfer_bench.data.trace.Correctness.max_relative_error"></a>

#### max\_relative\_error

<a id="flashinfer_bench.data.trace.Correctness.max_absolute_error"></a>

#### max\_absolute\_error

<a id="flashinfer_bench.data.trace.Correctness.__post_init__"></a>

#### \_\_post\_init\_\_

```python
def __post_init__()
```

<a id="flashinfer_bench.data.trace.Performance"></a>

## Performance Objects

```python
@dataclass
class Performance()
```

Performance metrics from evaluation.

<a id="flashinfer_bench.data.trace.Performance.latency_ms"></a>

#### latency\_ms

<a id="flashinfer_bench.data.trace.Performance.reference_latency_ms"></a>

#### reference\_latency\_ms

<a id="flashinfer_bench.data.trace.Performance.speedup_factor"></a>

#### speedup\_factor

<a id="flashinfer_bench.data.trace.Performance.__post_init__"></a>

#### \_\_post\_init\_\_

```python
def __post_init__()
```

<a id="flashinfer_bench.data.trace.Environment"></a>

## Environment Objects

```python
@dataclass
class Environment()
```

Environment information from evaluation.

<a id="flashinfer_bench.data.trace.Environment.hardware"></a>

#### hardware

<a id="flashinfer_bench.data.trace.Environment.libs"></a>

#### libs

<a id="flashinfer_bench.data.trace.Environment.__post_init__"></a>

#### \_\_post\_init\_\_

```python
def __post_init__()
```

<a id="flashinfer_bench.data.trace.EvaluationStatus"></a>

## EvaluationStatus Objects

```python
class EvaluationStatus(Enum)
```

<a id="flashinfer_bench.data.trace.EvaluationStatus.PASSED"></a>

#### PASSED

<a id="flashinfer_bench.data.trace.EvaluationStatus.INCORRECT_SHAPE"></a>

#### INCORRECT\_SHAPE

<a id="flashinfer_bench.data.trace.EvaluationStatus.INCORRECT_NUMERICAL"></a>

#### INCORRECT\_NUMERICAL

<a id="flashinfer_bench.data.trace.EvaluationStatus.INCORRECT_DTYPE"></a>

#### INCORRECT\_DTYPE

<a id="flashinfer_bench.data.trace.EvaluationStatus.RUNTIME_ERROR"></a>

#### RUNTIME\_ERROR

<a id="flashinfer_bench.data.trace.EvaluationStatus.COMPILE_ERROR"></a>

#### COMPILE\_ERROR

<a id="flashinfer_bench.data.trace.Evaluation"></a>

## Evaluation Objects

```python
@dataclass
class Evaluation()
```

Complete evaluation result for a workload.

<a id="flashinfer_bench.data.trace.Evaluation.status"></a>

#### status

<a id="flashinfer_bench.data.trace.Evaluation.log_file"></a>

#### log\_file

<a id="flashinfer_bench.data.trace.Evaluation.environment"></a>

#### environment

<a id="flashinfer_bench.data.trace.Evaluation.timestamp"></a>

#### timestamp

<a id="flashinfer_bench.data.trace.Evaluation.correctness"></a>

#### correctness

<a id="flashinfer_bench.data.trace.Evaluation.performance"></a>

#### performance

<a id="flashinfer_bench.data.trace.Evaluation.error"></a>

#### error

<a id="flashinfer_bench.data.trace.Evaluation.__post_init__"></a>

#### \_\_post\_init\_\_

```python
def __post_init__()
```

<a id="flashinfer_bench.data.trace.Trace"></a>

## Trace Objects

```python
@dataclass
class Trace()
```

A Trace links a specific Solution to a specific Definition, details the exact
workload configuration used for the run, and records the complete evaluation result.

Special case: A "workload trace" only contains definition and workload fields,
with solution and evaluation set to None. This represents a workload configuration
without an actual benchmark run.

<a id="flashinfer_bench.data.trace.Trace.definition"></a>

#### definition

Name of the Definition

<a id="flashinfer_bench.data.trace.Trace.workload"></a>

#### workload

<a id="flashinfer_bench.data.trace.Trace.solution"></a>

#### solution

Name of the Solution

<a id="flashinfer_bench.data.trace.Trace.evaluation"></a>

#### evaluation

<a id="flashinfer_bench.data.trace.Trace.__post_init__"></a>

#### \_\_post\_init\_\_

```python
def __post_init__()
```

<a id="flashinfer_bench.data.trace.Trace.is_workload"></a>

#### is\_workload

```python
def is_workload() -> bool
```

Check if this is a workload trace.

<a id="flashinfer_bench.data.trace.Trace.is_successful"></a>

#### is\_successful

```python
def is_successful() -> bool
```

Check if the benchmark run was successful.

<a id="flashinfer_bench.data.traceset"></a>

# flashinfer\_bench.data.traceset

TraceSet as a pure data warehouse for definitions, solutions, and traces.

<a id="flashinfer_bench.data.traceset.defaultdict"></a>

## defaultdict

<a id="flashinfer_bench.data.traceset.dataclass"></a>

## dataclass

<a id="flashinfer_bench.data.traceset.field"></a>

## field

<a id="flashinfer_bench.data.traceset.Path"></a>

## Path

<a id="flashinfer_bench.data.traceset.Dict"></a>

## Dict

<a id="flashinfer_bench.data.traceset.List"></a>

## List

<a id="flashinfer_bench.data.traceset.Optional"></a>

## Optional

<a id="flashinfer_bench.data.traceset.Definition"></a>

## Definition

<a id="flashinfer_bench.data.traceset.load_json_file"></a>

## load\_json\_file

<a id="flashinfer_bench.data.traceset.load_jsonl_file"></a>

## load\_jsonl\_file

<a id="flashinfer_bench.data.traceset.Solution"></a>

## Solution

<a id="flashinfer_bench.data.traceset.EvaluationStatus"></a>

## EvaluationStatus

<a id="flashinfer_bench.data.traceset.Trace"></a>

## Trace

<a id="flashinfer_bench.data.traceset.TraceSet"></a>

## TraceSet Objects

```python
@dataclass
class TraceSet()
```

A pure data warehouse for definitions, solutions, workloads, and traces.

This class only handles data storage, loading, saving, querying, and statistics.

<a id="flashinfer_bench.data.traceset.TraceSet.root"></a>

#### root

<a id="flashinfer_bench.data.traceset.TraceSet.definitions"></a>

#### definitions

<a id="flashinfer_bench.data.traceset.TraceSet.solutions"></a>

#### solutions

<a id="flashinfer_bench.data.traceset.TraceSet.workload"></a>

#### workload

<a id="flashinfer_bench.data.traceset.TraceSet.traces"></a>

#### traces

<a id="flashinfer_bench.data.traceset.TraceSet.from_path"></a>

#### from\_path

```python
@classmethod
def from_path(cls, path: str) -> "TraceSet"
```

Load a TraceSet from a directory structure.

<a id="flashinfer_bench.data.traceset.TraceSet.get_solution"></a>

#### get\_solution

```python
def get_solution(name: str) -> Optional[Solution]
```

Get a solution by name.

<a id="flashinfer_bench.data.traceset.TraceSet.filter_traces"></a>

#### filter\_traces

```python
def filter_traces(def_name: str,
                  atol: float = 1e-2,
                  rtol: float = 1e-2) -> List[Trace]
```

Filter traces for a definition based on error bounds.

<a id="flashinfer_bench.data.traceset.TraceSet.get_best_trace"></a>

#### get\_best\_trace

```python
def get_best_trace(def_name: str,
                   axes: Optional[Dict[str, int]] = None,
                   max_abs_error: float = 1e-2,
                   max_rel_error: float = 1e-2) -> Optional[Trace]
```

Get the best trace for a definition based on performance.

This returns the Trace object itself.

<a id="flashinfer_bench.data.traceset.TraceSet.summary"></a>

#### summary

```python
def summary() -> Dict[str, any]
```

Get a summary of all traces.

<a id="flashinfer_bench.integration.flashinfer"></a>

# flashinfer\_bench.integration.flashinfer

<a id="flashinfer_bench.integration.flashinfer.annotations"></a>

## annotations

<a id="flashinfer_bench.integration.flashinfer.get_manager"></a>

## get\_manager

<a id="flashinfer_bench.integration.flashinfer.GQAPagedDecodeAdapter"></a>

## GQAPagedDecodeAdapter

<a id="flashinfer_bench.integration.flashinfer.GQAPagedPrefillAdapter"></a>

## GQAPagedPrefillAdapter

<a id="flashinfer_bench.integration.flashinfer.MLAPagedAdapter"></a>

## MLAPagedAdapter

<a id="flashinfer_bench.integration.flashinfer.RaggedPrefillAdapter"></a>

## RaggedPrefillAdapter

<a id="flashinfer_bench.integration.flashinfer.install_flashinfer_integrations"></a>

#### install\_flashinfer\_integrations

```python
def install_flashinfer_integrations() -> None
```

Install patches for a set of adapters. If a target does not exist in
the current environment, skip silently. Idempotent.

<a id="flashinfer_bench.integration.flashinfer.adapters"></a>

# flashinfer\_bench.integration.flashinfer.adapters

Adapters for flashinfer integration.

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_decode"></a>

# flashinfer\_bench.integration.flashinfer.adapters.gqa\_paged\_decode

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_decode.annotations"></a>

## annotations

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_decode.Any"></a>

## Any

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_decode.Callable"></a>

## Callable

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_decode.Dict"></a>

## Dict

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_decode.List"></a>

## List

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_decode.torch"></a>

## torch

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_decode.apply"></a>

## apply

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_decode.get_runtime"></a>

## get\_runtime

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_decode.PatchSpec"></a>

## PatchSpec

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_decode.ArgBinder"></a>

## ArgBinder

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_decode.ContextStore"></a>

## ContextStore

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_decode.infer_kv_layout_from_args"></a>

## infer\_kv\_layout\_from\_args

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_decode.infer_paged_kv_layout_from_tensors"></a>

## infer\_paged\_kv\_layout\_from\_tensors

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_decode.normalize_paged_kv_to_nhd"></a>

## normalize\_paged\_kv\_to\_nhd

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_decode.pick_sm_scale_gqa"></a>

## pick\_sm\_scale\_gqa

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_decode.write_back_outputs"></a>

## write\_back\_outputs

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_decode._def_name_resolver"></a>

#### \_def\_name\_resolver

```python
def _def_name_resolver(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale)
```

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_decode.GQAPagedDecodeAdapter"></a>

## GQAPagedDecodeAdapter Objects

```python
class GQAPagedDecodeAdapter()
```

Adapter for flashinfer BatchDecodeWithPagedKVCacheWrapper(plan+run).
Covers page_size=1 only.

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_decode.GQAPagedDecodeAdapter.__init__"></a>

#### \_\_init\_\_

```python
def __init__() -> None
```

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_decode.GQAPagedDecodeAdapter.targets"></a>

#### targets

```python
def targets() -> List[PatchSpec]
```

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_decode.GQAPagedDecodeAdapter.make_wrapper"></a>

#### make\_wrapper

```python
def make_wrapper(spec: PatchSpec, orig: Callable[...,
                                                 Any]) -> Callable[..., Any]
```

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_prefill"></a>

# flashinfer\_bench.integration.flashinfer.adapters.gqa\_paged\_prefill

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_prefill.annotations"></a>

## annotations

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_prefill.Any"></a>

## Any

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_prefill.Callable"></a>

## Callable

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_prefill.Dict"></a>

## Dict

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_prefill.List"></a>

## List

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_prefill.torch"></a>

## torch

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_prefill.apply"></a>

## apply

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_prefill.get_runtime"></a>

## get\_runtime

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_prefill.PatchSpec"></a>

## PatchSpec

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_prefill.ArgBinder"></a>

## ArgBinder

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_prefill.ContextStore"></a>

## ContextStore

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_prefill.infer_kv_layout_from_args"></a>

## infer\_kv\_layout\_from\_args

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_prefill.infer_paged_kv_layout_from_tensors"></a>

## infer\_paged\_kv\_layout\_from\_tensors

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_prefill.normalize_paged_kv_to_nhd"></a>

## normalize\_paged\_kv\_to\_nhd

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_prefill.pick_sm_scale_gqa"></a>

## pick\_sm\_scale\_gqa

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_prefill.write_back_outputs"></a>

## write\_back\_outputs

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_prefill._def_name_resolver"></a>

#### \_def\_name\_resolver

```python
def _def_name_resolver(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices,
                       sm_scale)
```

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_prefill.GQAPagedPrefillAdapter"></a>

## GQAPagedPrefillAdapter Objects

```python
class GQAPagedPrefillAdapter()
```

Adapter for flashinfer BatchPrefillWithPagedKVCacheWrapper(plan+run).
Covers causal=True and page_size=1 only.

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_prefill.GQAPagedPrefillAdapter.__init__"></a>

#### \_\_init\_\_

```python
def __init__() -> None
```

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_prefill.GQAPagedPrefillAdapter.targets"></a>

#### targets

```python
def targets() -> List[PatchSpec]
```

<a id="flashinfer_bench.integration.flashinfer.adapters.gqa_paged_prefill.GQAPagedPrefillAdapter.make_wrapper"></a>

#### make\_wrapper

```python
def make_wrapper(spec: PatchSpec, orig: Callable[...,
                                                 Any]) -> Callable[..., Any]
```

<a id="flashinfer_bench.integration.flashinfer.adapters.mla_paged"></a>

# flashinfer\_bench.integration.flashinfer.adapters.mla\_paged

<a id="flashinfer_bench.integration.flashinfer.adapters.mla_paged.annotations"></a>

## annotations

<a id="flashinfer_bench.integration.flashinfer.adapters.mla_paged.Any"></a>

## Any

<a id="flashinfer_bench.integration.flashinfer.adapters.mla_paged.Callable"></a>

## Callable

<a id="flashinfer_bench.integration.flashinfer.adapters.mla_paged.Dict"></a>

## Dict

<a id="flashinfer_bench.integration.flashinfer.adapters.mla_paged.List"></a>

## List

<a id="flashinfer_bench.integration.flashinfer.adapters.mla_paged.torch"></a>

## torch

<a id="flashinfer_bench.integration.flashinfer.adapters.mla_paged.apply"></a>

## apply

<a id="flashinfer_bench.integration.flashinfer.adapters.mla_paged.get_runtime"></a>

## get\_runtime

<a id="flashinfer_bench.integration.flashinfer.adapters.mla_paged.PatchSpec"></a>

## PatchSpec

<a id="flashinfer_bench.integration.flashinfer.adapters.mla_paged.ArgBinder"></a>

## ArgBinder

<a id="flashinfer_bench.integration.flashinfer.adapters.mla_paged.ContextStore"></a>

## ContextStore

<a id="flashinfer_bench.integration.flashinfer.adapters.mla_paged.pick_sm_scale_mla"></a>

## pick\_sm\_scale\_mla

<a id="flashinfer_bench.integration.flashinfer.adapters.mla_paged.write_back_outputs"></a>

## write\_back\_outputs

<a id="flashinfer_bench.integration.flashinfer.adapters.mla_paged._decode_def_name"></a>

#### \_decode\_def\_name

```python
def _decode_def_name(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices,
                     sm_scale)
```

<a id="flashinfer_bench.integration.flashinfer.adapters.mla_paged._prefill_def_name"></a>

#### \_prefill\_def\_name

```python
def _prefill_def_name(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr,
                      kv_indices, sm_scale)
```

<a id="flashinfer_bench.integration.flashinfer.adapters.mla_paged.MLAPagedAdapter"></a>

## MLAPagedAdapter Objects

```python
class MLAPagedAdapter()
```

Adapter for flashinfer.mla.BatchMLAPagedAttentionWrapper(plan+run).
- Detects decode vs (incremental) prefill by comparing qo_indptr vs q_nope batch size.
- Covers page_size=1 only.

<a id="flashinfer_bench.integration.flashinfer.adapters.mla_paged.MLAPagedAdapter.__init__"></a>

#### \_\_init\_\_

```python
def __init__() -> None
```

<a id="flashinfer_bench.integration.flashinfer.adapters.mla_paged.MLAPagedAdapter.targets"></a>

#### targets

```python
def targets() -> List[PatchSpec]
```

<a id="flashinfer_bench.integration.flashinfer.adapters.mla_paged.MLAPagedAdapter.make_wrapper"></a>

#### make\_wrapper

```python
def make_wrapper(spec: PatchSpec, orig: Callable[...,
                                                 Any]) -> Callable[..., Any]
```

<a id="flashinfer_bench.integration.flashinfer.adapters.ragged_prefill"></a>

# flashinfer\_bench.integration.flashinfer.adapters.ragged\_prefill

<a id="flashinfer_bench.integration.flashinfer.adapters.ragged_prefill.annotations"></a>

## annotations

<a id="flashinfer_bench.integration.flashinfer.adapters.ragged_prefill.Any"></a>

## Any

<a id="flashinfer_bench.integration.flashinfer.adapters.ragged_prefill.Callable"></a>

## Callable

<a id="flashinfer_bench.integration.flashinfer.adapters.ragged_prefill.Dict"></a>

## Dict

<a id="flashinfer_bench.integration.flashinfer.adapters.ragged_prefill.List"></a>

## List

<a id="flashinfer_bench.integration.flashinfer.adapters.ragged_prefill.torch"></a>

## torch

<a id="flashinfer_bench.integration.flashinfer.adapters.ragged_prefill.apply"></a>

## apply

<a id="flashinfer_bench.integration.flashinfer.adapters.ragged_prefill.get_runtime"></a>

## get\_runtime

<a id="flashinfer_bench.integration.flashinfer.adapters.ragged_prefill.PatchSpec"></a>

## PatchSpec

<a id="flashinfer_bench.integration.flashinfer.adapters.ragged_prefill.ArgBinder"></a>

## ArgBinder

<a id="flashinfer_bench.integration.flashinfer.adapters.ragged_prefill.ContextStore"></a>

## ContextStore

<a id="flashinfer_bench.integration.flashinfer.adapters.ragged_prefill.infer_kv_layout_from_args"></a>

## infer\_kv\_layout\_from\_args

<a id="flashinfer_bench.integration.flashinfer.adapters.ragged_prefill.infer_ragged_kv_layout_from_tensors"></a>

## infer\_ragged\_kv\_layout\_from\_tensors

<a id="flashinfer_bench.integration.flashinfer.adapters.ragged_prefill.normalize_ragged_kv_to_nhd"></a>

## normalize\_ragged\_kv\_to\_nhd

<a id="flashinfer_bench.integration.flashinfer.adapters.ragged_prefill.pick_sm_scale_gqa"></a>

## pick\_sm\_scale\_gqa

<a id="flashinfer_bench.integration.flashinfer.adapters.ragged_prefill.write_back_outputs"></a>

## write\_back\_outputs

<a id="flashinfer_bench.integration.flashinfer.adapters.ragged_prefill._def_name_resolver"></a>

#### \_def\_name\_resolver

```python
def _def_name_resolver(q, k, v, qo_indptr, kv_indptr, sm_scale)
```

<a id="flashinfer_bench.integration.flashinfer.adapters.ragged_prefill.RaggedPrefillAdapter"></a>

## RaggedPrefillAdapter Objects

```python
class RaggedPrefillAdapter()
```

Adapter for flashinfer BatchPrefillWithRaggedKVCacheWrapper(plan+run).
Only covers causal=True. Used by both GQA and MLA ragged prefill.

<a id="flashinfer_bench.integration.flashinfer.adapters.ragged_prefill.RaggedPrefillAdapter.__init__"></a>

#### \_\_init\_\_

```python
def __init__() -> None
```

<a id="flashinfer_bench.integration.flashinfer.adapters.ragged_prefill.RaggedPrefillAdapter.targets"></a>

#### targets

```python
def targets() -> List[PatchSpec]
```

<a id="flashinfer_bench.integration.flashinfer.adapters.ragged_prefill.RaggedPrefillAdapter.make_wrapper"></a>

#### make\_wrapper

```python
def make_wrapper(spec: PatchSpec, orig: Callable[...,
                                                 Any]) -> Callable[..., Any]
```

<a id="flashinfer_bench.integration.flashinfer.common"></a>

# flashinfer\_bench.integration.flashinfer.common

<a id="flashinfer_bench.integration.flashinfer.common.math"></a>

## math

<a id="flashinfer_bench.integration.flashinfer.common.Any"></a>

## Any

<a id="flashinfer_bench.integration.flashinfer.common.Optional"></a>

## Optional

<a id="flashinfer_bench.integration.flashinfer.common.torch"></a>

## torch

<a id="flashinfer_bench.integration.flashinfer.common.infer_kv_layout_from_args"></a>

#### infer\_kv\_layout\_from\_args

```python
def infer_kv_layout_from_args(inst) -> Optional[str]
```

<a id="flashinfer_bench.integration.flashinfer.common.infer_paged_kv_layout_from_tensors"></a>

#### infer\_paged\_kv\_layout\_from\_tensors

```python
def infer_paged_kv_layout_from_tensors(paged_kv_cache,
                                       num_kv_heads: int) -> Optional[str]
```

<a id="flashinfer_bench.integration.flashinfer.common.infer_ragged_kv_layout_from_tensors"></a>

#### infer\_ragged\_kv\_layout\_from\_tensors

```python
def infer_ragged_kv_layout_from_tensors(ragged_k_or_v,
                                        num_kv_heads: int) -> Optional[str]
```

<a id="flashinfer_bench.integration.flashinfer.common.normalize_paged_kv_to_nhd"></a>

#### normalize\_paged\_kv\_to\_nhd

```python
def normalize_paged_kv_to_nhd(paged_kv_cache, kv_layout: str)
```

<a id="flashinfer_bench.integration.flashinfer.common.normalize_ragged_kv_to_nhd"></a>

#### normalize\_ragged\_kv\_to\_nhd

```python
def normalize_ragged_kv_to_nhd(ragged_k_or_v, kv_layout: str)
```

<a id="flashinfer_bench.integration.flashinfer.common.pick_sm_scale_gqa"></a>

#### pick\_sm\_scale\_gqa

```python
def pick_sm_scale_gqa(head_dim: int, maybe: Any) -> float
```

<a id="flashinfer_bench.integration.flashinfer.common.pick_sm_scale_mla"></a>

#### pick\_sm\_scale\_mla

```python
def pick_sm_scale_mla(head_dim_qk_nope: int, head_dim_qk_pe: int,
                      maybe: Any) -> float
```

<a id="flashinfer_bench.integration.flashinfer.common.write_back_outputs"></a>

#### write\_back\_outputs

```python
def write_back_outputs(*,
                       output: torch.Tensor,
                       lse: torch.Tensor,
                       want_lse: bool,
                       out_buf=None,
                       lse_buf=None)
```

<a id="flashinfer_bench.integration.patch_manager"></a>

# flashinfer\_bench.integration.patch\_manager

<a id="flashinfer_bench.integration.patch_manager.annotations"></a>

## annotations

<a id="flashinfer_bench.integration.patch_manager.importlib"></a>

## importlib

<a id="flashinfer_bench.integration.patch_manager.dataclass"></a>

## dataclass

<a id="flashinfer_bench.integration.patch_manager.Any"></a>

## Any

<a id="flashinfer_bench.integration.patch_manager.Callable"></a>

## Callable

<a id="flashinfer_bench.integration.patch_manager.Dict"></a>

## Dict

<a id="flashinfer_bench.integration.patch_manager.Literal"></a>

## Literal

<a id="flashinfer_bench.integration.patch_manager.Optional"></a>

## Optional

<a id="flashinfer_bench.integration.patch_manager.Tuple"></a>

## Tuple

<a id="flashinfer_bench.integration.patch_manager.Kind"></a>

#### Kind

<a id="flashinfer_bench.integration.patch_manager.PatchSpec"></a>

## PatchSpec Objects

```python
@dataclass
class PatchSpec()
```

<a id="flashinfer_bench.integration.patch_manager.PatchSpec.path"></a>

#### path

<a id="flashinfer_bench.integration.patch_manager.PatchSpec.kind"></a>

#### kind

<a id="flashinfer_bench.integration.patch_manager.PatchSpec.name"></a>

#### name

<a id="flashinfer_bench.integration.patch_manager.PatchSpec.ctx_key"></a>

#### ctx\_key

<a id="flashinfer_bench.integration.patch_manager.PatchManager"></a>

## PatchManager Objects

```python
class PatchManager()
```

Responsible for: resolve target, replace attr, restore original.

<a id="flashinfer_bench.integration.patch_manager.PatchManager.__init__"></a>

#### \_\_init\_\_

```python
def __init__() -> None
```

<a id="flashinfer_bench.integration.patch_manager.PatchManager._resolve"></a>

#### \_resolve

```python
def _resolve(path: str) -> Tuple[object, str, Any]
```

Resolve a dotted path to (owner, attr, original_attr).
Works for module functions or class attributes (methods).

<a id="flashinfer_bench.integration.patch_manager.PatchManager.patch"></a>

#### patch

```python
def patch(
    spec: PatchSpec, wrapper_factory: Callable[[PatchSpec, Callable[..., Any]],
                                               Callable[..., Any]]
) -> bool
```

Install a wrapper on target; return True if success, False if target missing.

<a id="flashinfer_bench.integration.patch_manager.PatchManager.unpatch_all"></a>

#### unpatch\_all

```python
def unpatch_all() -> None
```

Restore all originals.

<a id="flashinfer_bench.integration.patch_manager._manager"></a>

#### \_manager

<a id="flashinfer_bench.integration.patch_manager.get_manager"></a>

#### get\_manager

```python
def get_manager() -> PatchManager
```

<a id="flashinfer_bench.integration.utils"></a>

# flashinfer\_bench.integration.utils

<a id="flashinfer_bench.integration.utils.annotations"></a>

## annotations

<a id="flashinfer_bench.integration.utils.inspect"></a>

## inspect

<a id="flashinfer_bench.integration.utils.Any"></a>

## Any

<a id="flashinfer_bench.integration.utils.Dict"></a>

## Dict

<a id="flashinfer_bench.integration.utils.Mapping"></a>

## Mapping

<a id="flashinfer_bench.integration.utils.Tuple"></a>

## Tuple

<a id="flashinfer_bench.integration.utils.WeakKeyDictionary"></a>

## WeakKeyDictionary

<a id="flashinfer_bench.integration.utils.ArgBinder"></a>

## ArgBinder Objects

```python
class ArgBinder()
```

Cache inspect.signature and bind once per callable.

<a id="flashinfer_bench.integration.utils.ArgBinder.__init__"></a>

#### \_\_init\_\_

```python
def __init__(fn) -> None
```

<a id="flashinfer_bench.integration.utils.ArgBinder.from_callable"></a>

#### from\_callable

```python
@classmethod
def from_callable(cls, fn) -> "ArgBinder"
```

<a id="flashinfer_bench.integration.utils.ArgBinder.bind"></a>

#### bind

```python
def bind(args: Tuple[Any, ...], kwargs: Mapping[str, Any]) -> Dict[str, Any]
```

<a id="flashinfer_bench.integration.utils.ContextStore"></a>

## ContextStore Objects

```python
class ContextStore()
```

Per-instance loose store; adapter decides fields.

<a id="flashinfer_bench.integration.utils.ContextStore.__init__"></a>

#### \_\_init\_\_

```python
def __init__() -> None
```

<a id="flashinfer_bench.integration.utils.ContextStore.get"></a>

#### get

```python
def get(inst: object) -> Dict[str, Any]
```

<a id="flashinfer_bench.tracer"></a>

# flashinfer\_bench.tracer

<a id="flashinfer_bench.tracer.annotations"></a>

## annotations

<a id="flashinfer_bench.tracer.make_tracing_hook"></a>

## make\_tracing\_hook

<a id="flashinfer_bench.tracer.Tracer"></a>

## Tracer

<a id="flashinfer_bench.tracer.disable_tracing"></a>

## disable\_tracing

<a id="flashinfer_bench.tracer.enable_tracing"></a>

## enable\_tracing

<a id="flashinfer_bench.tracer.get_tracer"></a>

## get\_tracer

<a id="flashinfer_bench.tracer.TraceEntry"></a>

## TraceEntry

<a id="flashinfer_bench.tracer.TracingRule"></a>

## TracingRule

<a id="flashinfer_bench.tracer.__all__"></a>

#### \_\_all\_\_

<a id="flashinfer_bench.tracer.hook_impl"></a>

# flashinfer\_bench.tracer.hook\_impl

<a id="flashinfer_bench.tracer.hook_impl.annotations"></a>

## annotations

<a id="flashinfer_bench.tracer.hook_impl.Any"></a>

## Any

<a id="flashinfer_bench.tracer.hook_impl.Mapping"></a>

## Mapping

<a id="flashinfer_bench.tracer.hook_impl.get_runtime"></a>

## get\_runtime

<a id="flashinfer_bench.tracer.hook_impl.make_tracing_hook"></a>

#### make\_tracing\_hook

```python
def make_tracing_hook(tracer)
```

Create a lightweight apply hook that forwards to tracer.collect.

Hook signature: (def_name, runtime_kwargs) -> None

<a id="flashinfer_bench.tracer.presets"></a>

# flashinfer\_bench.tracer.presets

<a id="flashinfer_bench.tracer.presets.random"></a>

## random

<a id="flashinfer_bench.tracer.presets.Any"></a>

## Any

<a id="flashinfer_bench.tracer.presets.Dict"></a>

## Dict

<a id="flashinfer_bench.tracer.presets.List"></a>

## List

<a id="flashinfer_bench.tracer.presets.Optional"></a>

## Optional

<a id="flashinfer_bench.tracer.presets.torch"></a>

## torch

<a id="flashinfer_bench.tracer.presets.TraceEntry"></a>

## TraceEntry

<a id="flashinfer_bench.tracer.presets.TracingRule"></a>

## TracingRule

<a id="flashinfer_bench.tracer.presets.policy_keep_first_k"></a>

#### policy\_keep\_first\_k

```python
def policy_keep_first_k(k: int)
```

Keep first k entries as unique.

<a id="flashinfer_bench.tracer.presets.policy_keep_random_k"></a>

#### policy\_keep\_random\_k

```python
def policy_keep_random_k(k: int, seed: Optional[int] = None)
```

Keep random k entries as unique.

<a id="flashinfer_bench.tracer.presets.policy_dedup_by_axes"></a>

#### policy\_dedup\_by\_axes

```python
def policy_dedup_by_axes(k: int = 1)
```

Policy that deduplicates by same axes values.

k: The number of entries with the same axes values to keep.

<a id="flashinfer_bench.tracer.presets.policy_dedup_by_avg_seq_len"></a>

#### policy\_dedup\_by\_avg\_seq\_len

```python
def policy_dedup_by_avg_seq_len(k: int = 1)
```

Deduplicate by rounded average sequence length inferred from `kv_indptr`.

- For each entry, if `picked['kv_indptr']` or `picked['seq_indptr']` exists and is valid (a 1-D tensor with
length >= 2), we compute:
avg_seq_len = int(round(indptr[-1].item() / (len(indptr) - 1)))
Entries with the same `avg_seq_len` are considered duplicates, and at most
`k` entries are kept for each `avg_seq_len` value.

**Arguments**:

- `k` - max number of entries to keep per distinct average seq length (>0).



**Returns**:

  A policy(entries) -> subset(entries), stable by entry.order.

<a id="flashinfer_bench.tracer.presets.KEEP_ALL"></a>

#### KEEP\_ALL

<a id="flashinfer_bench.tracer.presets.KEEP_RANDOM_ONE"></a>

#### KEEP\_RANDOM\_ONE

<a id="flashinfer_bench.tracer.presets.DEDUP_BY_AXES"></a>

#### DEDUP\_BY\_AXES

<a id="flashinfer_bench.tracer.presets.DEDUP_BY_AVG_SEQ_LEN"></a>

#### DEDUP\_BY\_AVG\_SEQ\_LEN

<a id="flashinfer_bench.tracer.presets.KEY_AXES"></a>

#### KEY\_AXES

<a id="flashinfer_bench.tracer.presets.dump_int32"></a>

#### dump\_int32

```python
def dump_int32()
```

Select only int32 tensors for dumping. These inputs are usually indptrs.

<a id="flashinfer_bench.tracer.presets.DUMP_ALL"></a>

#### DUMP\_ALL

<a id="flashinfer_bench.tracer.presets.DUMP_NONE"></a>

#### DUMP\_NONE

<a id="flashinfer_bench.tracer.presets.DUMP_INT32"></a>

#### DUMP\_INT32

<a id="flashinfer_bench.tracer.presets.gemm_rule"></a>

#### gemm\_rule

<a id="flashinfer_bench.tracer.presets.mla_paged_prefill_rule"></a>

#### mla\_paged\_prefill\_rule

<a id="flashinfer_bench.tracer.presets.mla_ragged_prefill_rule"></a>

#### mla\_ragged\_prefill\_rule

<a id="flashinfer_bench.tracer.presets.mla_paged_decode_rule"></a>

#### mla\_paged\_decode\_rule

<a id="flashinfer_bench.tracer.presets.gqa_paged_prefill_rule"></a>

#### gqa\_paged\_prefill\_rule

<a id="flashinfer_bench.tracer.presets.gqa_ragged_prefill_rule"></a>

#### gqa\_ragged\_prefill\_rule

<a id="flashinfer_bench.tracer.presets.gqa_paged_decode_rule"></a>

#### gqa\_paged\_decode\_rule

<a id="flashinfer_bench.tracer.presets.all_dump_rule"></a>

#### all\_dump\_rule

<a id="flashinfer_bench.tracer.presets.axes_only_rule"></a>

#### axes\_only\_rule

<a id="flashinfer_bench.tracer.rule"></a>

# flashinfer\_bench.tracer.rule

<a id="flashinfer_bench.tracer.rule.gemm_rule"></a>

## gemm\_rule

<a id="flashinfer_bench.tracer.rule.gqa_paged_decode_rule"></a>

## gqa\_paged\_decode\_rule

<a id="flashinfer_bench.tracer.rule.gqa_paged_prefill_rule"></a>

## gqa\_paged\_prefill\_rule

<a id="flashinfer_bench.tracer.rule.gqa_ragged_prefill_rule"></a>

## gqa\_ragged\_prefill\_rule

<a id="flashinfer_bench.tracer.rule.mla_paged_decode_rule"></a>

## mla\_paged\_decode\_rule

<a id="flashinfer_bench.tracer.rule.mla_paged_prefill_rule"></a>

## mla\_paged\_prefill\_rule

<a id="flashinfer_bench.tracer.rule.mla_ragged_prefill_rule"></a>

## mla\_ragged\_prefill\_rule

<a id="flashinfer_bench.tracer.rule.fib_full_tracing"></a>

#### fib\_full\_tracing

<a id="flashinfer_bench.tracer.rule.fib_attn_tracing"></a>

#### fib\_attn\_tracing

<a id="flashinfer_bench.tracer.tracer"></a>

# flashinfer\_bench.tracer.tracer

<a id="flashinfer_bench.tracer.tracer.atexit"></a>

## atexit

<a id="flashinfer_bench.tracer.tracer.json"></a>

## json

<a id="flashinfer_bench.tracer.tracer.logging"></a>

## logging

<a id="flashinfer_bench.tracer.tracer.signal"></a>

## signal

<a id="flashinfer_bench.tracer.tracer.threading"></a>

## threading

<a id="flashinfer_bench.tracer.tracer.uuid"></a>

## uuid

<a id="flashinfer_bench.tracer.tracer.Path"></a>

## Path

<a id="flashinfer_bench.tracer.tracer.Any"></a>

## Any

<a id="flashinfer_bench.tracer.tracer.Dict"></a>

## Dict

<a id="flashinfer_bench.tracer.tracer.Hashable"></a>

## Hashable

<a id="flashinfer_bench.tracer.tracer.List"></a>

## List

<a id="flashinfer_bench.tracer.tracer.Optional"></a>

## Optional

<a id="flashinfer_bench.tracer.tracer.Set"></a>

## Set

<a id="flashinfer_bench.tracer.tracer.Tuple"></a>

## Tuple

<a id="flashinfer_bench.tracer.tracer.torch"></a>

## torch

<a id="flashinfer_bench.tracer.tracer.torch"></a>

## torch

<a id="flashinfer_bench.tracer.tracer.Definition"></a>

## Definition

<a id="flashinfer_bench.tracer.tracer.TraceEntry"></a>

## TraceEntry

<a id="flashinfer_bench.tracer.tracer.TracingRule"></a>

## TracingRule

<a id="flashinfer_bench.tracer.tracer._current_tracer"></a>

#### \_current\_tracer

<a id="flashinfer_bench.tracer.tracer._tracer_lock"></a>

#### \_tracer\_lock

<a id="flashinfer_bench.tracer.tracer.Tracer"></a>

## Tracer Objects

```python
class Tracer()
```

Process-wide singleton tracer for workload collection.

<a id="flashinfer_bench.tracer.tracer.Tracer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(rules: Dict[str, TracingRule],
             out_dir: Optional[Path] = None,
             blob_dir: Optional[Path] = None)
```

Initialize the tracer.

**Arguments**:

- `rules` - A set of tracing rules
- `out_dir` - Output directory for traces. Default is FIB_DATASET_PATH/traces/workloads
- `blob_dir` - Blob directory for safetensors. Default is FIB_DATASET_PATH/blob/workloads

<a id="flashinfer_bench.tracer.tracer.Tracer._validate"></a>

#### \_validate

```python
def _validate()
```

Validate tracer configuration at enable-time.

<a id="flashinfer_bench.tracer.tracer.Tracer.collect"></a>

#### collect

```python
def collect(def_name: str, runtime_args: Dict[str, Any])
```

Record a workload.

**Arguments**:

- `def_name` - Definition name
- `runtime_args` - Runtime arguments from sig.bind_partial

<a id="flashinfer_bench.tracer.tracer.Tracer.cuda_graph_scope"></a>

#### cuda\_graph\_scope

```python
def cuda_graph_scope()
```

Context manager for CUDA Graph collection.

<a id="flashinfer_bench.tracer.tracer.Tracer.snapshot_graph_tensors"></a>

#### snapshot\_graph\_tensors

```python
def snapshot_graph_tensors()
```

Snapshot tensors from CUDA Graph entries.

<a id="flashinfer_bench.tracer.tracer.Tracer.flush"></a>

#### flush

```python
def flush() -> Dict[str, Any]
```

Deduplicate and write collected workloads to disk.

**Returns**:

  Statistics about the flush operation

<a id="flashinfer_bench.tracer.tracer.Tracer._write_representatives"></a>

#### \_write\_representatives

```python
def _write_representatives(def_name: str, reps: List[TraceEntry],
                           output_path: Path)
```

<a id="flashinfer_bench.tracer.tracer.Tracer._save_tensors"></a>

#### \_save\_tensors

```python
def _save_tensors(def_name: str, workload_uuid: str,
                  tensors: Dict[str, torch.Tensor]) -> Path
```

<a id="flashinfer_bench.tracer.tracer.Tracer._cleanup"></a>

#### \_cleanup

```python
def _cleanup()
```

Cleanup handler for atexit.

<a id="flashinfer_bench.tracer.tracer.Tracer._signal_handler"></a>

#### \_signal\_handler

```python
def _signal_handler(signum, frame)
```

Signal handler for SIGTERM/SIGINT.

<a id="flashinfer_bench.tracer.tracer.enable_tracing"></a>

#### enable\_tracing

```python
def enable_tracing(rules: Optional[Dict[str, TracingRule]] = None,
                   out_dir: Optional[Path] = None,
                   blob_dir: Optional[Path] = None) -> Tracer
```

Enable tracing with the given tracing rule set.

Creates or replaces the process-wide singleton tracer.
If replacing, flushes the previous instance first.

**Arguments**:

- `rules` - A set of tracing rules. Default is `tracing_rules.fib_full_tracing`
- `out_dir` - Output directory for traces. Default is FIB_DATASET_PATH/traces/workloads
- `blob_dir` - Blob directory for safetensors. Default is FIB_DATASET_PATH/blob/workloads


**Returns**:

  The new tracer instance

<a id="flashinfer_bench.tracer.tracer.get_tracer"></a>

#### get\_tracer

```python
def get_tracer() -> Optional[Tracer]
```

Get the current tracer instance.

<a id="flashinfer_bench.tracer.tracer.disable_tracing"></a>

#### disable\_tracing

```python
def disable_tracing()
```

Disable tracing and flush any pending data.

<a id="flashinfer_bench.tracer.tracer._torch_dtype_from_def"></a>

#### \_torch\_dtype\_from\_def

```python
def _torch_dtype_from_def(def_dtype: str)
```

<a id="flashinfer_bench.tracer.tracer._axis_value"></a>

#### \_axis\_value

```python
def _axis_value(definition, axes: Dict[str, Any], axis_name: str) -> int
```

<a id="flashinfer_bench.tracer.tracer._materialize_shape"></a>

#### \_materialize\_shape

```python
def _materialize_shape(definition: Definition, axes: Dict[str, Any],
                       shape_spec) -> Tuple[int, ...]
```

<a id="flashinfer_bench.tracer.types"></a>

# flashinfer\_bench.tracer.types

<a id="flashinfer_bench.tracer.types.annotations"></a>

## annotations

<a id="flashinfer_bench.tracer.types.dataclass"></a>

## dataclass

<a id="flashinfer_bench.tracer.types.Any"></a>

## Any

<a id="flashinfer_bench.tracer.types.Callable"></a>

## Callable

<a id="flashinfer_bench.tracer.types.Dict"></a>

## Dict

<a id="flashinfer_bench.tracer.types.Hashable"></a>

## Hashable

<a id="flashinfer_bench.tracer.types.List"></a>

## List

<a id="flashinfer_bench.tracer.types.Optional"></a>

## Optional

<a id="flashinfer_bench.tracer.types.Set"></a>

## Set

<a id="flashinfer_bench.tracer.types.Union"></a>

## Union

<a id="flashinfer_bench.tracer.types.TracingRule"></a>

## TracingRule Objects

```python
@dataclass
class TracingRule()
```

Defines how to collect and deduplicate workloads for a definition.

<a id="flashinfer_bench.tracer.types.TracingRule.tensors_to_dump"></a>

#### tensors\_to\_dump

Which inputs to persist. List[str] for static selection, Callable for dynamic.

<a id="flashinfer_bench.tracer.types.TracingRule.dedup_policy"></a>

#### dedup\_policy

Final in-group deduplication decision. Returns the representatives.

<a id="flashinfer_bench.tracer.types.TracingRule.dedup_keys"></a>

#### dedup\_keys

Blocking function for candidate partitioning during dedup.

<a id="flashinfer_bench.tracer.types.TraceEntry"></a>

## TraceEntry Objects

```python
@dataclass
class TraceEntry()
```

In-memory buffer entry for collected workloads.

<a id="flashinfer_bench.tracer.types.TraceEntry.def_name"></a>

#### def\_name

<a id="flashinfer_bench.tracer.types.TraceEntry.axes"></a>

#### axes

<a id="flashinfer_bench.tracer.types.TraceEntry.definition_input_names"></a>

#### definition\_input\_names

<a id="flashinfer_bench.tracer.types.TraceEntry.picked"></a>

#### picked

<a id="flashinfer_bench.tracer.types.TraceEntry.order"></a>

#### order

<a id="flashinfer_bench.tracer.types.TraceEntry.cuda_graph_snapshot"></a>

#### cuda\_graph\_snapshot

<a id="flashinfer_bench.utils"></a>

# flashinfer\_bench.utils

<a id="flashinfer_bench.utils.os"></a>

## os

<a id="flashinfer_bench.utils.platform"></a>

## platform

<a id="flashinfer_bench.utils.sys"></a>

## sys

<a id="flashinfer_bench.utils.Dict"></a>

## Dict

<a id="flashinfer_bench.utils.List"></a>

## List

<a id="flashinfer_bench.utils.torch"></a>

## torch

<a id="flashinfer_bench.utils.Environment"></a>

## Environment

<a id="flashinfer_bench.utils.torch_dtype_from_def"></a>

#### torch\_dtype\_from\_def

```python
def torch_dtype_from_def(dtype_str: str)
```

<a id="flashinfer_bench.utils.list_cuda_devices"></a>

#### list\_cuda\_devices

```python
def list_cuda_devices() -> List[str]
```

<a id="flashinfer_bench.utils.env_snapshot"></a>

#### env\_snapshot

```python
def env_snapshot(device: str) -> Environment
```

<a id="flashinfer_bench.utils.hardware_from_device"></a>

#### hardware\_from\_device

```python
def hardware_from_device(device: str) -> str
```

<a id="flashinfer_bench.utils.redirect_stdio_to_file"></a>

#### redirect\_stdio\_to\_file

```python
def redirect_stdio_to_file(log_path: str) -> None
```
