import pytest

from flashinfer_bench.data import (
    BuildSpec,
    Solution,
    SourceFile,
    SupportedLanguages,
)


def test_sourcefile_validation_python():
    # Valid python content
    SourceFile(path="main.py", content="def run():\n    pass\n")
    # Empty path
    with pytest.raises(ValueError):
        SourceFile(path="", content="def run(): pass")
    # Non-string content
    with pytest.raises(ValueError):
        SourceFile(path="main.py", content=123)  # type: ignore[arg-type]
    # Invalid python
    with pytest.raises(ValueError):
        SourceFile(path="main.py", content="def run(: pass")


def test_buildspec_validation():
    # Valid
    BuildSpec(
        language=SupportedLanguages.PYTHON,
        target_hardware=["cuda"],
        entry_point="main.py::run",
        dependencies=["numpy"],
        build_commands=["echo build"],
    )
    # Invalid entry format
    with pytest.raises(ValueError):
        BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["cuda"],
            entry_point="main.py",  # missing ::
        )
    # Invalid file extension
    with pytest.raises(ValueError):
        BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["cuda"],
            entry_point="main.txt::run",
        )
    # Invalid target_hardware list and dependencies types
    with pytest.raises(ValueError):
        BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=[],
            entry_point="main.py::run",
        )
    with pytest.raises(ValueError):
        BuildSpec(
            language=SupportedLanguages.PYTHON,
            target_hardware=["cuda"],
            entry_point="main.py::run",
            dependencies=[1],  # type: ignore[list-item]
        )


def test_solution_validation_and_helpers():
    spec = BuildSpec(
        language=SupportedLanguages.TRITON,
        target_hardware=["cuda"],
        entry_point="main.py::run",
    )
    s1 = SourceFile(path="main.py", content="def run():\n    pass\n")
    s2 = SourceFile(path="util.py", content="x=1\n")
    sol = Solution(
        name="sol1",
        definition="def1",
        author="me",
        spec=spec,
        sources=[s1, s2],
    )
    assert sol.get_entry_source() is s1
    assert sol.requires_build() is False

    # CUDA requires build
    cuda_spec = BuildSpec(
        language=SupportedLanguages.CUDA,
        target_hardware=["cuda"],
        entry_point="main.py::run",
    )
    sol2 = Solution(
        name="sol2",
        definition="def1",
        author="you",
        spec=cuda_spec,
        sources=[s1],
    )
    assert sol2.requires_build() is True

    # Duplicate source paths
    with pytest.raises(ValueError):
        Solution(
            name="dup",
            definition="def1",
            author="x",
            spec=spec,
            sources=[s1, s1],
        )
    # Entry not present
    with pytest.raises(ValueError):
        Solution(
            name="missing_entry",
            definition="def1",
            author="x",
            spec=spec,
            sources=[s2],
        )
