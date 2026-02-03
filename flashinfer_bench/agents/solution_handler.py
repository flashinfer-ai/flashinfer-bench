"""Solution file handling utilities for LLM agents."""

import json
from pathlib import Path

from flashinfer_bench.data import BuildSpec, Solution, SourceFile

SOLUTION_METADATA_FILE = "SOLUTION.md"
VALID_SOURCE_EXTENSIONS = {".py", ".cu", ".cuh", ".cpp", ".c", ".h", ".hpp", ".cc", ".cxx"}


def _generate_metadata_md(solution: Solution) -> str:
    """Generate markdown content for solution metadata."""
    lines = [
        f"# {solution.name}",
        "",
        f"**Definition:** {solution.definition}",
        f"**Author:** {solution.author}",
        "",
        "## Description",
        "",
        solution.description or "_No description provided._",
        "",
        "## Build Specification",
        "",
        "```json",
        json.dumps(solution.spec.model_dump(), indent=2),
        "```",
        "",
        "## Source Files",
        "",
    ]

    for source in solution.sources:
        lines.append(f"- `{source.path}`")

    return "\n".join(lines)


def extract_solution_to_files(solution: Solution, base_path: str) -> str:
    """Extract a Solution object to a directory of files.

    Creates a directory containing:
    - All source files from the solution
    - A metadata file (SOLUTION.md) with build specs and metadata

    Parameters
    ----------
    solution : Solution
        Solution object to extract.
    base_path : str
        Base directory path where files will be created.

    Returns
    -------
    str
        Path to the created solution directory.
    """
    base_path_obj = Path(base_path)

    if not base_path_obj.exists():
        raise ValueError(f"Base path does not exist: {base_path}")

    if not base_path_obj.is_dir():
        raise ValueError(f"Base path is not a directory: {base_path}")

    solution_dir = base_path_obj / solution.name
    solution_dir.mkdir(exist_ok=True)

    for source in solution.sources:
        file_path = solution_dir / source.path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(source.content)

    metadata_path = solution_dir / SOLUTION_METADATA_FILE
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write(_generate_metadata_md(solution))

    return str(solution_dir)


def pack_solution_from_files(
    path: str, spec: BuildSpec, name: str, definition: str, author: str, description: str = ""
) -> Solution:
    """Pack a directory of files into a Solution object.

    Only includes source code files (.py, .cu, .cuh, .cpp, .c, .h, .hpp).

    Parameters
    ----------
    path : str
        Path to directory containing solution source files.
    spec : BuildSpec
        BuildSpec object specifying build configuration.
    name : str
        Solution name.
    definition : str
        Definition name.
    author : str
        Author name.
    description : str, optional
        Solution description. Default is "".

    Returns
    -------
    Solution
        Solution object constructed from files.
    """
    path_obj = Path(path)

    if not path_obj.exists():
        raise ValueError(f"Path does not exist: {path}")

    if not path_obj.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    sources = []

    for file_path in sorted(path_obj.iterdir()):
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()
        if ext not in VALID_SOURCE_EXTENSIONS:
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        sources.append(SourceFile(path=file_path.name, content=content))

    if not sources:
        raise ValueError(f"No source files found in directory: {path}")

    solution = Solution(
        name=name,
        definition=definition,
        author=author,
        description=description,
        spec=spec,
        sources=sources,
    )

    return solution
