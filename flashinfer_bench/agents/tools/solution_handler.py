import os
from pathlib import Path
from typing import Optional

from flashinfer_bench import Solution, SourceFile, BuildSpec, SupportedLanguages
from flashinfer_bench.data import load_json_file, save_json_file


class SolutionHandler:
    """
    Handler for converting between Solution objects and file system representations.
    """

    @staticmethod
    def _can_convert(path: str) -> bool:
        path_obj = Path(path)

        if not path_obj.exists():
            return False

        if not path_obj.is_dir():
            return False

        # FIB solution sources do not support nested directories
        for item in path_obj.iterdir():
            if item.is_dir():
                return False

        return True

    @staticmethod
    def to_files(solution: Solution, base_path: str) -> str:
        """
        Convert a Solution object to a directory of files.

        Args:
            solution: Solution object to convert
            base_path: Base directory path where files will be created

        Returns:
            Path to the created solution directory

        Raises:
            ValueError: If base_path doesn't exist or is not a directory
        """
        base_path_obj = Path(base_path)

        if not base_path_obj.exists():
            raise ValueError(f"Base path does not exist: {base_path}")

        if not base_path_obj.is_dir():
            raise ValueError(f"Base path is not a directory: {base_path}")

        # Create solution directory
        solution_dir = base_path_obj / solution.name
        solution_dir.mkdir(exist_ok=True)

        # Write each source file
        for source in solution.sources:
            file_path = solution_dir / source.path

            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write file content
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(source.content)

        # Save solution metadata as JSON
        metadata_path = solution_dir / "solution_metadata.json"
        save_json_file(solution, metadata_path)

        return str(solution_dir)

    @staticmethod
    def from_files(
        path: str,
        solution_name: Optional[str] = None,
        definition: Optional[str] = None,
        author: Optional[str] = None,
    ) -> Solution:
        """
        Convert a directory of files to a Solution object.

        Args:
            path: Path to directory containing solution files
            solution_name: Solution name (required if no solution_metadata.json)
            definition: Definition name (required if no solution_metadata.json)
            author: Author name (required if no solution_metadata.json)

        Returns:
            Solution object constructed from files

        Raises:
            ValueError: If path is invalid or required metadata is missing
        """
        path_obj = Path(path)

        if not path_obj.exists():
            raise ValueError(f"Path does not exist: {path}")

        if not path_obj.is_dir():
            raise ValueError(f"Path is not a directory: {path}")

        # Check for metadata file
        metadata_path = path_obj / "solution_metadata.json"

        if metadata_path.exists():
            # Load solution from metadata
            solution = load_json_file(Solution, metadata_path)
            return solution

        # Otherwise, construct solution from files
        if not SolutionHandler._can_convert(str(path_obj)):
            raise ValueError(
                f"Directory contains subdirectories. Only single-level directories are supported: {path}"
            )

        if not solution_name or not definition or not author:
            raise ValueError(
                "solution_name, definition, and author are required when no solution_metadata.json is present"
            )

        # Read all files in directory
        sources = []
        entry_point = None
        language = None

        for file_path in sorted(path_obj.iterdir()):
            if file_path.is_file() and file_path.name != "solution_metadata.json":
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                relative_path = file_path.name
                sources.append(SourceFile(path=relative_path, content=content))

                # Infer language and entry point from file extensions
                if file_path.suffix == ".py":
                    language = SupportedLanguages.PYTHON
                    if entry_point is None:
                        entry_point = f"{relative_path}::run"
                elif file_path.suffix == ".cu" or file_path.name == "main.cpp":
                    language = SupportedLanguages.CUDA
                    if file_path.name == "main.cpp":
                        entry_point = "main.cpp::run"

        if not sources:
            raise ValueError(f"No source files found in directory: {path}")

        if language is None:
            language = SupportedLanguages.PYTHON

        if entry_point is None:
            # Default to first source file
            entry_point = f"{sources[0].path}::run"

        # Create BuildSpec
        spec = BuildSpec(
            language=language,
            target_hardware=["H100"],  # Default
            entry_point=entry_point,
        )

        # Create Solution
        solution = Solution(
            name=solution_name,
            definition=definition,
            author=author,
            spec=spec,
            sources=sources,
        )

        return solution

    @staticmethod
    def update_solution_files(solution: Solution, base_path: str) -> str:
        """
        Update existing solution files or create new ones.

        Args:
            solution: Solution object to write
            base_path: Base directory path where files will be updated/created

        Returns:
            Path to the solution directory
        """
        return SolutionHandler.to_files(solution, base_path)

    @staticmethod
    def get_source_content(solution: Solution, file_path: str) -> Optional[str]:
        """
        Get the content of a specific source file from a solution.

        Args:
            solution: Solution object
            file_path: Relative path of the source file

        Returns:
            Content of the source file, or None if not found
        """
        for source in solution.sources:
            if source.path == file_path:
                return source.content
        return None

    @staticmethod
    def update_source_content(solution: Solution, file_path: str, new_content: str) -> Solution:
        """
        Update the content of a specific source file in a solution.

        Args:
            solution: Solution object
            file_path: Relative path of the source file
            new_content: New content for the file

        Returns:
            New Solution object with updated content

        Raises:
            ValueError: If file_path not found in solution
        """
        updated_sources = []
        found = False

        for source in solution.sources:
            if source.path == file_path:
                updated_sources.append(SourceFile(path=file_path, content=new_content))
                found = True
            else:
                updated_sources.append(source)

        if not found:
            raise ValueError(f"Source file not found in solution: {file_path}")

        # Create new solution with updated sources
        return Solution(
            name=solution.name,
            definition=solution.definition,
            author=solution.author,
            spec=solution.spec,
            sources=updated_sources,
            description=solution.description,
        )
