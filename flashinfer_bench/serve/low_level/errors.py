"""Error types for the low-level GPU server."""

from __future__ import annotations


class InvalidProgramError(Exception):
    def __init__(self, message: str):
        """Initialize an invalid-program error.

        Parameters
        ----------
        message
            Human-readable validation error message.

        Returns
        -------
        None
            This constructor initializes the exception state in place.
        """
        self.message = message
        super().__init__(message)


class ExecutionFailedError(Exception):
    def __init__(self, message: str, instruction_index: int | None = None):
        """Initialize an execution-failed error.

        Parameters
        ----------
        message
            Human-readable execution failure message.
        instruction_index
            Optional zero-based instruction index associated with the failure.

        Returns
        -------
        None
            This constructor initializes the exception state in place.
        """
        self.message = message
        self.instruction_index = instruction_index
        super().__init__(message)


class TimeoutError(Exception):
    def __init__(self, timeout_seconds: float):
        """Initialize a timeout error.

        Parameters
        ----------
        timeout_seconds
            Effective timeout exceeded by the request in seconds.

        Returns
        -------
        None
            This constructor initializes the exception state in place.
        """
        self.timeout_seconds = timeout_seconds
        super().__init__(f"Timeout after {timeout_seconds}s")
