from typing import Callable, List, Protocol

from flashinfer_bench.tracing.workload_entry import WorkloadEntry


class FilterPolicy(Protocol):
    """Protocol for workload deduplication policy.

    A filter policy maintains internal state and supports both online and offline
    deduplication strategies. Entries are submitted one at a time via submit(),
    and selected entries are retrieved via drain().
    """

    def submit(self, entry: WorkloadEntry) -> None:
        """Submit a workload entry for deduplication consideration.

        Parameters
        ----------
        entry : WorkloadEntry
            The workload entry to submit.
        """
        ...

    def drain(self) -> List[WorkloadEntry]:
        """Drain and return all selected entries.

        Returns
        -------
        List[WorkloadEntry]
            List of entries that passed the deduplication policy.
            After calling this method, the internal buffer is cleared.
        """
        ...

    def reset(self) -> None:
        """Reset the internal state of the deduplication policy.

        This method should be called when starting a new deduplication session
        to clear any cached state or statistics.
        """
        ...


FilterPolicyFactory = Callable[[], FilterPolicy]
"""Factory function for filter policy."""
