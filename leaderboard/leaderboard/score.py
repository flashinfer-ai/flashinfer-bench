def format_score(score: float) -> str:
    """
    Format score as a string with 3 decimal places.
    
    Args:
        score: The score in seconds.

    Returns:
        A string representing the score in microseconds with 3 decimal places.
    """
    return f"{score * 1_000_000:.3f}μs"
