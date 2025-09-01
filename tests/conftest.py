import pytest


@pytest.fixture
def sample_reference_code():
    # Minimal valid reference function
    return "def run(A, B):\n    return A\n"
