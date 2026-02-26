import pytest


@pytest.fixture
def output_dir(tmp_path):
    """Construct a path for the directory where outputs can be stored,
    and cleans up outputs after each test finishes.
    """
    yield tmp_path
