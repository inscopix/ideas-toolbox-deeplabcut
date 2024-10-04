import os
import pytest
import shutil


@pytest.fixture
def output_dir():
    """Construct a path for the directory where outputs can be stored,
    and cleans up outputs after each test finishes.
    """
    _output_dir = "/ideas/outputs"
    os.makedirs(_output_dir)
    yield _output_dir
    # clean up output dir or else dlc will not re-run analysis
    shutil.rmtree(_output_dir)
