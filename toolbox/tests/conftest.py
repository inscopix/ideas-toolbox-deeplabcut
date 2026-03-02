import pytest


@pytest.fixture(scope="session", autouse=True)
def set_multiprocessing_spawn():
    """Unit tests which execute deep-learning models are run within a subprocess,
    in order to ensure gpu memory is completely freed between test cases,
    preventing out-of-memory errors.

    This fixture is automatically run once before the testing session starts 
    in order configure the correct start method for subprocesses executed during unit tests.
    
    By default the start method in multiprocessing is `fork`, which creates a copy 
    of the parent process memory for the child process. This is faster but causes
    problems with deep-learning libraries and cause API calls to hang indefinitely.
    The start method `spawn` ensures a completely new python process is created for testing,
    preventing memory and communication errors.
    """
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)


@pytest.fixture
def output_dir(tmp_path):
    """Construct a path for the directory where outputs can be stored,
    and cleans up outputs after each test finishes.
    """
    yield tmp_path
