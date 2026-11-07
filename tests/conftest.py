import pytest

import f1lap


@pytest.fixture(scope="session")
def race():
    return f1lap.synthetic_race(n_drivers=20, seed=7, name="t7")
