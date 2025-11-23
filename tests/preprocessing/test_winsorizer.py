import numpy as np
import pytest

from ps3.preprocessing import Winsorizer

# TODO: Test your implementation of a simple Winsorizer
@pytest.mark.parametrize(
    "lower_quantile, upper_quantile", [(0, 1), (0.05, 0.95), (0.5, 0.5)]
)
def test_winsorizer(lower_quantile, upper_quantile):

    X = np.random.normal(0, 1, 1000)

    winsoriser = Winsorizer(lower_quantile, upper_quantile)
    winsoriser.fit(X)

    lq = winsoriser.lower_quantile_
    uq = winsoriser.upper_quantile_
    assert lq is not None
    assert uq is not None
    # Make sure there are some entries outside (or on) the bounds before transformation
    assert any(X >= uq)
    assert any(X <= lq)

    X_transformed = winsoriser.transform(X)
    # Make sure there are no entries outside bounds after transformation
    assert not all(X_transformed > uq)
    assert not all(X_transformed < lq)