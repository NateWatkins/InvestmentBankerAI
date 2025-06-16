import pytest

pd = pytest.importorskip("pandas")

from scripts.compute_ema import compute_ema

@pytest.fixture
def sample_df():
    return pd.DataFrame({"Close": [10, 20, 30, 40, 50]})

def test_compute_ema_values(sample_df):
    period = 3
    result = compute_ema(sample_df.copy(), period)
    expected = sample_df["Close"].ewm(span=period, adjust=False).mean()
    pd.testing.assert_series_equal(result[f"EMA{period}"], expected, check_names=False)
