import unittest

from signal_service import (
    calculate_market_momentum,
    calculate_news_sentiment,
    calculate_price_momentum,
    classify_signal,
    generate_signal,
)


class SignalServiceTests(unittest.TestCase):
    def test_empty_news_is_neutral(self):
        self.assertEqual(calculate_news_sentiment([]), 0.0)

    def test_price_momentum_is_clamped(self):
        self.assertEqual(calculate_price_momentum(10), 1.0)
        self.assertEqual(calculate_price_momentum(-10), -1.0)

    def test_relative_market_momentum(self):
        self.assertAlmostEqual(calculate_market_momentum(4, 1), 1.0)

    def test_thresholds(self):
        self.assertEqual(classify_signal(0.4), "BUY")
        self.assertEqual(classify_signal(-0.4), "SELL")
        self.assertEqual(classify_signal(0), "HOLD")

    def test_signal_schema_and_benchmark(self):
        result = generate_signal(
            symbol="INFY",
            sentiment_results=[{"score": 0.8}],
            market_data={"price": 1500, "change_percent": 2, "volume": 1000},
            benchmark_data={"change_percent": 1},
        )
        self.assertEqual(result["symbol"], "INFY")
        self.assertIn(result["signal"], {"BUY", "HOLD", "SELL"})
        self.assertIsNotNone(result["market_change_percent"])
        self.assertIn("timestamp", result)


if __name__ == "__main__":
    unittest.main()
