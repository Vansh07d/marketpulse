import unittest

import latency


class FakePipeline:
    def __init__(self, store):
        self.store = store

    def lpush(self, key, value):
        self.store.setdefault(key, []).insert(0, value)

    def ltrim(self, key, start, stop):
        self.store[key] = self.store[key][start:stop + 1]

    def expire(self, key, seconds):
        pass

    def execute(self):
        pass


class FakeRedis:
    def __init__(self):
        self.store = {}
        self.client = self

    def pipeline(self):
        return FakePipeline(self.store)

    def lrange(self, key, start, stop):
        return self.store.get(key, [])[start:stop + 1]


ARTICLE = {
    "id": "a1", "title": "Infosys wins deal",
    "published_at": "2026-10-24T08:00:00+00:00",
    "ingested_at": "2026-10-24T08:04:00+00:00",
    "scored_at": "2026-10-24T08:04:02+00:00",
}
SIGNALLED = "2026-10-24T08:04:05+00:00"


class LatencyTests(unittest.TestCase):
    def test_build_record_durations(self):
        record = latency.build_record(ARTICLE, SIGNALLED)
        self.assertEqual(record["ingest_lag_s"], 240.0)
        self.assertEqual(record["inference_s"], 2.0)
        self.assertEqual(record["signal_s"], 3.0)
        self.assertEqual(record["end_to_end_s"], 245.0)

    def test_missing_or_bad_timestamps_give_none(self):
        record = latency.build_record({"title": "x", "published_at": "garbage"}, SIGNALLED)
        self.assertIsNone(record["end_to_end_s"])
        self.assertIsNone(record["inference_s"])

    def test_future_published_time_is_clamped_to_zero(self):
        article = {**ARTICLE, "published_at": "2026-10-24T09:00:00+00:00"}
        self.assertEqual(latency.build_record(article, SIGNALLED)["end_to_end_s"], 0.0)

    def test_percentiles(self):
        values = [1, 2, 3, 4, 5]
        self.assertEqual(latency.percentile(values, 50), 3)
        self.assertAlmostEqual(latency.percentile(values, 95), 4.8)
        self.assertIsNone(latency.percentile([], 50))

    def test_summarize_ignores_missing_values(self):
        records = [{"end_to_end_s": 10.0}, {"end_to_end_s": None}, {"end_to_end_s": 30.0}]
        summary = latency.summarize(records)["end_to_end_s"]
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["p50"], 20.0)

    def test_record_and_read_back(self):
        redis = FakeRedis()
        latency.record_latencies(redis, "infy", [ARTICLE], SIGNALLED)
        stored = latency.get_latencies(redis, "INFY")
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["id"], "a1")

    def test_records_are_bounded(self):
        redis = FakeRedis()
        for _ in range(latency.MAX_RECORDS + 20):
            latency.record_latencies(redis, "INFY", [ARTICLE], SIGNALLED)
        self.assertEqual(len(latency.get_latencies(redis, "INFY", limit=10_000)), latency.MAX_RECORDS)

    def test_format_seconds(self):
        self.assertEqual(latency.format_seconds(None), "N/A")
        self.assertEqual(latency.format_seconds(12.34), "12.3s")
        self.assertEqual(latency.format_seconds(300), "5.0 min")
        self.assertEqual(latency.format_seconds(7200), "2.0 h")


if __name__ == "__main__":
    unittest.main()