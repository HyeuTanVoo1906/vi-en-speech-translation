"""Synthetic failure-mode checks for the audio alignment audit."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from validate_train_alignment import SR, find_alignment, ncc


class AlignmentChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/"source.flac"
        self.rng = np.random.default_rng(13)
        self.source = self.rng.normal(0, .1, SR*50).astype("float32")
        self.source[:SR*5] = 0
        self.source[SR*40:] = 0
        sf.write(self.path, self.source, SR)

    def tearDown(self):
        self.temp.cleanup()

    def test_known_offsets_and_silent_windows(self):
        for nominal, actual, status in [(25, 25, "pass"), (25, 25.42, "adjust"),
                                         (25, 24.65, "adjust"), (20, 33, "adjust")]:
            clip = self.source[round(actual*SR):round((actual+5)*SR)]
            result = find_alignment(self.path, clip, nominal, nominal+5)
            self.assertEqual(result["status"], status, result)
            self.assertAlmostEqual(result["offset_s"], actual-nominal, places=3)

    def test_unrelated_audio_is_not_accepted(self):
        result = find_alignment(self.path, self.rng.normal(0, .1, SR*5), 25, 30)
        self.assertEqual(result["status"], "review")

    def test_corrupt_last_third_is_not_accepted(self):
        clip = self.source[SR*25:SR*30].copy()
        clip[-SR:] = self.rng.normal(0, .1, SR)
        self.assertEqual(find_alignment(self.path, clip, 25, 30)["status"], "review")

    def test_zero_source_has_zero_correlation(self):
        scores = ncc(np.zeros(20000), self.rng.normal(size=2000))
        self.assertTrue(np.all(scores == 0))

    def test_nearly_twenty_seconds_history_does_not_reject_alignment(self):
        clip = self.source[round(19.95*SR):round(24.95*SR)]
        result = find_alignment(self.path, clip, 20, 25)
        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["history_20s_ready"])


if __name__ == "__main__":
    unittest.main()
