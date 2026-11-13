import tempfile
import unittest
from pathlib import Path

import numpy as np

from csv_io import load_kaggle_csv


class LoadKaggleCsvTests(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shots.csv"
            path.write_text(text, encoding="utf-8")
            return load_kaggle_csv(path)

    def test_rejects_partial_observable_labels(self):
        for columns in ("L0", "L1,L3", "L0,L1,L2"):
            with self.subTest(columns=columns):
                values = ",".join("1" for _ in columns.split(","))
                with self.assertRaisesRegex(ValueError, "observable"):
                    self.load(f"id,basis,D0,{columns}\nshot,X,0,{values}\n")

    def test_rejects_duplicate_observable_labels(self):
        with self.assertRaisesRegex(ValueError, "observable"):
            self.load("id,basis,D0,L0,L1,L2,L3,L0\nshot,X,0,0,1,0,1,1\n")

    def test_reads_all_labels_in_logical_order(self):
        data = self.load("id,basis,D0,L3,L0,L2,L1\nshot,X,0,1,0,0,1\n")
        np.testing.assert_array_equal(data["observable_flips"], [[False, True, False, True]])

    def test_accepts_unlabeled_input(self):
        data = self.load("id,basis,D0\nshot,Z,1\n")
        self.assertNotIn("observable_flips", data)
        np.testing.assert_array_equal(data["detector_events"], [[True]])


if __name__ == "__main__":
    unittest.main()
