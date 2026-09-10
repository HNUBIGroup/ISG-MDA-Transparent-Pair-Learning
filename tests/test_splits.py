from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isg_mda.splits import audit_split, build_split


class SplitTest(unittest.TestCase):
    def setUp(self):
        positives = [(m, d) for m in range(8) for d in range(5) if (m + d) % 3 == 0]
        negatives = [(m, d) for m in range(8) for d in range(5) if (m + d) % 3 != 0]
        self.positive = pd.DataFrame(positives, columns=["mirna_index", "drug_index"])
        self.negative = pd.DataFrame(negatives, columns=["mirna_index", "drug_index"])

    def test_random_is_deterministic(self):
        first = build_split(self.positive, self.negative, "random", 0.2, 4)
        second = build_split(self.positive, self.negative, "random", 0.2, 4)
        self.assertTrue(first[0].equals(second[0]))
        self.assertTrue(audit_split(first[0], first[1], "random")["passed"])

    def test_cold_entity_disjoint(self):
        train, test, _ = build_split(self.positive, self.negative, "mirna_cold", 0.25, 2)
        self.assertTrue(audit_split(train, test, "mirna_cold")["passed"])


if __name__ == "__main__":
    unittest.main()

