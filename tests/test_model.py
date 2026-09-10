from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isg_mda.model import InteractionPredictor


class InteractionPredictorTest(unittest.TestCase):
    def test_pair_shape_and_probability(self):
        predictor = InteractionPredictor(128)
        mirna = torch.randn(3, 128)
        drug = torch.randn(3, 128)
        self.assertEqual(tuple(predictor.pair_features(mirna, drug).shape), (3, 512))
        probabilities = predictor(mirna, drug)
        self.assertEqual(tuple(probabilities.shape), (3,))
        self.assertTrue(torch.all((probabilities >= 0) & (probabilities <= 1)))


if __name__ == "__main__":
    unittest.main()

