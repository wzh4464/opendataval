#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.utils import (
    DataProcessor,
    ModelFactory,
    compute_statistics,
    select_device,
    set_random_seeds,
)


class TestBasic(unittest.TestCase):
    def test_device_selection(self):
        device = select_device("auto")
        self.assertIn(device.type, {"cuda", "mps", "cpu"})

        device_cpu = select_device("cpu")
        self.assertEqual(device_cpu.type, "cpu")

    def test_random_seeds(self):
        set_random_seeds(42)
        torch_rand1 = torch.rand(5)
        np_rand1 = np.random.rand(5)

        set_random_seeds(42)
        torch_rand2 = torch.rand(5)
        np_rand2 = np.random.rand(5)

        self.assertTrue(torch.allclose(torch_rand1, torch_rand2))
        self.assertTrue(np.allclose(np_rand1, np_rand2))

    def test_model_factory(self):
        supported = ModelFactory.get_supported_models()
        self.assertIn("bert", supported)
        self.assertIn("mlp", supported)
        self.assertIn("logistic", supported)

        # BERT may require network; skip test if it fails due to env
        try:
            _ = ModelFactory.create_model(
                model_type="bert",
                output_dim=2,
                pretrained_model_name="distilbert-base-uncased",
            )
        except Exception as e:
            self.skipTest(f"Skip BERT creation due to env: {e}")

        mlp = ModelFactory.create_model(model_type="mlp", input_dim=100, output_dim=2)
        self.assertIsNotNone(mlp)

        lr = ModelFactory.create_model(
            model_type="logistic", input_dim=100, output_dim=2
        )
        self.assertIsNotNone(lr)

    def test_statistics(self):
        data = np.random.randn(100)
        stats = compute_statistics(data)

        expected_keys = {"mean", "std", "min", "max", "median", "count"}
        self.assertEqual(set(stats.keys()), expected_keys)
        self.assertEqual(stats["count"], 100)
        self.assertIsInstance(stats["mean"], float)

    def test_data_processor(self):
        x_train, y_train, x_valid, y_valid, x_test, y_test, fetcher = (
            DataProcessor.prepare_data(
                dataset_name="iris",
                train_count=50,
                valid_count=25,
                test_count=25,
                random_state=42,
            )
        )

        self.assertEqual(len(x_train), 50)
        self.assertEqual(len(x_valid), 25)
        self.assertEqual(len(x_test), 25)

        # Shape metadata
        self.assertTrue(hasattr(fetcher, "covar_dim"))
        self.assertTrue(hasattr(fetcher, "label_dim"))


if __name__ == "__main__":
    unittest.main()
