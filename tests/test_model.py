import unittest

import torch

from src.model import TransformerClassifier, count_parameters


class TransformerClassifierTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = TransformerClassifier(
            vocab_size=30,
            pad_idx=0,
            max_seq_len=8,
            d_model=16,
            n_heads=4,
            n_layers=2,
            d_ff=32,
            num_classes=2,
            dropout=0.0,
        )
        self.model.eval()

    def test_output_shape(self):
        input_ids = torch.tensor([
            [2, 5, 7, 0],
            [4, 8, 0, 0],
        ])
        attention_mask = input_ids != 0

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)

        self.assertEqual(tuple(logits.shape), (2, 2))

    def test_padding_tokens_do_not_change_sequence_prediction(self):
        input_ids_a = torch.tensor([[2, 5, 7, 0]])
        input_ids_b = torch.tensor([[2, 5, 7, 12]])
        attention_mask = torch.tensor([[1, 1, 1, 0]])

        with torch.no_grad():
            logits_a = self.model(input_ids_a, attention_mask)
            logits_b = self.model(input_ids_b, attention_mask)

        self.assertTrue(torch.allclose(logits_a, logits_b, atol=1e-6))

    def test_parameter_count_is_positive(self):
        self.assertGreater(count_parameters(self.model), 0)

    def test_incompatible_head_count_is_rejected(self):
        with self.assertRaises(ValueError):
            TransformerClassifier(
                vocab_size=30,
                pad_idx=0,
                d_model=15,
                n_heads=4,
            )


if __name__ == "__main__":
    unittest.main()
