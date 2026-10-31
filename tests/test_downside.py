"""Loss gradients, matched controls, and validation-only selection checks."""
import unittest
import numpy as np
import torch

from scripts.downside_compare import metrics, select_candidate, validation_utility
from utils.trading import (downside_second_moment, profit_aware_loss, profit_per_step,
                           smooth_signal)


class DownsideTests(unittest.TestCase):
    def test_lower_partial_moment_denominator_and_gradient(self):
        returns = torch.tensor([-0.02, 0.01, 0.0, -0.01], dtype=torch.float64, requires_grad=True)
        penalty = downside_second_moment(returns)
        self.assertAlmostEqual(penalty.item(), (0.02 ** 2 + 0.01 ** 2) / 4)
        penalty.backward()
        np.testing.assert_allclose(returns.grad.numpy(), [-0.01, 0.0, 0.0, -0.005])

    def test_zero_gamma_exactly_preserves_original_loss(self):
        pred = torch.tensor([0.002, -0.003, 0.001], dtype=torch.float64)
        actual = torch.tensor([-0.01, 0.02, 0.003], dtype=torch.float64)
        expected = -profit_per_step(smooth_signal(pred, 90), actual).mean()
        self.assertTrue(torch.equal(profit_aware_loss(pred, actual, 90, downside_gamma=0), expected))
        net = profit_per_step(smooth_signal(pred, 90), actual)
        extended = profit_aware_loss(pred, actual, 90, downside_gamma=30)
        self.assertAlmostEqual((extended - expected).item(), 30 * downside_second_moment(net).item())

    def test_gradient_through_position_cost_and_downside(self):
        pred = torch.tensor([0.002, -0.003, 0.001], dtype=torch.float64, requires_grad=True)
        actual = torch.tensor([-0.01, 0.02, 0.003], dtype=torch.float64)
        self.assertTrue(torch.autograd.gradcheck(
            lambda values: profit_aware_loss(values, actual, 90, downside_gamma=30), (pred,)))

    def test_common_validation_score_and_tie(self):
        actual = np.array([0.01, -0.01, 0.005])
        candidates = [{"kappa": kappa, "validation_utility": validation_utility(
            np.full(3, pred), actual, np.tanh(90 * np.full(3, pred)), 90)}
                      for kappa, pred in [(0.25, 0.002), (1.0, 0.001), (4.0, 0.0001)]]
        # A common score must use the fixed reference downside coefficient,
        # not a candidate's training gamma or its differently weighted loss.
        pred = np.full(3, 0.001)
        signal = np.tanh(90 * pred)
        net = signal * actual
        net[0] -= signal[0] * 0.00296
        expected = net.mean() - 0.7 * np.mean((pred-actual)**2) - 90 * np.mean(np.maximum(-net,0)**2)
        self.assertAlmostEqual(candidates[1]['validation_utility'], expected)
        self.assertEqual(select_candidate(candidates)['validation_utility'],
                         max(row['validation_utility'] for row in candidates))
        self.assertEqual(select_candidate([{"kappa": 4, "validation_utility": 1},
                                           {"kappa": 0.25, "validation_utility": 1}])["kappa"], 0.25)

    def test_cash_and_cost_accounting(self):
        cash = metrics([0, 0], [0.01, -0.02], [0, 0])
        self.assertEqual(cash["downside_lpm2"], 0)
        self.assertEqual(cash["geometric_return"], 0)
        self.assertEqual(cash["near_flat_share"], 1)
        outcome = metrics([0.001, -0.001], [0.01, -0.02], [0.5, -0.5])
        net = np.array([0.005 - 0.5 * 0.00296, 0.01 - 0.00396])
        self.assertAlmostEqual(outcome["additive_return"], net.sum())
        self.assertAlmostEqual(outcome["geometric_return"], np.prod(1 + net) - 1)


if __name__ == "__main__":
    unittest.main()
