from __future__ import annotations

import numpy as np
import torch

# Philippine Stock Exchange round-trip friction, as a fraction of traded value.
# Both legs pay broker commission (0.25%), 12% VAT on that commission, and the
# PSE transaction + SCCP clearing fees (0.296% total); the sell leg also pays
# the 0.1% stock transaction tax (0.396% total). Used exactly as given.
PH_BUY_COST = 0.00296
PH_SELL_COST = 0.00396


def calibrate_alpha(train_returns: np.ndarray) -> float:
    """Pick ``alpha`` so a one-std-dev predicted return saturates ``tanh`` to ~0.76.

    ``alpha = 1 / std(train_returns)``. Daily equity returns are O(1%), so a
    fixed ``alpha`` tuned for one asset is badly miscalibrated for another with
    different volatility (e.g. SPY vs NVDA) — deriving it from the *training*
    split's own return scale keeps ``tanh(alpha * pred)`` meaningfully
    sensitive across assets instead of sitting in its near-linear dead zone
    (``alpha`` too small) or saturating for every prediction (``alpha`` too
    large).
    """
    std = float(np.std(train_returns))
    if std <= 0:
        return 1.0
    return 1.0 / std


def smooth_signal(predicted_return: torch.Tensor, alpha: float) -> torch.Tensor:
    """Continuous trading position: ``tanh(alpha * predicted_return)``.

    Unlike the old straight-through ``sign`` hack, forward and backward use the
    *same* smooth function, so gradients remain alive for every ``|r|`` (no
    saturation) and there is no fake forward/backward coupling to maintain.
    The output is a position size in ``(-1, 1)``, where magnitude scales how
    confident we are and sign is the direction.
    """
    return torch.tanh(alpha * predicted_return)


def thresholded_signal(
    predicted_return: torch.Tensor,
    alpha: float,
    threshold: float = 0.0,
) -> torch.Tensor:
    """Discrete long/flat/short execution signal, gated by confidence.

    Confidence is ``|tanh(alpha * predicted_return)|``. When it clears
    ``threshold`` the position is taken at full size (``sign(predicted_return)``);
    otherwise the position is flat (``0``). ``threshold=0.0`` reduces to the old
    always-in-the-market ``sign()`` rule.
    """
    confidence = smooth_signal(predicted_return, alpha).abs()
    full_signal = torch.sign(predicted_return)
    return torch.where(confidence >= threshold, full_signal, torch.zeros_like(full_signal))


def shifted_previous_signal(
    signal: torch.Tensor,
    previous_signal: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Build the previous-signal series used by the transaction cost term."""

    if signal.ndim != 1:
        raise ValueError("signal must be a 1D tensor with shape [batch]")

    if previous_signal is None:
        first_signal = torch.zeros((), device=signal.device, dtype=signal.dtype)
    else:
        first_signal = torch.as_tensor(previous_signal, device=signal.device, dtype=signal.dtype).reshape(())

    if signal.numel() == 1:
        return first_signal.unsqueeze(0)

    return torch.cat([first_signal.unsqueeze(0), signal[:-1]])


def position_change_cost(
    signal,
    previous_signal,
    buy_cost_rate: float = PH_BUY_COST,
    sell_cost_rate: float = PH_SELL_COST,
):
    """Asymmetric cost of moving from ``previous_signal`` to ``signal``.

    An increase in position (``delta > 0``: opening a long or covering a short)
    is a purchase and pays ``buy_cost_rate`` per unit; a decrease (closing a long
    or opening a short) is a sale and pays ``sell_cost_rate`` per unit:

        cost = c_buy * max(delta, 0) + c_sell * max(-delta, 0)

    Written with ``abs`` only, so the same expression works on floats, numpy
    arrays and torch tensors (and stays differentiable for the loss). With
    ``buy_cost_rate == sell_cost_rate == c`` it reduces to ``c * |delta|``.
    """
    delta = signal - previous_signal
    magnitude = abs(delta)
    return buy_cost_rate * 0.5 * (magnitude + delta) + sell_cost_rate * 0.5 * (magnitude - delta)


def transaction_cost(
    signal: torch.Tensor,
    previous_signal: torch.Tensor | float | None = None,
    buy_cost_rate: float = PH_BUY_COST,
    sell_cost_rate: float = PH_SELL_COST,
) -> torch.Tensor:
    """Differentiable per-step transaction cost on a batch of signals."""

    prev_signal = shifted_previous_signal(signal, previous_signal)
    return position_change_cost(signal, prev_signal, buy_cost_rate, sell_cost_rate)


def profit_per_step(
    signal: torch.Tensor,
    actual_return: torch.Tensor,
    previous_signal: torch.Tensor | float | None = None,
    buy_cost_rate: float = PH_BUY_COST,
    sell_cost_rate: float = PH_SELL_COST,
) -> torch.Tensor:
    """Net profit per step: signal_t * return_t - cost_t."""

    cost = transaction_cost(signal, previous_signal, buy_cost_rate, sell_cost_rate)
    return signal * actual_return - cost


def profit_aware_loss(
    predicted_return: torch.Tensor,
    actual_return: torch.Tensor,
    alpha: float,
    buy_cost_rate: float = PH_BUY_COST,
    sell_cost_rate: float = PH_SELL_COST,
    previous_signal: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Loss = -mean(pos_t * return_t - cost_t) over a smooth tanh position.

    ``pos = tanh(alpha * r_hat)`` is a continuous position size in (-1, 1), so
    gradients flow for every prediction magnitude (no saturation). Transaction
    costs apply to any position change, including fractional sizing changes,
    at the buy rate for increases and the sell rate for decreases.
    """

    pos = smooth_signal(predicted_return, alpha)
    profit = profit_per_step(pos, actual_return, previous_signal, buy_cost_rate, sell_cost_rate)
    return -profit.mean()
