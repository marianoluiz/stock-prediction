from __future__ import annotations

import torch


def smooth_signal(predicted_return: torch.Tensor, alpha: float) -> torch.Tensor:
    """Continuous trading position: ``tanh(alpha * predicted_return)``.

    Unlike the old straight-through ``sign`` hack, forward and backward use the
    *same* smooth function, so gradients remain alive for every ``|r|`` (no
    saturation) and there is no fake forward/backward coupling to maintain.
    The output is a position size in ``(-1, 1)``, where magnitude scales how
    confident we are and sign is the direction.
    """
    return torch.tanh(alpha * predicted_return)


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


def transaction_cost(
    signal: torch.Tensor,
    previous_signal: torch.Tensor | float | None = None,
    cost_rate: float = 1.0,
) -> torch.Tensor:
    """Differentiable transaction cost: c * |signal_t - signal_{t-1}|."""

    prev_signal = shifted_previous_signal(signal, previous_signal)
    return cost_rate * torch.abs(signal - prev_signal)


def profit_per_step(
    signal: torch.Tensor,
    actual_return: torch.Tensor,
    previous_signal: torch.Tensor | float | None = None,
    cost_rate: float = 1.0,
) -> torch.Tensor:
    """Net profit per step: signal_t * return_t - cost_t."""

    cost = transaction_cost(signal, previous_signal, cost_rate)
    return signal * actual_return - cost


def profit_aware_loss(
    predicted_return: torch.Tensor,
    actual_return: torch.Tensor,
    alpha: float,
    transaction_cost_rate: float = 0.001,
    previous_signal: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Loss = -mean(pos_t * return_t - cost_t) over a smooth tanh position.

    ``pos = tanh(alpha * r_hat)`` is a continuous position size in (-1, 1), so
    gradients flow for every prediction magnitude (no saturation). Transaction
    costs apply to any position change, including fractional sizing changes.
    """

    pos = smooth_signal(predicted_return, alpha)
    profit = profit_per_step(pos, actual_return, previous_signal, transaction_cost_rate)
    return -profit.mean()
