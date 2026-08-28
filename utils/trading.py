from __future__ import annotations

import torch


class StraightThroughSignal(torch.autograd.Function):
    """Binary signal in the forward pass; smooth tanh slope in the backward pass.

    The forward pass returns ``sign(pred)`` (a full long +1 / short -1 position)
    so that the loss sees exactly the signal used for validation/test metrics.
    The backward pass pretends the derivative is that of ``tanh(alpha * pred)``
    so gradients keep flowing through the otherwise flat ``sign`` curve.
    """

    @staticmethod
    def forward(ctx, predicted_return: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(predicted_return)
        ctx.alpha = alpha
        return torch.sign(predicted_return)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (predicted_return,) = ctx.saved_tensors
        slope = ctx.alpha * (1 - torch.tanh(ctx.alpha * predicted_return) ** 2)
        return grad_output * slope, None


def trading_signal(predicted_return: torch.Tensor, alpha: float) -> torch.Tensor:
    """Trading signal with straight-through gradients.

    Forward: ``sign(predicted_return)`` in {-1, +1} (identical to val/test).
    Backward: derivative of ``tanh(alpha * predicted_return)``; ``alpha`` tunes
    how smooth that backward slope is (higher = more aggressive/binary-like).
    """
    return StraightThroughSignal.apply(predicted_return, alpha)


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
    """Loss = -mean(signal_t * return_t - cost_t)."""

    signal = trading_signal(predicted_return, alpha)
    profit = profit_per_step(signal, actual_return, previous_signal, transaction_cost_rate)
    return -profit.mean()
