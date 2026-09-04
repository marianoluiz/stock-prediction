import torch
from torch import nn


class GRUReturnPredictor(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_scale: float | None = None,
    ) -> None:
        """
        output_scale: if set, the predicted return is ``output_scale *
            tanh(raw_logit)``, hard-bounding predictions to
            ``(-output_scale, output_scale)``. Without this, a plain
            ``Linear`` head is unbounded, and a saturating trading-signal
            loss (e.g. ``tanh(alpha * pred)``) has no incentive to keep
            ``pred`` near the actual return scale — nothing stops the raw
            logit from diverging as it chases marginal reward past the
            saturation point. Pass ``output_scale`` ~ a few std devs of the
            training return series (e.g. via ``calibrate_alpha``) to keep
            predictions on a sane scale regardless of loss shape.
        """
        super().__init__()
        self.output_scale = output_scale
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, sequence_length, features]
        if x.ndim > 3:
            if x.ndim == 4 and x.shape[1] == 1:
                x = x.squeeze(1)
            elif x.ndim == 4 and x.shape[-1] == 1:
                x = x.squeeze(-1)
            else:
                raise ValueError(f"Expected input with shape [batch, seq, features], got {tuple(x.shape)}")
        out, _ = self.gru(x)
        last_hidden = out[:, -1, :]  # [batch, hidden_size]
        raw = self.head(last_hidden).squeeze(-1)  # [batch]
        if self.output_scale is not None:
            return self.output_scale * torch.tanh(raw)
        return raw
