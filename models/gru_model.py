import torch
from torch import nn


class GRUReturnPredictor(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
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
        pred_return = self.head(last_hidden).squeeze(-1)  # [batch]
        return pred_return
