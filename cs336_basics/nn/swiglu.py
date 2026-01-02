import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float

from .linear import Linear


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """SwiGLU feed-forward network, composed of a SiLU activation function and a GLU.

        Args:
            d_model (int): Dimensionality of the feedforward input and output.
            d_ff (int): Dimensionality of the up-project happening internally.
        """
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(in_features=d_model, out_features=d_ff)
        self.w2 = Linear(in_features=d_ff, out_features=d_model)
        self.w3 = Linear(in_features=d_model, out_features=d_ff)

    def forward(
        self,
        x: Float[Tensor, " ... d_model"],
    ) -> Float[Tensor, " ... d_model"]:
        x_w1 = self.w1(x)
        silu = x_w1 * torch.sigmoid(x_w1)

        x_w3 = self.w3(x)

        ffn = self.w2(silu * x_w3)

        return ffn
