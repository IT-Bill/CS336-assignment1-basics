import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from einops import einsum, rearrange


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        d_k: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        """RoPE Module

        Args:
            d_k (int): dimension of query and key vectors
            theta (float): Θ value for the RoPE
            max_seq_len (int): Maximum sequence length that will be inputted
        """

        super().__init__()

        self.d_k = d_k
        self.seq_len = max_seq_len

        positions = torch.arange(0, max_seq_len, 1, device=device)
        k = torch.arange(0, d_k // 2, 1, device=device)
        inv_freqs = theta ** (-2 * k / d_k)
        angles = einsum(positions, inv_freqs, "seq_len, half_d_k -> seq_len half_d_k")

        sin, cos = torch.sin(angles), torch.cos(angles)

        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Int[Tensor, "... seq_len"],
    ) -> Float[Tensor, "... seq_len d_k"]:
        x_vectors = rearrange(x, "... seq_len (half_d_k two) -> ... seq_len half_d_k two", two=2)

        cos_sel = self.cos[token_positions]  # type: ignore
        sin_sel = self.sin[token_positions]  # type: ignore

        a, b = x_vectors[..., 0], x_vectors[..., 1]

        a2 = a * cos_sel - b * sin_sel
        # a2 = einsum(a, cos_sel, "... seq_len half_d_k, seq_len half_d_k -> ... seq_len half_d_k") - einsum(
        #     b, sin_sel, "... seq_len half_d_k, seq_len half_d_k -> ... seq_len half_d_k"
        # )

        b2 = a * sin_sel + b * cos_sel
        # b2 = einsum(a, sin_sel, "... seq_len half_d_k, seq_len half_d_k -> ... seq_len half_d_k") + einsum(
        #     b, cos_sel, "... seq_len half_d_k, seq_len half_d_k -> ... seq_len half_d_k"
        # )

        out_vectors = torch.stack((a2, b2), dim=-1)

        return rearrange(out_vectors, "... seq_len half_d_k two -> ... seq_len (half_d_k two)", two=2)
