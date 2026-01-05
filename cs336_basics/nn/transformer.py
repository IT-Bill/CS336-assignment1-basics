import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float
from einops import rearrange

from .rmsnorm import RMSNorm
from .attention import Attention
from .swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.ln1 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        self.attn = Attention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )

        self.ln2 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )

        self.token_positions = torch.arange(
            0,
            max_seq_len,
            1,
            device=device,
            dtype=torch.long,
        )
        self.token_positions = rearrange(self.token_positions, "seq_len -> 1 seq_len")
        

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
    ) -> Float[Tensor, "... seq_len d_k"]:
        identity = x

        out = self.ln1(x)
        out = self.attn(out, self.token_positions[..., :x.shape[-2]])
        out += identity
        
        # ! Update the original result
        identity = out

        out = self.ln2(out)
        out = self.ffn(out)
        out += identity

        return out
