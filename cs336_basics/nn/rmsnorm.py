import torch
from torch import nn, Tensor
from einops import reduce
from jaxtyping import Float


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.G = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )
    
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        inv_rms = torch.rsqrt(reduce(x.pow(2), "... d_model -> ... 1", "mean") + self.eps)
        rmsnorm = x * inv_rms * self.G
        
        return rmsnorm.to(in_dtype)
        