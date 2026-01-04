import torch
from torch import Tensor
from jaxtyping import Float, Bool
from einops import einsum
from .softmax import softmax

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]: 
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    
    scores = einsum(Q, K, " ... queries d_k, ... keys d_k -> ... queries keys")
    
    scaled_scores = scores * torch.rsqrt(torch.tensor(Q.shape[-1]))
    
    if mask is not None:
        # True indicates valid, False indicates invalid
        scaled_scores = scaled_scores.masked_fill(mask == 0, -1e9)
    
    softmax_scores = softmax(scaled_scores, dim=-1)
    
    # keys == values
    attention = einsum(softmax_scores, V, "... queries keys, ... keys d_v -> ... queries d_v")
    
    return attention
    
    