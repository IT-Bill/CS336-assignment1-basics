import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from einops import rearrange

from .linear import Linear
from .rmsnorm import RMSNorm
from .attention import Attention
from .swiglu import SwiGLU
from .embedding import Embedding


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


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Given the weights of a Transformer language model and input indices,
        return the output of running a forward pass on the input indices.

        This function should use RoPE.

        Args:
            vocab_size (int): The number of unique items in the output vocabulary to be predicted.
            context_length (int): The maximum number of tokens to process at once.
            d_model (int): The dimensionality of the model embeddings and sublayer outputs.
            num_layers (int): The number of Transformer layers to use.
            num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
                evenly divisible by `num_heads`.
            d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
            rope_theta (float): The RoPE $Theta$ parameter.
        """
        super().__init__()
        
        self.embedding = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype
        )
        self.blocks = [
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ]
        self.ln_final = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype
        )
        
    
    def forward(
        self,
        x: Int[Tensor, "... seq_len"],
    ) -> Float[Tensor, "... seq_len vocab_size"]:
        
        out = self.embedding(x)
        
        for block in self.blocks:
            out = block(out)
        
        out = self.ln_final(out)
         
        out = self.lm_head(out)
        
        # out = softmax(out, dim=-1)
        
        return out 
        