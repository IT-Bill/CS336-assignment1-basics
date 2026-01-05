import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Bool, Int
from einops import einsum, rearrange
from .linear import Linear
from .softmax import softmax
from .rope import RotaryPositionalEmbedding


class Attention(nn.Module):
    causal_mask: Bool[Tensor, " ... seq_len seq_len"]

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.rope = (
            RotaryPositionalEmbedding(
                d_k=d_model // num_heads,
                theta=theta,
                max_seq_len=max_seq_len,
            )
            if max_seq_len and theta
            else None
        )

        self.q_weight = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_weight = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_weight = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_weight = Linear(d_model, d_model, device=device, dtype=dtype)

        self.register_buffer("causal_mask", torch.empty(0, 0), persistent=False)

    def get_causal_mask(self, seq_len: int) -> Bool[Tensor, " ... seq_len seq_len"]:
        # 1. 检查是否已经有 mask，且够不够大
        if self.causal_mask.shape[-1] >= seq_len:
            # 够大，直接切片使用
            return self.causal_mask[:seq_len, :seq_len]

        # 2. 不够大
        new_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()

        self.register_buffer("causal_mask", new_mask, persistent=False)

        return self.causal_mask

    def multi_head_self_attention(
        self,
        x: Float[Tensor, "... seq_len d_in"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, "... seq_len d_out"]:
        q_weight_x: Float[Tensor, "... seq_len d_k"] = self.q_weight(x)
        k_weight_x: Float[Tensor, "... seq_len d_k"] = self.k_weight(x)
        v_weight_x: Float[Tensor, "... seq_len d_v"] = self.v_weight(x)

        heads_q_weight_x = rearrange(
            q_weight_x,
            # ! not (d_k head)
            "... seq_len (head d_k) -> ... head seq_len d_k",
            head=self.num_heads,
        )
        heads_k_weight_x = rearrange(
            k_weight_x,
            "... seq_len (head d_k) -> ... head seq_len d_k",
            head=self.num_heads,
        )
        heads_v_weight_x = rearrange(
            v_weight_x,
            "... seq_len (head d_v) -> ... head seq_len d_v",
            head=self.num_heads,
        )

        if token_positions is not None:
            if self.rope is None:
                raise RuntimeError("Without RoPE member")

            heads_q_weight_x = self.rope(heads_q_weight_x, token_positions)
            heads_k_weight_x = self.rope(heads_k_weight_x, token_positions)

        multi_head = self.scaled_dot_product_attention(
            heads_q_weight_x,
            heads_k_weight_x,
            heads_v_weight_x,
            self.get_causal_mask(x.shape[-2]),
        )

        multi_head = rearrange(multi_head, "... head seq_len d_v -> ... seq_len (head d_v)")

        return self.o_weight(multi_head)

    @staticmethod
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
