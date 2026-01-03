import torch
from torch import Tensor
from jaxtyping import Float


def softmax(
    x: Float[Tensor, "..."],
    dim: int,
) -> Float[Tensor, "..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """

    max_val, _ = torch.max(x, dim=dim, keepdim=True)
    x_shifted = x - max_val

    exp_val = torch.exp(x_shifted)
    sum_val = torch.sum(exp_val, dim=dim, keepdim=True)

    return exp_val / sum_val
