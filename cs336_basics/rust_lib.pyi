import os

def pre_tokenize(
    path: str | os.PathLike,
    special_tokens: list[str],
    boundaries: list[tuple[int, int]],
    num_threads: int,
) -> dict[bytes, int]: ...
