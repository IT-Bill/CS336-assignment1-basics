import os
import mmap
import regex as re
from collections import defaultdict

def _remove_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    return [s for s in re.split("|".join([re.escape(t) for t in special_tokens]), text) if s]


def _pre_tokenize(text: str, special_tokens: list[str]) -> dict[tuple[int, ...], int]:
    PRE_TOKENIZER_RE = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    sub_chunks = _remove_special_tokens(text, special_tokens)

    token_count: dict[tuple[int, ...], int] = defaultdict(int)

    for sub_chunk in sub_chunks:
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        for m in PRE_TOKENIZER_RE.finditer(sub_chunk):
            token_count[tuple(m.group().encode())] += 1

    return token_count


def pre_tokenize_worker(
    path: str | os.PathLike,
    special_tokens: list[str],
    start: int,
    end: int,
):
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        return _pre_tokenize(mm[start:end].decode(), special_tokens)
