import os
import regex as re
from typing import BinaryIO
from collections import Counter


def _find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _remove_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    return [s for s in re.split("|".join([re.escape(t) for t in special_tokens]), text) if s]


def _pre_tokenize(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    PRE_TOKENIZER_RE = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
    BYTE_TABLE: tuple[bytes, ...] = tuple(bytes((i,)) for i in range(256))

    sub_chunks = _remove_special_tokens(text, special_tokens)

    token_count: Counter[tuple[bytes, ...]] = Counter()

    for sub_chunk in sub_chunks:
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        for m in PRE_TOKENIZER_RE.finditer(sub_chunk):
            token_count[tuple(BYTE_TABLE[b] for b in m.group().encode())] += 1

    return token_count


def _merge(
    token_count: Counter[tuple[bytes, ...]], num_merges: int
) -> tuple[Counter[tuple[bytes, ...]], list[tuple[bytes, bytes]]]:
    merged_pairs: list[tuple[bytes, bytes]] = []

    # Number of merges
    for _ in range(num_merges):
        nearby_count: Counter[tuple[bytes, bytes]] = Counter()

        for bytes_tuple, count in token_count.items():
            for i in range(len(bytes_tuple) - 1):
                nearby_count[(bytes_tuple[i], bytes_tuple[i + 1])] += count

        # Lexiographically greater pair
        most_common_pair, max_count = nearby_count.most_common(1)[0]
        for bytes_pair, count in nearby_count.items():
            if count < max_count:
                break
            if bytes_pair > most_common_pair:
                most_common_pair = bytes_pair
        
        merged_pairs.append(most_common_pair)
        # print(nearby_count.most_common(1)[0])

        # Update token_count
        new_token_count: Counter[tuple[bytes, ...]] = Counter()

        for bytes_tuple, count in token_count.items():
            new_bytes_list: list[bytes] = []
            num_bytes = len(bytes_tuple)
            i = 0
            while i < num_bytes:
                if i + 1 < num_bytes and (bytes_tuple[i], bytes_tuple[i + 1]) == most_common_pair:
                    new_bytes_list.append(bytes_tuple[i] + bytes_tuple[i + 1])
                    i += 2
                else:
                    new_bytes_list.append(bytes_tuple[i])
                    i += 1

            new_token_count[tuple(new_bytes_list)] = count

        token_count = new_token_count

    return token_count, merged_pairs


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    num_processor = 1
    num_merges = vocab_size - 256 - len(special_tokens)

    with open(input_path, "rb") as f:
        boundaries = _find_chunk_boundaries(f, num_processor, b"<|endoftext|>")

        # TODO: How to parallelize
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

            token_count = _pre_tokenize(chunk, special_tokens)

            # print(token_count)

            token_count, merged_pairs = _merge(token_count, num_merges)

            # print(token_count)

    vocab: dict[int, bytes] = {}
    vocab.update({i: i.to_bytes() for i in range(256)})
    vocab.update({i: token.encode() for i, token in enumerate(special_tokens, start=len(vocab))})
    vocab.update({i: b1 + b2 for i, (b1, b2) in enumerate(merged_pairs, start=len(vocab))})

    return vocab, merged_pairs

if __name__ == "__main__":
    vocab, merged_pairs = train_bpe(
        input_path="./data/TinyStoriesV2-GPT4-valid.txt",
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    
    print(vocab)
