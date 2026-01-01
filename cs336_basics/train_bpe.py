import os
import json
from typing import BinaryIO
from cs336_basics import rust_lib
from cs336_basics.tokenization_utils import convert_bytes_to_gpt2_string


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
    num_processor = 64
    num_merges = vocab_size - 256 - len(special_tokens)

    boundaries = None

    with open(input_path, "rb") as f:
        boundaries = _find_chunk_boundaries(f, num_processor, b"<|endoftext|>")

    # token_count = rust_lib.bpe.pre_tokenize(
    #     path=input_path,
    #     special_tokens=special_tokens,
    #     boundaries=list(zip(boundaries[:-1], boundaries[1:])),
    #     num_threads=num_processor,
    # )

    # merged_pairs = rust_lib.bpe.merge(token_count, num_merges)
    
    merged_pairs = rust_lib.bpe.train(
        path=input_path,
        special_tokens=special_tokens,
        boundaries=list(zip(boundaries[:-1], boundaries[1:])),
        num_merges=num_merges,
        num_threads=num_processor,
    )

    vocab: dict[int, bytes] = {}
    vocab.update({i: i.to_bytes() for i in range(256)})
    vocab.update({i: token.encode() for i, token in enumerate(special_tokens, start=len(vocab))})
    vocab.update({i: b1 + b2 for i, (b1, b2) in enumerate(merged_pairs, start=len(vocab))})

    return vocab, merged_pairs


if __name__ == "__main__":
    # vocab, merged_pairs = train_bpe(
    #     input_path="./data/tiny.txt",
    #     vocab_size=500,
    #     special_tokens=["<|endoftext|>"],
    # )

    # vocab, merged_pairs = train_bpe(
    #     input_path="./data/TinyStoriesV2-GPT4-train.txt",
    #     vocab_size=10000,
    #     special_tokens=["<|endoftext|>"],
    # )

    vocab, merged_pairs = train_bpe(
        input_path="./data/owt_train.txt",
        vocab_size=32000,
        special_tokens=["<|endoftext|>"],
    )

    # vocab, merged_pairs = train_bpe(
    #     input_path="./data/owt_valid.txt",
    #     vocab_size=1000,
    #     special_tokens=["<|endoftext|>"],
    # )

    # 1. 获取映射表
    # 2. 创建一个新的字典，格式为 { "token_string": token_id }
    #    你现在的 vocab 是 { token_id: token_bytes }
    gpt2_style_vocab = {}

    for token_id, token_bytes in vocab.items():
        # 将 bytes 转为带 Ġ 的字符串
        token_str = convert_bytes_to_gpt2_string(token_bytes)
        gpt2_style_vocab[token_str] = token_id

    # 3. 保存
    with open("log/vocab.json", "w", encoding="utf-8") as f:
        # 这里的 ensure_ascii=False 是关键，否则 Ġ 会被存成 \u0120
        json.dump(gpt2_style_vocab, f, ensure_ascii=False, indent=2)

    pretty_merges = []
    for p1, p2 in merged_pairs:
        s1 = convert_bytes_to_gpt2_string(p1)
        s2 = convert_bytes_to_gpt2_string(p2)
        # 通常 merges.txt 存的是 "Ġ s", "t r" 这种空格分隔的形式
        pretty_merges.append(f"{s1} {s2}")

    with open("log/merges.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(pretty_merges))
