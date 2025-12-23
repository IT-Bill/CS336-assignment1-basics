import os
import json
import heapq
import functools
from tqdm import tqdm
from typing import BinaryIO
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from cs336_basics.pre_tokenize import pre_tokenize_worker


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


def _merge(token_count: dict[tuple[bytes, ...], int], num_merges: int) -> list[tuple[bytes, bytes]]:
    def _rev_bytes(b: bytes):
        return bytes(255 - x for x in b)

    merged_pairs: list[tuple[bytes, bytes]] = []

    pair_count: dict[tuple[bytes, bytes], int] = defaultdict(int)

    for bytes_tuple, count in token_count.items():
        for i in range(len(bytes_tuple) - 1):
            pair_count[(bytes_tuple[i], bytes_tuple[i + 1])] += count

    common_pair_heap: list[tuple[int, tuple[bytes, bytes], tuple[bytes, bytes]]] = [
        (-v, (_rev_bytes(k[0]), _rev_bytes(k[1])), k) for k, v in pair_count.items()
    ]
    heapq.heapify(common_pair_heap)

    # Number of merges
    for _ in tqdm(range(num_merges)):
        most_common_pair = None

        while common_pair_heap:
            # Lexiographically greater pair
            neg, _, most_common_pair = heapq.heappop(common_pair_heap)

            # Check whether it is outdated
            if pair_count.get(most_common_pair, 0) == -neg:
                break  # Current most_common_pair is valid
            # Otherwise, invalid

        if most_common_pair is None:
            break

        # (A, B)
        merged_pairs.append(most_common_pair)

        # AB
        most_common_bytes = most_common_pair[0] + most_common_pair[1]

        # Remove merged pair
        del pair_count[most_common_pair]

        for pair in list(pair_count.keys()):
            new_pair = None

            # L = pair[0], A = pair[1] = most_common_pair[0]
            # {L, A} -> {L, AB}
            if pair[1] == most_common_pair[0]:
                new_pair = (pair[0], most_common_bytes)

            # B = pair[0] = most_common_pair[1], R = pair[1]
            # {B, R} -> {AB, R}
            elif pair[0] == most_common_pair[1]:
                new_pair = (most_common_bytes, pair[1])

            if new_pair is None:
                continue

            # Update key and heap
            count = pair_count.pop(pair)
            pair_count[new_pair] = count
            heapq.heappush(common_pair_heap, (-count, (_rev_bytes(new_pair[0]), _rev_bytes(new_pair[1])), new_pair))

    return merged_pairs


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

    boundaries = None

    with open(input_path, "rb") as f:
        boundaries = _find_chunk_boundaries(f, num_processor, b"<|endoftext|>")

    token_count: dict[tuple[bytes, ...], int] = defaultdict(int)

    with ProcessPoolExecutor(num_processor) as ex:
        futures = [
            ex.submit(pre_tokenize_worker, input_path, special_tokens, start, end)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        for future in tqdm(as_completed(futures), total=num_processor):
            for k, v in future.result().items():
                token_count[k] += v

    merged_pairs = _merge(token_count, num_merges)

    vocab: dict[int, bytes] = {}
    vocab.update({i: i.to_bytes() for i in range(256)})
    vocab.update({i: token.encode() for i, token in enumerate(special_tokens, start=len(vocab))})
    vocab.update({i: b1 + b2 for i, (b1, b2) in enumerate(merged_pairs, start=len(vocab))})

    return vocab, merged_pairs


@functools.lru_cache
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def convert_tokens_to_string(tokens_bytes: bytes) -> str:
    """把 bytes 转换成 GPT-2 风格的 Unicode 字符串"""
    byte_encoder = bytes_to_unicode()
    return "".join([byte_encoder[b] for b in tokens_bytes])


if __name__ == "__main__":
    vocab, merged_pairs = train_bpe(
        input_path="./data/tiny.txt",
        vocab_size=263,
        special_tokens=["<|endoftext|>"],
    )

    # vocab, merged_pairs = train_bpe(
    #     input_path="./data/TinyStoriesV2-GPT4-train.txt",
    #     vocab_size=10000,
    #     special_tokens=["<|endoftext|>"],
    # )

    # vocab, merged_pairs = train_bpe(
    #     input_path="./data/owt_train.txt",
    #     vocab_size=32000,
    #     special_tokens=["<|endoftext|>"],
    # )

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
        token_str = convert_tokens_to_string(token_bytes)
        gpt2_style_vocab[token_str] = token_id

    # 3. 保存
    with open("log/vocab.json", "w", encoding="utf-8") as f:
        # 这里的 ensure_ascii=False 是关键，否则 Ġ 会被存成 \u0120
        json.dump(gpt2_style_vocab, f, ensure_ascii=False, indent=2)

    pretty_merges = []
    for p1, p2 in merged_pairs:
        s1 = convert_tokens_to_string(p1)
        s2 = convert_tokens_to_string(p2)
        # 通常 merges.txt 存的是 "Ġ s", "t r" 这种空格分隔的形式
        pretty_merges.append(f"{s1} {s2}")

    with open("log/merges.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(pretty_merges))
