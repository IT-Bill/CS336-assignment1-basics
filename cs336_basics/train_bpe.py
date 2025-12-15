import os
import mmap
import json
import functools
import regex as re
from tqdm import tqdm
from typing import BinaryIO
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed


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


def _pre_tokenize(text: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    PRE_TOKENIZER_RE = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    BYTE_TABLE = tuple(bytes((i,)) for i in range(256))

    sub_chunks = _remove_special_tokens(text, special_tokens)

    token_count: dict[tuple[bytes, ...], int] = defaultdict(int)

    for sub_chunk in sub_chunks:
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        for m in PRE_TOKENIZER_RE.finditer(sub_chunk):
            # token_count[tuple(BYTE_TABLE[b] for b in m.group().encode())] += 1
            token_count[tuple(map(BYTE_TABLE.__getitem__, m.group().encode()))] += 1

    return token_count


def _pre_tokenize_worker(
    path: str | os.PathLike,
    special_tokens: list[str],
    start: int,
    end: int,
):
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        return _pre_tokenize(mm[start:end].decode(), special_tokens)


def _merge(
    token_count: dict[tuple[bytes, ...], int], num_merges: int
) -> tuple[dict[tuple[bytes, ...], int], list[tuple[bytes, bytes]]]:
    merged_pairs: list[tuple[bytes, bytes]] = []

    # Number of merges
    for _ in tqdm(range(num_merges)):
        nearby_count: dict[tuple[bytes, bytes], int] = defaultdict(int)

        for bytes_tuple, count in token_count.items():
            for i in range(len(bytes_tuple) - 1):
                nearby_count[(bytes_tuple[i], bytes_tuple[i + 1])] += count

        # Lexiographically greater pair
        most_common_pair, _ = max(nearby_count.items(), key=lambda c: (c[1], c[0]))

        merged_pairs.append(most_common_pair)

        # Update token_count
        new_token_count: dict[tuple[bytes, ...], int] = defaultdict(int)

        for bytes_tuple, count in token_count.items():
            # Perf: Skip if each bytes of most_common_pair not in bytes_tuple
            if most_common_pair[0] not in bytes_tuple or most_common_pair[1] not in bytes_tuple:
                new_token_count[bytes_tuple] = count
                continue

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
    num_processor = 32
    num_merges = vocab_size - 256 - len(special_tokens)

    boundaries = None

    with open(input_path, "rb") as f:
        boundaries = _find_chunk_boundaries(f, num_processor, b"<|endoftext|>")

    token_count: dict[tuple[bytes, ...], int] = defaultdict(int)

    with ProcessPoolExecutor(num_processor) as ex:
        futures = [
            ex.submit(_pre_tokenize_worker, input_path, special_tokens, start, end)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        for future in tqdm(as_completed(futures), total=num_processor):
            for k, v in future.result().items():
                token_count[k] += v

    token_count, merged_pairs = _merge(token_count, num_merges)

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
        input_path="./data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )

    # vocab, merged_pairs = train_bpe(
    #     input_path="./data/owt_train.txt",
    #     vocab_size=32000,
    #     special_tokens=["<|endoftext|>"],
    # )

    # vocab, merged_pairs = train_bpe(
    #     input_path="./data/TinyStoriesV2-GPT4-valid.txt",
    #     vocab_size=500,
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
