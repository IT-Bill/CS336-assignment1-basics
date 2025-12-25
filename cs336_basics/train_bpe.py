from dataclasses import dataclass
import os
import json
import heapq
import functools
from tqdm import tqdm
from typing import BinaryIO
from collections import defaultdict
from cs336_basics import rust_lib


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


def _merge(token_count: dict[bytes, int], num_merges: int) -> list[tuple[bytes, bytes]]:
    # symbol id -> bytes
    symbol_bytes: list[bytes] = [bytes((i,)) for i in range(256)]

    merged_pairs: list[tuple[bytes, bytes]] = []

    @dataclass
    class Symbol:
        id: int
        index: int
        alive: bool
        prev: "Symbol | None" = None
        next: "Symbol | None" = None

        def __str__(self) -> str:
            current_bytes = symbol_bytes[self.id]
            prev_bytes = symbol_bytes[self.prev.id] if self.prev else None
            next_bytes = symbol_bytes[self.next.id] if self.next else None
            return f"Symbol({current_bytes}, {self.alive}, {prev_bytes}, {next_bytes})"

        __repr__ = __str__

    @dataclass
    class Token:
        symbols: list[Symbol]
        count: int

        def __str__(self) -> str:
            return f"Token({(symbol_bytes[sym.id] for sym in symbols)}, {count})"

        __repr__ = __str__

    @dataclass(slots=True, order=False)
    class RevBytes:
        data: bytes

        def __lt__(self, other: "RevBytes"):
            return self.data > other.data

    def _rev_pair(p: tuple[int, int]) -> tuple[RevBytes, RevBytes]:
        return RevBytes(symbol_bytes[p[0]]), RevBytes(symbol_bytes[p[1]])

    # (A, B) -> (token_idx, sym_idx)
    pair_occurrences: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    # (A, B) -> count
    pair_count: dict[tuple[int, int], int] = defaultdict(int)
    # (token_idx, sym_idx) -> sym
    token_list: list[Token] = []

    for token_idx, (sym_ids, count) in enumerate(token_count.items()):
        length = len(sym_ids)
        symbols: list[Symbol] = []
        for sym_idx in range(length):
            sym_id = sym_ids[sym_idx]
            symbols.append(Symbol(sym_id, sym_idx, True))

            if sym_idx + 1 < length:
                pair = (sym_id, sym_ids[sym_idx + 1])
                pair_occurrences[pair].append((token_idx, sym_idx))
                pair_count[pair] += count

        # TODO: optimize
        for sym_idx in range(length - 1):
            symbols[sym_idx].next = symbols[sym_idx + 1]
            symbols[sym_idx + 1].prev = symbols[sym_idx]

        token_list.append(Token(symbols, count))

    pair_count_heap: list[tuple[int, tuple[RevBytes, RevBytes], tuple[int, int]]] = [
        (-v, _rev_pair(k), k) for k, v in pair_count.items()
    ]
    heapq.heapify(pair_count_heap)

    for _ in tqdm(range(num_merges)):
        most_common_pair = None

        while pair_count_heap:
            # Lexiographically greater pair
            neg, _, most_common_pair = heapq.heappop(pair_count_heap)

            # Check whether it is outdated
            if pair_count.get(most_common_pair, 0) == -neg:
                break  # Current most_common_pair is valid
            # Otherwise, invalid

        if most_common_pair is None:
            break

        merged_symbol_id = len(symbol_bytes)

        # Add new pair to symbol_id_to_bytes and merged_pairs
        sym_a_id, sym_b_id = most_common_pair
        symbol_bytes.append(symbol_bytes[sym_a_id] + symbol_bytes[sym_b_id])
        merged_pairs.append((symbol_bytes[sym_a_id], symbol_bytes[sym_b_id]))

        occurrences = pair_occurrences[most_common_pair]
        occ_keep_idxs: list[int] = []

        for occ_idx, (token_idx, sym_idx) in enumerate(occurrences):
            token = token_list[token_idx]
            a = token.symbols[sym_idx]
            b = a.next
            if not a.alive or (b is None or not b.alive) or b.id != sym_b_id:
                # Not (A, B)
                continue

            occ_keep_idxs.append(occ_idx)

            # Update symbol A to symbol AB
            a.id = merged_symbol_id
            # Skip b, remember this is double-link list
            a.next = b.next
            if b.next:
                b.next.prev = a
            b.alive = False

            l, r = a.prev, b.next  # noqa: E741

            if l is not None:
                # Decrease (L, A)
                la_id = (l.id, sym_a_id)
                pair_count[la_id] -= token.count
                heapq.heappush(pair_count_heap, (-pair_count[la_id], _rev_pair(la_id), la_id))

                # Increase (L, AB)
                lab_id = (l.id, merged_symbol_id)
                pair_count[lab_id] += token.count
                pair_occurrences[lab_id].append((token_idx, l.index))
                heapq.heappush(pair_count_heap, (-pair_count[lab_id], _rev_pair(lab_id), lab_id))

            if r is not None:
                # Decrease (B, R)
                br_id = (b.id, r.id)
                pair_count[br_id] -= token.count
                heapq.heappush(pair_count_heap, (-pair_count[br_id], _rev_pair(br_id), br_id))

                # Increase (AB, R)
                abr_id = (merged_symbol_id, r.id)
                pair_count[abr_id] += token.count
                # ! Not b.index, but a.index
                # pair_occurrences[abr_id].append((token_idx, b.index))
                pair_occurrences[abr_id].append((token_idx, a.index))
                heapq.heappush(pair_count_heap, (-pair_count[abr_id], _rev_pair(abr_id), abr_id))

        pair_occurrences[most_common_pair] = [occurrences[i] for i in occ_keep_idxs]

        # Remove (A, B) from pair_count
        del pair_count[most_common_pair]

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
    num_processor = 64
    num_merges = vocab_size - 256 - len(special_tokens)

    boundaries = None

    with open(input_path, "rb") as f:
        boundaries = _find_chunk_boundaries(f, num_processor, b"<|endoftext|>")

    token_count = rust_lib.pre_tokenize(
        path=input_path,
        special_tokens=special_tokens,
        boundaries=list(zip(boundaries[:-1], boundaries[1:])),
        num_threads=num_processor,
    )

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
    # vocab, merged_pairs = train_bpe(
    #     input_path="./data/tiny.txt",
    #     vocab_size=262,
    #     special_tokens=["<|endoftext|>"],
    # )

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
