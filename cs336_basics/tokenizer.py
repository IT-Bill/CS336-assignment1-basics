import json
import regex

from tqdm import tqdm
from collections.abc import Iterable, Iterator

from cs336_basics.tokenization_utils import gpt2_bytes_to_unicode


class Tokenizer:
    PRE_TOKENIZE_PATTERN = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.vocab_rev = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merges_index_dict = {(self.vocab_rev[a], self.vocab_rev[b]): i for i, (a, b) in enumerate(merges)}
        self.merge_to_vocab = {i: self.vocab_rev[a + b] for i, (a, b) in enumerate(merges)}
        # 出现其中一个 token 是另一个 token 的前缀
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else []
        # 使用 () 包裹 pattern
        self.special_token_pattern = (
            regex.compile(f"({'|'.join(map(regex.escape, self.special_tokens))})") if self.special_tokens else None
        )

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_filepath) as vocab_f:
            gpt2_vocab = json.load(vocab_f)
        gpt2_bpe_merges = []
        with open(merges_filepath) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
        vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }
        # If any of the special tokens don't exist in the vocab, append them to the vocab.
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]
        return Tokenizer(vocab, merges, special_tokens)

    def _encode_token(self, token: str) -> list[int]:
        if token is None or not token:
            return []

        token_bytes = token.encode()

        # Include single byte and special_token
        # ! Don't omit `is not None`
        if self.vocab_rev.get(token_bytes) is not None:
            return [self.vocab_rev[token_bytes]]

        pairs = [
            (self.vocab_rev[bytes((token_bytes[i],))], self.vocab_rev[bytes((token_bytes[i + 1],))])
            for i in range(len(token_bytes) - 1)
        ]
        # 需要合并的 pairs 的 index
        while True:
            pairs_idx: list[int] = []
            min_merge_idx = len(self.merges)
            for i, pair in enumerate(pairs):
                if merge_idx := self.merges_index_dict.get(pair):
                    if merge_idx < min_merge_idx:
                        # 这个 pair 更优先 merge
                        min_merge_idx = merge_idx
                        # 清空原来的
                        pairs_idx = [i]

                    elif merge_idx == min_merge_idx:
                        pairs_idx.append(i)

            if pairs_idx:
                # ! The length will change during iteration
                # num_pairs = len(pairs)
                vocab_int = self.merge_to_vocab[min_merge_idx]
                for i in reversed(pairs_idx):
                    if i + 1 < len(pairs):
                        pairs[i + 1] = (vocab_int, pairs[i + 1][1])

                    if len(pairs) > 1:
                        del pairs[i]

                    if i - 1 >= 0:
                        pairs[i - 1] = (pairs[i - 1][0], vocab_int)

            else:
                # Nothing to merge
                break

        return [pairs[0][0]] + [pair[1] for pair in pairs]

    def encode(self, text: str) -> list[int]:
        if self.special_token_pattern:
            # split 后得到的是 [chunk, special_token, chunk, special_token, chunk]
            sub_chunks: list[str] = regex.split(self.special_token_pattern, text)
        else:
            sub_chunks = [text]

        result: list[int] = []
        for sub_chunk in tqdm(sub_chunks):
            if sub_chunk in self.special_tokens:
                result.extend(self._encode_token(sub_chunk))

            else:
                for m in self.PRE_TOKENIZE_PATTERN.finditer(sub_chunk):
                    result.extend(self._encode_token(m.group()))

        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[id] for id in ids]).decode(errors="replace")



if __name__ == "__main__":
    import os
    import mmap
    import random
    
    def problem_a():
        tokenizer = Tokenizer.from_files(
            "log/vocab_owt.json",
            "log/merges_owt.txt",
            ["<|endoftext|>"]
        )
        
        with open("data/owt_train.txt", "rb") as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            f.seek(0)
            
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            MB = 1024 * 10
            
            for _ in range(10):
                start = random.randint(0, file_size - MB)
                token_ids = tokenizer.encode(mm[start: start + MB].decode())
                
                print(f"Compression Ratio: {MB} / {len(token_ids) * 4} = {MB / (len(token_ids) * 4)}")
    
    problem_a()