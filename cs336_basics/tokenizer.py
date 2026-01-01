import json
import regex

from collections.abc import Iterable, Iterator

from cs336_basics.tokenization_utils import convert_bytes_to_gpt2_string


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
        # with open(vocab_filepath) as f:
        #     vocab_json: dict[str, int] = json.load(f)
        #     vocab: dict[int, bytes] = {v: k.encode() for k, v in vocab_json.items()}

        # with open(merges_filepath) as f:
        #     merges: list[tuple[bytes, bytes]] = []
        #     for line in f.readlines():
        #         pair = line.strip().split()
        #         assert len(pair) == 2

        #         merges.append((pair[0].encode(), pair[1].encode()))

        # return Tokenizer(vocab, merges, special_tokens)
        raise NotImplementedError()

    def _encode_token(self, token: str) -> list[int]:
        if token is None or not token:
            return []
        
        token_bytes = token.encode()
        
        # Include single byte and special_token
        # ! Don't omit `is not None`
        if self.vocab_rev.get(token_bytes) is not None:
            return [self.vocab_rev[token_bytes]]
        
        pairs = [(self.vocab_rev[bytes((token_bytes[i], ))], self.vocab_rev[bytes((token_bytes[i + 1], ))]) for i in range(len(token_bytes) - 1)]
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
        for sub_chunk in sub_chunks:
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
        
