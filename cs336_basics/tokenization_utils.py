import functools


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


@functools.lru_cache
def unicode_to_bytes():
    return {v: k for k, v in bytes_to_unicode().items()}


def convert_bytes_to_gpt2_string(b: bytes) -> str:
    """把 bytes 转换成 GPT-2 风格的 Unicode 字符串"""
    byte_encoder = bytes_to_unicode()
    return "".join([byte_encoder[i] for i in b])


# def convert_gpt2_string_to_bytes(s: str) -> bytes:
#     unicode_encoder = unicode_to_bytes()
    
#     return bytes([unicode_encoder[i] if unicode_encoder.get(i) else i for i in s])
    

