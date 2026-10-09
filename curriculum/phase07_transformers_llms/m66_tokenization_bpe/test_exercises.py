import re
from pathlib import Path

import pytest

from exercises import (
    PATTERN,
    Tokenizer,
    build_vocab,
    compression_ratio,
    decode,
    encode,
    get_pair_counts,
    merge,
    pretokenize,
    train_bpe,
)

CURRICULUM = next(p for p in Path(__file__).resolve().parents if (p / "phase01_foundations").is_dir())
SAMPLES = ["", "hello world", "naïve café", "नमस्ते दुनिया", "emoji 🎉🐍 test", "  spaces\n\ttabs\n\n",
           "def f(x):\n    return x ** 2  # square", "it's 2024, isn't it?", "snake_case __init__ a_1"]


def corpus(pattern):
    return "".join(p.read_text(encoding="utf-8") for p in sorted(CURRICULUM.glob(pattern)))


@pytest.fixture(scope="module")
def tok():
    return Tokenizer().train(corpus("phase01*/*/README.md"), 400)


def test_get_pair_counts():
    counts = get_pair_counts([1, 2, 3, 1, 2])
    assert dict(counts) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    assert list(counts) == [(1, 2), (2, 3), (3, 1)]
    total = get_pair_counts([5, 6], counts, weight=3)
    assert total is counts and counts[(5, 6)] == 3 and counts[(1, 2)] == 2
    assert dict(get_pair_counts([7])) == {}


def test_merge():
    assert merge([1, 2, 3, 1, 2], (1, 2), 9) == [9, 3, 9]
    assert merge([1, 1, 1], (1, 1), 9) == [9, 1]
    assert merge([1, 1, 1, 1], (1, 1), 9) == [9, 9]
    assert merge([], (1, 2), 9) == []
    ids = [1, 2]
    merge(ids, (1, 2), 9)
    assert ids == [1, 2], "return a new list"


def test_train_bpe_small():
    merges = train_bpe("aaabdaaabac", 259)
    assert list(merges.items()) == [((97, 97), 256), ((256, 97), 257), ((257, 98), 258)]
    assert train_bpe("ab", 1000) == {(97, 98): 256}, "stop when no pairs are left"


def test_build_vocab():
    vocab = build_vocab({(97, 97): 256, (256, 97): 257})
    assert len(vocab) == 258
    assert vocab[65] == b"A" and vocab[256] == b"aa" and vocab[257] == b"aaa"


def test_encode_decode():
    merges = train_bpe("aaabdaaabac", 259)
    vocab = build_vocab(merges)
    assert encode("aaabdaaabac", merges) == [258, 100, 258, 97, 99]
    assert encode("xyz", merges) == [120, 121, 122]
    assert encode("aaaa", merges) == [256, 256]   # (256, 256) was never learned
    big = train_bpe(corpus("phase01*/m01*/README.md"), 320)
    big_vocab = build_vocab(big)
    for s in SAMPLES:
        assert decode(encode(s, big), big_vocab) == s
    assert decode([0xE0, 0xA4], build_vocab({})) == "�", "use errors='replace'"


def test_encode_uses_merge_order():
    merges = {(98, 99): 256, (97, 98): 257}       # "bc" was learned before "ab"
    assert encode("abc", merges) == [97, 256]


def test_pretokenize():
    assert pretokenize("Hello world, it's 2024!") == ["Hello", " world", ",", " it", "'s", " 2024", "!"]
    for s in SAMPLES:
        assert "".join(pretokenize(s)) == s


@pytest.mark.timeout(60)
def test_tokenizer_train(tok):
    assert len(tok.merges) == 144 and len(tok.vocab) == 400
    assert list(tok.merges.values()) == list(range(256, 400))
    for i in range(256, 400):
        piece = tok.vocab[i].decode("utf-8", errors="ignore")
        if piece:
            assert len(re.findall(PATTERN, piece)) == 1, f"token {piece!r} crosses a chunk boundary"
    assert b" the" in tok.vocab.values()


def test_tokenizer_roundtrip(tok):
    for s in SAMPLES + [corpus("phase02*/m12*/README.md")]:
        assert tok.decode(tok.encode(s)) == s
    assert len(tok.encode(" the the the")) == 3


def test_special_tokens(tok):
    t = Tokenizer().train("hello world " * 20, 270)
    t.register_special_tokens(["<|endoftext|>", "<|user|>"])
    n = len(t.vocab)
    assert t.special_tokens == {"<|endoftext|>": n, "<|user|>": n + 1}
    ids = t.encode("hello<|endoftext|><|user|> world")
    assert ids == t.encode("hello") + [n, n + 1] + t.encode(" world")
    assert t.decode(ids) == "hello<|endoftext|><|user|> world"
    assert t.encode("<|endoftext|>") == [n]


def test_save_load(tmp_path, tok):
    tok.register_special_tokens(["<|endoftext|>"])
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = Tokenizer.load(path)
    assert loaded.merges == tok.merges and loaded.special_tokens == tok.special_tokens
    s = "Some new text<|endoftext|>with a special token."
    assert loaded.encode(s) == tok.encode(s)


@pytest.mark.timeout(60)
def test_compression(tok):
    held_out = corpus("phase04*/m36*/README.md")
    small = Tokenizer().train(corpus("phase01*/*/README.md"), 300)
    r_small, r_big = compression_ratio(small, held_out), compression_ratio(tok, held_out)
    assert 1.0 < r_small < r_big
    assert compression_ratio(Tokenizer(), held_out) == pytest.approx(1.0)
    hindi = "मशीन लर्निंग कृत्रिम बुद्धिमत्ता का एक हिस्सा है। "
    english = "Machine learning is a part of artificial intelligence. "
    assert len(tok.encode(hindi)) > 3 * len(tok.encode(english)), \
        "the same sentence costs far more tokens in a language the tokenizer wasn't trained on"
