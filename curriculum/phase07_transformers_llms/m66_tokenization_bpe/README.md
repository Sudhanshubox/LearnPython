# m66 · Tokenization: build a BPE tokenizer

**By the end you can:** explain why LLMs read *tokens* rather than characters or words, implement byte-level **Byte Pair Encoding** (the algorithm behind GPT's tokenizers) from scratch (training, encoding and decoding), add regex pre-tokenization and special tokens, and measure how well a tokenizer compresses text.

**Why it matters for AI:** the tokenizer is the first and last thing every LLM touches, and many strange LLM behaviours come from it: models that can't count the letters in "strawberry", do worse arithmetic on some numbers than others, or cost more in Hindi than in English. Understanding tokenization explains all of these, and you'll build your own GPTs on top of this module.

---

## 1. Characters, words, or something in between?

A language model predicts the next symbol from a fixed vocabulary. The choice of symbol is a trade-off:

- **Characters:** a tiny vocabulary and no unknown symbols, but sequences are long (attention cost grows with length, m68) and each symbol carries little meaning.
- **Words:** short sequences, but a huge vocabulary, and any new word, typo or name is "unknown".
- **Subwords:** frequent words become single tokens ("the"), rare words are split into pieces ("token" + "ization"). This is what every modern LLM uses.

## 2. Start from bytes

Text is Unicode; `"नमस्ते".encode("utf-8")` turns any string into **bytes** (integers 0–255), using 1 to 4 bytes per character (m02). Starting from the 256 possible bytes means the tokenizer can represent *any* text: emoji, Devanagari, code. Nothing is ever unknown. That's **byte-level** BPE, used by GPT-2 onwards.

## 3. Byte Pair Encoding

BPE (originally a 1994 compression algorithm; Sennrich et al., 2016, brought it to NLP) builds the vocabulary greedily:

```
ids = list(text.encode("utf-8"))           # start: one token per byte
repeat (vocab_size - 256) times:
    count every adjacent pair in ids
    take the most frequent pair, e.g. (116, 104) for "th"
    give it a new id (256, 257, ...) and replace every occurrence in ids
    remember the merge: (116, 104) -> 256
```

After training, the **merges** (in order) *are* the tokenizer. Each new token's bytes are the concatenation of its two parts' bytes, so the vocabulary maps id → bytes.

Replacing a pair scans left to right and doesn't overlap: merging (1, 1) in [1, 1, 1] gives [new, 1].

## 4. Encoding new text

To encode a string, start from its bytes and apply merges **in the order they were learned**: repeatedly find the adjacent pair with the *lowest* merge index (learned earliest) and merge it, until no pair in the sequence has a merge. Decoding just concatenates each token's bytes and decodes UTF-8. A single token can end in the middle of a multi-byte character, so decode the *joined* bytes, with `errors="replace"` for invalid sequences.

## 5. Pre-tokenization

Plain BPE happily learns tokens like "g." or "e th" that span word boundaries and punctuation, wasting vocabulary. GPT-2 first splits text into chunks with a regex (words with their leading space, numbers, punctuation runs, whitespace) and runs BPE **within** each chunk only. A simplified version of the pattern:

```python
PATTERN = r"'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"
re.findall(PATTERN, "Hello world, it's 2024!")
# ['Hello', ' world', ',', ' it', "'s", ' 2024', '!']
```

Training then counts pairs within each chunk (counting identical chunks once, weighted by frequency, which is much faster), and encoding encodes each chunk separately and concatenates the ids.

(GPT-4's pattern also splits numbers into groups of at most 3 digits, which makes arithmetic more consistent.)

## 6. Special tokens

Models need markers that never occur in normal text: `<|endoftext|>` between documents, or `<|user|>` / `<|assistant|>` in chat models. They get their own ids after the merges, and the encoder must recognise them as whole units *before* pre-tokenizing; otherwise "<|endoftext|>" would be split into ordinary characters, and a user could inject one by typing it.

## 7. Measuring a tokenizer

- **Compression ratio:** bytes per token. GPT-4's tokenizer averages about 4 bytes per token on English text. Higher means shorter sequences, so more text fits in the context window and costs less.
- **Fertility across languages:** a tokenizer trained mostly on English splits Hindi or Tamil into many more tokens per word, so the same sentence costs more and fits less in the context. This is an active fairness issue.

---

## Problem-solving habit #66: test the round trip

For any encoder/decoder pair (tokenizers, serializers, compressors, file formats), the most powerful test is `decode(encode(x)) == x` on many varied inputs: empty strings, emoji, other scripts, very long inputs. It catches most bugs with one line. Property-based testing libraries like `hypothesis` generate the inputs for you.

## Common mistakes

- Decoding each token separately instead of joining the bytes first (breaks multi-byte characters).
- Applying merges in frequency order instead of learned order when encoding.
- Overlapping replacements when merging (e.g. [1, 1, 1]).
- Forgetting that a leading space is part of the token: " the" and "the" are different tokens.

## Go deeper (optional, research-level)

1. Watch Andrej Karpathy's video "Let's build the GPT Tokenizer" and read OpenAI's `tiktoken` source. What does GPT-4's regex do differently from GPT-2's?
2. Train your tokenizer on English only, then measure tokens per word on English and on Hindi text. Read *Language Model Tokenizers Introduce Unfairness Between Languages* (Petrov et al., 2023).
3. Read about the "SolidGoldMagikarp" glitch tokens (2023). How can a token exist in the vocabulary but almost never in the training data, and what happens when the model sees it?

## Your turn

Open the **Exercises** tab. Start with the basic pieces, then build the full `Tokenizer` class. The tests train on this course's own lessons.
