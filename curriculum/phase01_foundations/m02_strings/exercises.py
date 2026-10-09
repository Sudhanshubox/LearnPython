"""m02 exercises: strings. Replace each `raise NotImplementedError` with your solution."""


# 1. Return the initials of a name, uppercase, each followed by a dot.
#    initials("ada lovelace") -> "A.L."
#    initials("  guido   van rossum ") -> "G.V.R."
def initials(name):
    raise NotImplementedError


# 2. Is the text a palindrome, ignoring case, spaces and punctuation?
#    is_palindrome("A man, a plan, a canal: Panama") -> True
#    is_palindrome("hello") -> False
def is_palindrome(text):
    raise NotImplementedError


# 3. Count the vowels (a, e, i, o, u), in either case.
#    count_vowels("Programming in Python") -> 5
def count_vowels(text):
    raise NotImplementedError


# 4. Caesar cipher: shift each letter `shift` places along the alphabet,
#    wrapping around from z to a. Keep the case; leave non-letters unchanged.
#    caesar("Hello, World!", 3) -> "Khoor, Zruog!"
#    caesar("abc", -1) -> "zab"
#    Hint: ord() and chr(), and the % operator for wrapping.
def caesar(text, shift):
    raise NotImplementedError


# 5. Run-length encoding: compress runs of the same character.
#    rle_encode("aaabccdddd") -> "a3b1c2d4"
#    rle_encode("") -> ""
def rle_encode(text):
    raise NotImplementedError


# 6. Undo rle_encode. Counts can have more than one digit!
#    rle_decode("a3b1c2d4") -> "aaabccdddd"
#    rle_decode("x12") -> "xxxxxxxxxxxx"
def rle_decode(encoded):
    raise NotImplementedError


# 7. Return (number of characters, number of UTF-8 bytes) for the text.
#    text_size("hi") -> (2, 2)
#    text_size("नमस्ते") -> (6, 18)
def text_size(text):
    raise NotImplementedError
