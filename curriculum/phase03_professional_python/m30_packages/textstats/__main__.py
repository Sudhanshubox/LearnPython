"""Command line: python -m textstats FILE

Prints three lines for the file's text:
    words: <word count>
    sentences: <number of sentences>
    top: <the 3 most common words, comma-separated>
e.g.
    words: 120
    sentences: 8
    top: the, of, and

If the file doesn't exist, print "error: no such file: FILE" and return 1.
"""

import sys


def main(argv=None):
    """argv defaults to sys.argv[1:]. Returns the exit code (0 on success)."""
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
