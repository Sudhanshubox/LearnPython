"""Command-line interface.

    python -m searchkit index DOCS_DIR --out INDEX.json [--ranker bm25|tfidf]
        Index every *.txt file in DOCS_DIR (sorted by name). The doc_id is the file name
        without .txt, the title is the file's first line, the text is the whole file.
        Prints "Indexed N documents." Returns 1 with "error: no .txt files in DIR" if none.

    python -m searchkit search INDEX.json "QUERY" [-k 3]
        Prints one line per result:
            "{rank}. {doc_id} ({score:.3f})  {snippet}"
        or "No results." if there are none.
"""

import argparse
import sys
from pathlib import Path

from .engine import SearchEngine


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
