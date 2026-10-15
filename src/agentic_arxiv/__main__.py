"""Module launch uses the same command implementation as the console script."""

from agentic_arxiv.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
