"""
CLI entry point for the trade-analyzer package.

Delegates to the CLI runner in patches/add_cli_runner.py so that the
`trade-analyzer` console script works after `pip install`.
"""

import sys
import os

# Ensure the project root is on sys.path so patches/ and breakaway_gap_scan
# can be resolved regardless of where the command is invoked from.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from patches.add_cli_runner import main_cli


def main():
    """Entry point for the ``trade-analyzer`` console script."""
    main_cli()


if __name__ == "__main__":
    main()
