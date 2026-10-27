"""Command-line entry point."""

import argparse
from pathlib import Path

from asanoha_export import nblist


def main():
    parser = argparse.ArgumentParser(prog="asanoha-export")
    commands = parser.add_subparsers(dest="command", required=True)
    fixtures = commands.add_parser("fixtures", help="write the test fixtures")
    fixtures.add_argument("out", type=Path, help="the repository's fixtures directory")
    args = parser.parse_args()
    nblist.write(args.out / "nblist")
