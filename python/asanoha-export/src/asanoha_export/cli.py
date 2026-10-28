"""Command-line entry point."""

import argparse
from pathlib import Path

from asanoha_export import bench, nblist


def main():
    parser = argparse.ArgumentParser(prog="asanoha-export")
    commands = parser.add_subparsers(dest="command", required=True)
    fixtures = commands.add_parser("fixtures", help="write the test fixtures")
    fixtures.add_argument("out", type=Path, help="the repository's fixtures directory")
    commands.add_parser("bench-nblist", help="time the matscipy and vesin neighbor lists")
    args = parser.parse_args()
    if args.command == "fixtures":
        nblist.write(args.out / "nblist")
    else:
        bench.run()
