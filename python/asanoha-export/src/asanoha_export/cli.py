"""Command-line entry point."""

import argparse
from pathlib import Path

from asanoha_export import bench, layer, model, nblist, ops, sph, tp


def main():
    parser = argparse.ArgumentParser(prog="asanoha-export")
    commands = parser.add_subparsers(dest="command", required=True)
    fixtures = commands.add_parser("fixtures", help="write the test fixtures")
    fixtures.add_argument("out", type=Path, help="the repository's fixtures directory")
    commands.add_parser("bench-nblist", help="time the matscipy and vesin neighbor lists")
    args = parser.parse_args()
    if args.command == "fixtures":
        nblist.write(args.out / "nblist")
        sph.write(args.out / "sph")
        tp.write(args.out / "tp")
        ops.write(args.out / "ops")
        layer.write(args.out / "nequix")
        model.write(args.out / "nequix")
    else:
        bench.run()
