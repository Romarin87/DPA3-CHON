#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ase.io import read, write
from sella import IRC


def load_dp_xtb():
    repo_root = Path(__file__).resolve().parents[2]
    ase_interface_dir = repo_root / "ase_interface"
    if str(ase_interface_dir) not in sys.path:
        sys.path.insert(0, str(ase_interface_dir))
    from deepmd_xtb import DP_xTB

    return DP_xTB


def parse_args():
    parser = argparse.ArgumentParser(description="Sella IRC example with DP_xTB calculator.")
    parser.add_argument("--model", required=True, help="Path to DeepMD model (*.pt).")
    parser.add_argument(
        "--input",
        default="ts_optimized.xyz",
        help="Starting TS structure for IRC.",
    )
    parser.add_argument("--steps", type=int, default=1000, help="Maximum IRC steps.")
    parser.add_argument(
        "--direction",
        choices=["both", "forward", "reverse"],
        default="both",
        help="IRC propagation direction.",
    )
    parser.add_argument(
        "--xtb-method",
        default="GFN2-xTB",
        help="xTB method passed to DP_xTB.",
    )
    parser.add_argument(
        "--prefix",
        default="irc",
        help="Prefix for output files: <prefix>_<direction>.log/.xyz",
    )
    return parser.parse_args()


def direction_list(direction):
    if direction == "both":
        return ["forward", "reverse"]
    return [direction]


def main():
    args = parse_args()
    DP_xTB = load_dp_xtb()

    for direction in direction_list(args.direction):
        atoms = read(args.input)
        atoms.calc = DP_xTB(model=args.model, method=args.xtb_method)
        irc = IRC(atoms, logfile=f"{args.prefix}_{direction}.log")
        irc.run(steps=args.steps, direction=direction)
        write(f"{args.prefix}_{direction}.xyz", atoms)


if __name__ == "__main__":
    main()
