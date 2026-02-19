#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ase.io import read, write
from sella import Sella


def load_dp_xtb():
    repo_root = Path(__file__).resolve().parents[2]
    ase_interface_dir = repo_root / "ase_interface"
    if str(ase_interface_dir) not in sys.path:
        sys.path.insert(0, str(ase_interface_dir))
    from deepmd_xtb import DP_xTB

    return DP_xTB


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sella TS optimization example with DP_xTB calculator."
    )
    parser.add_argument("--model", required=True, help="Path to DeepMD model (*.pt).")
    parser.add_argument(
        "--input",
        default="ts_guess.xyz",
        help="Initial TS guess structure.",
    )
    parser.add_argument(
        "--output",
        default="ts_optimized.xyz",
        help="Optimized TS structure.",
    )
    parser.add_argument("--fmax", type=float, default=0.01, help="Sella fmax.")
    parser.add_argument("--steps", type=int, default=500, help="Maximum steps.")
    parser.add_argument(
        "--xtb-method",
        default="GFN2-xTB",
        help="xTB method passed to DP_xTB.",
    )
    parser.add_argument(
        "--no-internal",
        action="store_true",
        help="Disable internal-coordinate mode.",
    )
    parser.add_argument("--logfile", default="ts_opt.log", help="Sella log file.")
    parser.add_argument("--traj", default="ts_opt.traj", help="Sella trajectory file.")
    return parser.parse_args()


def main():
    args = parse_args()
    DP_xTB = load_dp_xtb()

    atoms = read(args.input)
    atoms.calc = DP_xTB(model=args.model, method=args.xtb_method)

    opt = Sella(
        atoms,
        internal=not args.no_internal,
        logfile=args.logfile,
        trajectory=args.traj,
    )
    opt.run(fmax=args.fmax, steps=args.steps)
    write(args.output, atoms)


if __name__ == "__main__":
    main()
