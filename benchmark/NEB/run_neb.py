#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ase.io import read, write
from ase.mep import NEB
from ase.optimize import FIRE


def load_dp_xtb():
    repo_root = Path(__file__).resolve().parents[2]
    ase_interface_dir = repo_root / "ase_interface"
    if str(ase_interface_dir) not in sys.path:
        sys.path.insert(0, str(ase_interface_dir))
    from deepmd_xtb import DP_xTB

    return DP_xTB


def parse_args():
    parser = argparse.ArgumentParser(
        description="ASE NEB example with DP_xTB calculator."
    )
    parser.add_argument("--model", required=True, help="Path to DeepMD model (*.pt).")
    parser.add_argument("--initial", default="initial.xyz", help="Initial structure.")
    parser.add_argument("--final", default="final.xyz", help="Final structure.")
    parser.add_argument(
        "--images",
        type=int,
        default=7,
        help="Total number of images, including endpoints (>=2).",
    )
    parser.add_argument(
        "--interpolate",
        choices=["linear", "idpp"],
        default="idpp",
        help="Interpolation method.",
    )
    parser.add_argument("--k", type=float, default=0.1, help="Spring constant for NEB.")
    parser.add_argument(
        "--no-climb",
        action="store_true",
        help="Disable climbing-image NEB.",
    )
    parser.add_argument("--fmax", type=float, default=0.05, help="Optimizer fmax.")
    parser.add_argument("--steps", type=int, default=300, help="Maximum steps.")
    parser.add_argument(
        "--xtb-method",
        default="GFN2-xTB",
        help="xTB method passed to DP_xTB.",
    )
    parser.add_argument("--logfile", default="neb.log", help="Optimizer log file.")
    parser.add_argument("--traj", default="neb.traj", help="Optimizer trajectory.")
    parser.add_argument(
        "--output",
        default="neb_images.xyz",
        help="Output XYZ containing all optimized images.",
    )
    return parser.parse_args()


def validate_endpoints(initial, final):
    if len(initial) != len(final):
        raise ValueError("Initial/final structures have different atom counts.")
    if initial.get_chemical_symbols() != final.get_chemical_symbols():
        raise ValueError("Initial/final structures have different element ordering.")


def main():
    args = parse_args()
    if args.images < 2:
        raise ValueError("--images must be >= 2.")

    DP_xTB = load_dp_xtb()
    initial = read(args.initial)
    final = read(args.final)
    validate_endpoints(initial, final)

    images = [initial]
    images.extend(initial.copy() for _ in range(args.images - 2))
    images.append(final)

    neb = NEB(images, k=args.k, climb=not args.no_climb)
    if args.interpolate == "idpp":
        neb.interpolate(method="idpp")
    else:
        neb.interpolate()

    for image in images:
        image.calc = DP_xTB(model=args.model, method=args.xtb_method)

    opt = FIRE(neb, logfile=args.logfile, trajectory=args.traj)
    opt.run(fmax=args.fmax, steps=args.steps)
    write(args.output, images)


if __name__ == "__main__":
    main()
