# Benchmark Examples

This directory contains minimal examples for:

- ASE NEB (`benchmark/NEB`)
- Sella transition-state optimization (`benchmark/TSOpt`)
- Sella IRC (`benchmark/IRC`)

All examples use the same calculator implementation:

- `ase_interface/deepmd_xtb.py` -> `DP_xTB`

Each script accepts a `--model` argument. Point it to your DeepMD delta model file (`*.pt`) before running.
