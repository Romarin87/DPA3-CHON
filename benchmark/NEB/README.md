# ASE NEB Example

This example runs NEB with ASE and `DP_xTB` from `ase_interface/deepmd_xtb.py`.

## Required input files

- `initial.xyz`
- `final.xyz`

## Run

```bash
cd benchmark/NEB
python run_neb.py --model /path/to/your_model.pt
```

Useful options:

- `--images 7` (total images, including endpoints)
- `--interpolate idpp` or `--interpolate linear`
- `--no-climb` (disable climbing image)
