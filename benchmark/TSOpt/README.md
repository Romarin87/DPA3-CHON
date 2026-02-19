# Sella TS Optimization Example

This example follows the same Sella call style used in `sella_dpa3_delta.py` and uses `DP_xTB`.

## Required input file

- `ts_guess.xyz`

## Run

```bash
cd benchmark/TSOpt
python run_tsopt.py --model /path/to/your_model.pt
```

Useful options:

- `--fmax 0.01`
- `--steps 500`
- `--no-internal` (disable `internal=True`)
