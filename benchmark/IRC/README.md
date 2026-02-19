# Sella IRC Example

This example starts from a TS structure and runs IRC with `DP_xTB` in forward/reverse directions.

## Required input file

- `ts_optimized.xyz` (or another TS structure file via `--input`)

## Run

```bash
cd benchmark/IRC
python run_irc.py --model /path/to/your_model.pt
```

Useful options:

- `--direction both|forward|reverse`
- `--steps 1000`
- `--prefix irc`
