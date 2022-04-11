# Repro for wandb sweep + git

Should work w/ Ubuntu + Python 3. To run, needs `apt install python3-venv`.

If you're on Mac / Windows, just use whatever setup to provision a venv using
`requirements.txt`.

For exact versions, see `requirements.freeze.txt`.

```sh
cd .../wandb_sweep_example
source ./setup.sh  # Will setup venv if need be
./repro.py <ENTITY>  # Replace w/ your entity
```
