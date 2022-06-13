# Simple WandB Sweep Example

Should work w/ Ubuntu + Python 3. To run, needs `apt install python3-venv`.

If you're on Mac / Windows, just use whatever setup to provision a venv using
`requirements.txt`.

For exact versions, see `requirements.freeze.txt`.

```sh
cd /path/to/wandb_sweep_example
source ./setup.sh  # Will setup venv if need be
./simple_sweep.py <ENTITY>  # Replace w/ your entity
```
