# Basic Test for wandb sweeps + pytorch-lightning

Basic:

```
./isolate.sh ./setup.sh env WANDB_MODE=dryrun ./train_wandb_pl_main.py
```

More complicated:

```sh
./isolate.sh ./setup.sh ./train_wandb_pl_sweep.py
```

Requires `python3-venv` on Ubuntu 18.04. Don't know/care about other platforms
;)

## Debug if stuck

```
./isolate.sh ./setup.sh pystuck --port={port}
```
