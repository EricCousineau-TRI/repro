# Basic Test for wandb + pytorch-lightning

```
./isolate.sh ./setup.sh ./train_wandb_pl_main.py
```

For Ubuntu 18.04, requires `python3-venv` on Ubuntu 18.04.

## Currently

Above train script is stuck... In both CPython 3.6.9 and 3.8.0.

Seemed like it worked before (commit 583acd21), but meh, can't easily pinpoint
what changed.
