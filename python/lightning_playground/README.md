# Example

Setup

```sh
./setup/venv-sync.sh
source ./setup/activate.sh
```

## DDP Synchronization Examples

Note that change *param* values before DDP is launched (`trainer.fit()`) is fine.

However, calling it after DDP is launched causes synchronization issue for parameters.

```sh
$ python ./ddp_normalizer_example.py
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/2
!!! rank 1: HACK! Making separate param !!!
Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/2
rank 0: on_fit_start
  param=0.0  K=1.0

rank 1: on_fit_start
  param=0.0  K=1.0

rank 1: on_train_epoch_end
  param=0.0  K=1.350000023841858
rank 0: on_train_epoch_end
  param=0.0  K=1.350000023841858

rank 1: on_fit_end
  param=0.0  K=1.350000023841858

rank 0: on_fit_end
  param=0.0  K=1.350000023841858
Final error for K: tensor([-0.6500])


$ python ./ddp_normalizer_example.py --change-after-fit-called
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/2
Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/2
rank 0: on_fit_start
  param=0.0  K=1.0

!!! rank 1: HACK! Making separate param !!!
rank 1: on_fit_start
  param=10.0  K=20.0

rank 1: on_train_epoch_end
  param=0.0  K=17.024999618530273
rank 0: on_train_epoch_end
  param=0.0  K=-1.9749999046325684

rank 1: on_fit_end
  param=0.0  K=17.024999618530273

rank 0: on_fit_end
  param=0.0  K=-1.9749999046325684
Final error for K: tensor([-3.9750])
```
