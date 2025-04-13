# Diffusion PushT Demo

Meant to briefly reproduce results from [Diffusion Policy](https://arxiv.org/abs/2303.04137v5) paper, Table 1.

Adapted from <https://diffusion-policy.cs.columbia.edu/>

Taken from original notebook:
<https://colab.research.google.com/drive/1gxdkgRVfM55zihY9TFLja97cSVZOZq2B?usp=sharing>

Minor tweaks (some per request)
- Add `requirements.in` and `requirements.txt` for easier repro
- Increase num_epochs to 4500 to get closer to paper results
- Multiple env evals, reward threshold of 90% coverage

## Setup

To reproduce, install `uv`.

Then setup venv + install requirements:
```sh
uv venv ./venv
source ./venv/bin/activate
uv pip install -r ./requirements.txt
```

To update requirements, edit `requirements.in` then run:
```sh
uv pip compile ./requirements.in --output-file /tmp/requirements.txt --generate-hashes
grep '\--find-links' ./requirements.in > /tmp/find-links.txt || true
cat /tmp/find-links.txt /tmp/requirements.txt > ./requirements.txt
```

## Launch Notebook

```sh
source venv/bin/activate
jupyter lab ./diffusion_policy_state_pusht_demo.ipynb
```

## Results

When using checkpoint for 4500 epochs, I see 90% success rate.

## System Info

```sh
$ neofetch --off
# Some edits
OS: Ubuntu 22.04.5 LTS x86_64 
Host: 21FWSA9500 ThinkPad P1 Gen 6 
Kernel: 6.8.0-57-generic 
CPU: 13th Gen Intel i9-13900H (20) @ 5.200GHz 
GPU: NVIDIA 01:00.0 NVIDIA Corporation Device 2730 
Memory: 64019MiB
$ nvidia-smi | grep Driver
| NVIDIA-SMI 535.183.01             Driver Version: 535.183.01   CUDA Version: 12.2     |
$ nvidia-smi --list-gpus
GPU 0: NVIDIA RTX 5000 Ada Generation Laptop GPU (UUID: GPU-9327b32d-02c2-568b-255c-cf1dec136aec)
```
