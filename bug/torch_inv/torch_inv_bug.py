"""
Most likely related to:
https://github.com/pytorch/pytorch/issues/53047
"""
import torch


class NonDeterministic(RuntimeError):
    pass


def inverse(K):
    Kinv = K.inverse()
    if not torch.isfinite(Kinv).all():
        print(f"  Non-finite: Kinv = {Kinv}")
        raise NonDeterministic()
    return Kinv


def check_determinism(K):
    Kinv = inverse(K)
    Kinv_again = inverse(K)
    if not torch.equal(Kinv, Kinv_again):
        print(f"  Kinv - Kinv_again = {Kinv - Kinv_again}")
        raise NonDeterministic()


def do_experiment(device, dtype, N):
    num_repeats = 10000

    Ki = torch.tensor([[1.0]], device=device, dtype=dtype)
    # (N, 1, 1)
    K = torch.stack([Ki] * N)

    for i in range(10000):
        try:
            check_determinism(K)
        except NonDeterministic:
            print(f"  Non-deterministic on repeat {i}")
            return
    else:
        print("  No error")


@torch.no_grad()
def main():
    dtype = torch.float32
    cpu = torch.device("cpu")  # works
    gpu = torch.device("cuda")  # does not work

    for N in [1, 2, 3]:
        print(f"[ N = {N} ]")

        print(f"device: {cpu}")
        do_experiment(cpu, dtype, N)

        for i in range(2):
            print(f"device: {gpu} (repeat {i})")
            do_experiment(gpu, dtype, N)
        print()


if __name__ == "__main__":
    main()
