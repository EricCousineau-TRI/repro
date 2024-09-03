import sys
import subprocess
import multiprocessing.dummy as mp_threading
import textwrap

import nicegui
import tqdm

IGNORE = [
    # This is not actually importable?
    "cython_runtime",
]


def get_modules():
    modules = []
    for module in sys.modules.keys():
        # only public modules.
        if module.startswith("_"):
            continue
        # ignore
        if module in IGNORE:
            continue
        if module.startswith("nicegui."):
            # this is redundant, since top-level `nicegui` will ultimately
            # import everything.
            continue
        modules.append(module)
    return list(sorted(modules))


def is_module_ok(module):
    code = textwrap.dedent(f"""
        import {module}
        import multiprocessing as mp

        mp.set_start_method('spawn')
        """
    )
    cmd = [sys.executable, "-c", code]
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    modules = get_modules()
    print("Checking:")
    print(textwrap.indent("\n".join(modules), "  "))

    bad = []
    with mp_threading.Pool(processes=10) as pool:
        it = pool.imap(is_module_ok, modules)
        wrapped = zip(it, modules, strict=True)
        for result, module in tqdm.tqdm(wrapped, total=len(modules)):
            if not result:
                print(f"BAD: {module}")
                bad.append(module)

    if len(bad) > 0:
        print()
        print("Failure! Bad modules:")
        print(textwrap.indent("\n".join(bad), "  "))
        sys.exit(1)
    else:
        print("All good")


if __name__ == "__main__":
    main()
