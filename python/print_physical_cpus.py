"""
Merap.
"""

from collections import defaultdict, namedtuple
import os

Physical = namedtuple("Physical", ("core", "sub"))


def parse_cpuinfo(text):
    chunks = text.strip().split("\n\n")
    num_chunks = len(chunks)
    num_cpu = os.cpu_count()
    assert num_chunks == num_cpu, (num_chunks, num_cpu)

    physicals = defaultdict(list)
    for i, chunk in enumerate(chunks):
        raw = dict()
        for line in chunk.splitlines():
            pieces = line.split("\t: ")
            if len(pieces) == 2:
                key, value = pieces
                key = key.strip()
                value = value.strip()
                raw[key] = value
            else:
                assert len(pieces) == 1
                raw[key] = ""
        assert raw["processor"] == str(i)
        core_id = int(raw["core id"])
        physical_id = int(raw["physical id"])
        # TODO(eric.cousineau): Check sibling?
        key = Physical(core_id, physical_id)
        physicals[key].append(i)
    return physicals


def main():
    with open("/proc/cpuinfo", "r") as f:
        text = f.read()
    physicals = parse_cpuinfo(text)

    for key, procs in physicals.items():
        print(f"{key}: {procs}")


assert __name__ == "__main__"
main()
