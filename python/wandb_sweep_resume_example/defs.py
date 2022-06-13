from os.path import abspath, dirname, join

import yaml


SOURCE_DIR = dirname(abspath(__file__))
SWEEP_CONFIG_FILE = join(SOURCE_DIR, "sweep_example.yaml")
SHOULD_FAIL_FILE = join(SOURCE_DIR, "should_fail.yaml")


def should_fail():
    with open(SHOULD_FAIL_FILE, "r") as f:
        value = yaml.safe_load(f)
    assert isinstance(value, bool)
    return value


def set_should_fail(value):
    assert isinstance(value, bool)
    with open(SHOULD_FAIL_FILE, "w") as f:
        yaml.dump(value, f)
