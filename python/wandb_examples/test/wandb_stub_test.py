from contextlib import contextmanager
import os
import unittest

import wandb
# See: https://github.com/wandb/client/issues/1101
import wandb.tensorboard


@contextmanager
def mock_attr(obj, attr, replace):
    """Temporarily replaces an attribute."""
    original = getattr(obj, attr)
    setattr(obj, attr, replace)
    yield
    setattr(obj, attr, original)


@contextmanager
def restore_dict(container):
    """Copies original values of container, then restores them afterwards."""
    original = dict(container)
    yield
    container.clear()
    container.update(original)


def fake_init_headless(run, cloud=True):
    """
    Used to monkeypatch for wandb.init() so it does not try to open pty
    streams under `bazel test`.
    """
    assert not cloud
    run.set_environment(os.environ)


class TestWandb(unittest.TestCase):
    def test_wandb(self):
        with restore_dict(os.environ):
            # https://docs.wandb.com/library/advanced/environment-variables
            os.environ["WANDB_CONFIG_DIR"] = os.environ["TEST_TMPDIR"]
            os.environ["WANDB_DIR"] = os.environ["TEST_TMPDIR"]
            os.environ["WANDB_MODE"] = "dryrun"
        #     with mock_attr(wandb, "_init_headless", fake_init_headless):
            wandb.init(project="test_project", sync_tensorboard=True)


if __name__ == "__main__":
    unittest.main()
