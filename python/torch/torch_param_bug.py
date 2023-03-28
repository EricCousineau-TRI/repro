import unittest

import torch
from torch import nn


def param_in(p_check, params):
    # TODO(eric.cousineau): Why is this necessary?
    ids = [id(p) for p in params]
    return id(p_check) in ids


class Test(unittest.TestCase):
    def test_param_in(self):
        param = nn.Parameter(torch.zeros(2))
        others = [nn.Parameter(torch.zeros(3, 4))]
        with self.assertRaises(RuntimeError) as cm:
            param in others
            # others.__contains__(param)  # Alternative spelling
        # Is this a PyTorch bug?
        self.assertIn("non-singleton dimension", str(cm.exception))
        self.assertFalse(param_in(param, others))
        self.assertTrue(param_in(others[0], others))


if __name__ == "__main__":
    unittest.main()
