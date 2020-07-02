from textwrap import dedent
import unittest

import yaml

import wandb_examples.containers as mut


class TestContainer(unittest.TestCase):
    def test_attr_dict(self):
        """Shows "symmetric" access between dictionary __get/setitem__ and
        object __get/setattr__ provided by AttrDict."""
        dut = mut.AttrDict()
        self.assertIsInstance(dut, dict)
        dut["value"] = 1
        self.assertEqual(dut.value, 1)
        dut.new_value = 2
        self.assertEqual(dut["new_value"], 2)

        dut = mut.AttrDict.create_recursive(dict(nested=dict(value=1)))
        self.assertIsInstance(dut.nested, mut.AttrDict)
        self.assertEqual(dut.nested.value, 1)

    def test_recursive_dict_update(self):
        a = {"a": {"b": {"c": 1}}, "e": 2}
        b = {"a": {"b": {"d": 2}}, "e": 5}
        actual = mut.recursive_dict_update(a, b)
        self.assertIs(actual, a)
        expected = {"a": {"b": {"c": 1, "d": 2}}, "e": 5}
        self.assertEqual(actual, expected)

        a_bad = {"a": {"b": 2}}
        b_bad = {"a": 10}
        with self.assertRaises(AssertionError):
            mut.recursive_dict_update(a_bad, b_bad)

    def test_normalize_wandb_sweep_config(self):
        """
        Tests for:
        https://github.com/wandb/client/issues/982
        """
        raw_config = yaml.safe_load(dedent("""\
        loss:
          weight_rot: 0.1
          weight_xyz: 1.0
        loss.weight_rot: 0.5
        net:
          dropout_probability: 0.2
          num_hidden_units: 128
          num_layers: 2
        net.dropout_probability: 0.32
        net.num_hidden_units: 254
        net.num_layers: 3
        nest_top:
          nest_mid:
            nest_bottom: 10
        nest_top.nest_mid.nest_bottom: 15
        """.rstrip()))

        expected_config = yaml.safe_load(dedent("""\
        loss:
          weight_rot: 0.5
          weight_xyz: 1.0
        net:
          dropout_probability: 0.32
          num_hidden_units: 254
          num_layers: 3
        nest_top:
          nest_mid:
            nest_bottom: 15
        """.rstrip()))

        actual_config = mut.normalize_wandb_sweep_config(raw_config)
        self.maxDiff = None
        self.assertDictEqual(actual_config, expected_config)

        bad_config = {"a.b": 1, "a.b.c": 2}
        with self.assertRaises(AssertionError):
            mut.normalize_wandb_sweep_config(bad_config)


if __name__ == "__main__":
    unittest.main()
