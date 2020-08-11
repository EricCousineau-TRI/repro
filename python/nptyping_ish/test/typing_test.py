import unittest

import numpy as np
import torch

import typing_ as mut


class Test(unittest.TestCase):
    def test_dimension(self):
        D_ = mut.Dimension("D")
        self.assertEqual(mut.Any, D_)
        self.assertEqual(hash(mut.Any), hash(D_))
        self.assertNotEqual(1, D_)
        self.assertEqual(str(D_), "D")

        D2_ = mut.Dimension("D2", 2)
        self.assertEqual(2, D2_)
        self.assertEqual(str(D2_), "D2")

    def test_private_compare_shape(self):
        compare_shape = mut._compare_shape
        self.assertTrue(compare_shape((1, 2), (1, 2)))
        self.assertFalse(compare_shape((1, 2), (1,)))
        self.assertTrue(compare_shape((mut.Any, 2), (1, 2)))
        self.assertFalse(compare_shape((mut.Any, 2), (1, 3)))
        self.assertTrue(compare_shape((mut.Any, mut.Any), (10, 200)))

        self.assertTrue(compare_shape((...,), (5, 10, 1)))
        self.assertTrue(compare_shape((...,), ()))

        self.assertTrue(compare_shape((..., 1), (5, 10, 1)))
        self.assertFalse(compare_shape((..., 1), (2,)))
        self.assertTrue(compare_shape((1, ...), (1, 5, 10)))
        self.assertFalse(compare_shape((1, ...), (2,)))

        self.assertTrue(compare_shape((1, 2, ..., 1), (1, 2, 1)))
        self.assertTrue(compare_shape((1, 2, ..., 1), (1, 2, 2, 5, 1)))
        self.assertFalse(compare_shape((1, 2, ..., 1), (1, 2, 2, 5, 2)))

        D_ = mut.Dimension("D")
        self.assertTrue(compare_shape((D_, 2), (100, 2)))
        self.assertFalse(compare_shape((D_, 2), (100, 3)))
        D2_ = mut.Dimension("D2", 2)
        self.assertTrue(compare_shape((D_, D2_), (100, 2)))
        self.assertFalse(compare_shape((D_, D2_), (100, 3)))

    def test_ndarray(self):
        """
        Checks general behavior of GenericArray, as well as specifics for
        NDArray.
        """
        array_scalar_int = np.array(1)
        array_scalar_float = np.array(1.0)
        array_2x2_int = np.array([[1, 2], [3, 4]])
        array_2x2_float = np.array([[1.0, 2.0], [3.0, 4.0]])

        self.assertEqual(mut.NDArray.shape, (...,))
        self.assertEqual(mut.NDArray.dtype, mut.Any)
        self.assertEqual(str(mut.NDArray), "NDArray[..., Any]")

        self.assertTrue(mut.NDArray.check_array(array_scalar_int))
        self.assertTrue(mut.NDArray.check_array(array_scalar_float))
        self.assertTrue(mut.NDArray.check_array(array_2x2_int))
        self.assertTrue(mut.NDArray.check_array(array_2x2_float))
        self.assertFalse(mut.NDArray.check_array(None))
        self.assertFalse(mut.NDArray.check_array("hey"))
        self.assertFalse(mut.NDArray.check_array([1, 2, 3]))

        Double2x2 = mut.NDArray[(2, 2), float]
        self.assertEqual(str(Double2x2), "NDArray[(2, 2), float64]")
        self.assertEqual(Double2x2.shape, (2, 2))
        self.assertEqual(Double2x2.dtype, np.float64)
        self.assertIs(Double2x2, mut.NDArray[(2, 2,), np.float64])

        self.assertFalse(Double2x2.check_array(array_scalar_int))
        self.assertFalse(Double2x2.check_array(array_scalar_float))
        self.assertFalse(Double2x2.check_array(array_2x2_int))
        self.assertTrue(Double2x2.check_array(array_2x2_float))

        # Equality comparison only for GenericArray.
        self.assertEqual(mut.NDArray, Double2x2)
        self.assertEqual(Double2x2, mut.NDArray)

        # Use to_type otherwise.
        self.assertNotEqual(mut.NDArray, array_2x2_float)
        self.assertIs(mut.NDArray.to_type(array_2x2_float), Double2x2)

        # Check binding.
        Partial2x2 = mut.NDArray[
            (2, 2),
        ]
        self.assertEqual(str(Partial2x2), "NDArray[(2, 2), Any]")
        PartialDouble = mut.NDArray[np.float64]
        self.assertEqual(str(PartialDouble), "NDArray[..., float64]")

        def bound_tuple(x):
            return (x.is_shape_bound(), x.is_dtype_bound())

        self.assertIs(mut.NDArray.unbound(), mut.NDArray)
        self.assertIs(mut.NDArray.bound(), mut.NDArray[..., mut.Any])
        self.assertEqual(bound_tuple(mut.NDArray), (False, False))
        self.assertEqual(bound_tuple(Partial2x2), (True, False))
        self.assertEqual(bound_tuple(PartialDouble), (False, True))
        self.assertEqual(bound_tuple(Double2x2), (True, True))

        # Inline check of order-invariance partial binding.
        self.assertIs(Partial2x2[np.float64], Double2x2)
        self.assertIs(PartialDouble[(2, 2),], Double2x2)

        # Explicitly check partial binding.
        self.assertIs(Partial2x2[np.float64], Double2x2)
        with self.assertRaises(RuntimeError):
            Partial2x2[
                (2, 2,),
            ]

        with self.assertRaises(RuntimeError):
            PartialDouble[np.float64]

        # Show annoy facet of getitem :(
        with self.assertRaises(TypeError):
            mut.NDArray[(2, 2)]

        DoubleArray = mut.NDArray[np.float64]
        self.assertFalse(DoubleArray.check_array(array_scalar_int))
        self.assertTrue(DoubleArray.check_array(array_scalar_float))
        self.assertFalse(DoubleArray.check_array(array_2x2_int))
        self.assertTrue(DoubleArray.check_array(array_2x2_float))

        with self.assertRaises(RuntimeError):
            # Cannot re-specify dtype (even if it's the same).
            mut.LongArray[np.int64]

        #
        self.assertEqual(
            str(mut.NDArray[(10, 20), float]), "NDArray[(10, 20), float64]"
        )
        self.assertEqual(str(mut.NDArray[np.int64]), "NDArray[..., int64]")
        # Ensure that we do not generate new instantiations.
        self.assertIs(mut.NDArray[np.int64], mut.NDArray[np.int64])
        # Check (implicit) aliases.
        self.assertIs(mut.NDArray[int], mut.NDArray[np.int64])
        self.assertIs(mut.NDArray[float], mut.NDArray[np.float64])

        # - These represent the same effective type, but different
        # instantiations.
        self.assertIsNot(mut.NDArray[..., mut.Any], mut.NDArray)
        self.assertIsNot(mut.NDArray[mut.Any], mut.NDArray)
        self.assertIsNot(mut.NDArray[..., mut.Any], mut.NDArray[mut.Any])

        self.assertIs(mut.DoubleArray, mut.NDArray[np.float64])
        self.assertIs(mut.LongArray, mut.NDArray[np.int64])
        self.assertIs(mut.FloatArray, mut.NDArray[np.float32])
        self.assertIs(mut.IntArray, mut.NDArray[np.int32])

    def test_tensor(self):
        tensor_scalar_int = torch.tensor(1)
        tensor_scalar_float = torch.tensor(1.0)
        tensor_2x2_int = torch.tensor([[1, 2], [3, 4]])
        tensor_2x2_float = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        self.assertEqual(str(mut.Tensor), "Tensor[..., Any]")
        self.assertTrue(mut.Tensor.check_array(tensor_scalar_int))
        self.assertTrue(mut.Tensor.check_array(tensor_scalar_float))
        self.assertTrue(mut.Tensor.check_array(tensor_2x2_int))
        self.assertTrue(mut.Tensor.check_array(tensor_2x2_float))

        Float2x2 = mut.Tensor[(2, 2), torch.float32]
        self.assertEqual(str(Float2x2), "Tensor[(2, 2), float32]")
        self.assertFalse(Float2x2.check_array(tensor_scalar_int))
        self.assertFalse(Float2x2.check_array(tensor_scalar_float))
        self.assertFalse(Float2x2.check_array(tensor_2x2_int))
        self.assertTrue(Float2x2.check_array(tensor_2x2_float))

        self.assertIs(
            mut.Tensor.to_type(tensor_scalar_int), mut.Tensor[(), torch.int64]
        )
        self.assertIs(
            mut.Tensor.to_type(tensor_scalar_float),
            mut.Tensor[(), torch.float32],
        )
        self.assertIs(
            mut.Tensor.to_type(tensor_2x2_int), mut.Tensor[(2, 2), torch.int64]
        )
        self.assertIs(
            mut.Tensor.to_type(tensor_2x2_float),
            mut.Tensor[(2, 2), torch.float32],
        )

        # Briefly check module aliases.
        self.assertIs(mut.DoubleTensor, mut.Tensor[torch.float64])
        self.assertIs(mut.LongTensor, mut.Tensor[torch.int64])
        self.assertIs(mut.FloatTensor, mut.Tensor[torch.float32])
        self.assertIs(mut.IntTensor, mut.Tensor[torch.int32])

        # Check dtype aliases.
        self.assertIs(mut.Tensor[int], mut.Tensor[torch.int64])
        self.assertIs(mut.Tensor[float], mut.Tensor[torch.float64])

        # Show that numpy dtypes won't work.
        with self.assertRaises(TypeError):
            mut.Tensor[(2, 2), np.float32]


if __name__ == "__main__":
    unittest.main()
