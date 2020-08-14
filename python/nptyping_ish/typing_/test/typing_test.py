from contextlib import closing
import dataclasses as dc
from io import BytesIO
import multiprocessing as mp
import pickle
from textwrap import dedent
import unittest

import numpy as np
import torch

import typing_ as mut


RigidTransform = mut.RigidTransform


@dc.dataclass
class Extra:
    x: int = 0


@dc.dataclass
class MyClass:
    a: int
    b: str
    c: mut.DoubleArray[3]
    d: mut.RgbArray
    e: Extra
    f: mut.Dict[int, str]
    g: RigidTransform


class Test(unittest.TestCase):
    def test_dimension(self):
        D_ = mut.Dimension("D")
        self.assertEqual(mut.Any, D_)
        self.assertNotEqual(hash(mut.Any), hash(D_))
        self.assertNotEqual(2, D_)
        self.assertEqual(str(D_), "D")
        self.assertNotEqual(D_, mut.Dimension("D"))
        self.assertIsNot(mut.Any, D_)

        D2_ = mut.Dimension("D2", 2)
        self.assertEqual(2, D2_)
        self.assertNotEqual(mut.Any, D2_)
        self.assertEqual(str(D2_), "D2")
        self.assertIs(D2_, D2_)
        self.assertIsNot(D2_, D_)
        self.assertIsNot(D_, mut.Any)

    def test_generic(self):
        # Check caching.
        self.assertIs(mut.List[int], mut.List[int])
        with self.assertRaises(RuntimeError):
            mut.List[int, str]
        self.assertIs(mut.Dict[int, str], mut.Dict[int, str])
        with self.assertRaises(RuntimeError):
            mut.Dict[int]

        self.assertIs(mut.Tuple[int, float, str], mut.Tuple[int, float, str])

        self.assertIs(mut.SizedList[2, int], mut.SizedList[2, int])
        D2_ = mut.Dimension("D2", 2)
        self.assertIs(mut.SizedList[D2_, int], mut.SizedList[D2_, int])
        self.assertIsNot(mut.SizedList[D2_, int], mut.SizedList[2, int])

    def test_private_compare_shape(self):
        compare_shape = mut.array._compare_shape
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
        Partial2x2 = mut.NDArray[(2, 2), :]
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
        self.assertIs(PartialDouble[(2, 2), :], Double2x2)

        # Explicitly check partial binding.
        self.assertIs(Partial2x2[np.float64], Double2x2)
        with self.assertRaises(RuntimeError):
            Partial2x2[(2, 2,), :]

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

    def test_pformat_dataclass(self):
        actual = mut.pformat_dataclass(Extra)
        expected = dedent(
            """\
            @dataclass
            class Extra:
                x: int = <default>
            """
        )
        self.assertEqual(expected, actual)

        actual = mut.pformat_dataclass(MyClass)
        expected = dedent(
            """\
            @dataclass
            class MyClass:
                a: int
                b: str
                c: NDArray[3, float64]
                d: NDArray[(H, W, C), float64]
                e: Extra
                f: Dict[int, str]
                g: RigidTransform
            """
        )
        self.assertEqual(expected, actual)

    def test_soa(self):
        cls = MyClass
        new_cls = mut.SoA[cls]
        actual = mut.pformat_dataclass(new_cls)
        expected = dedent(
            """\
            @dataclass
            class SoA[MyClass]:
                a: SizedList[N, int]
                b: SizedList[N, str]
                c: SizedList[N, NDArray[3, float64]]
                d: SizedList[N, NDArray[(H, W, C), float64]]
                e: SizedList[N, Extra]
                f: SizedList[N, Dict[int, str]]
                g: SizedList[N, RigidTransform]
            """
        )
        self.assertEqual(expected, actual)

    def test_batch_array(self):
        N_ = mut.N_

        self.assertEqual(mut.Batch[mut.NDArray], mut.Tensor[(N_, ...), :])
        Float2x2Array = mut.NDArray[(2, 2), np.float32]
        self.assertIs(
            mut.Batch[Float2x2Array], mut.Tensor[(N_, 2, 2), torch.float32]
        )

        self.assertEqual(mut.Batch[mut.Tensor], mut.Tensor[(N_, ...), :])
        Float2x2Tensor = mut.Tensor[(2, 2), torch.float32]
        self.assertIs(
            mut.Batch[Float2x2Tensor], mut.Tensor[(N_, 2, 2), torch.float32]
        )

        with self.assertRaises(RuntimeError):
            mut.Batch[mut.Batch[Float2x2Tensor]]

    def test_batch_generics(self):
        self.assertIs(
            mut.Batch[mut.Tuple[int, float]],
            mut.Tuple[mut.Batch[int], mut.Batch[float]],
        )

    def test_batch_cls(self):
        # N.B. Any image transforms should be handled in the dataloader itself!
        cls = MyClass
        new_cls = mut.Batch[cls]
        actual = mut.pformat_dataclass(new_cls)
        expected = dedent(
            """\
            @dataclass
            class Batch[MyClass]:
                a: Tensor[N, int32]
                b: SizedList[N, str]
                c: Tensor[(N, 3), float32]
                d: Tensor[(N, H, W, C), float32]
                e: Batch[Extra]
                f: Dict[int, SizedList[N, str]]
                g: SizedList[N, RigidTransform]
            """
        )
        self.assertEqual(expected, actual)

    def test_batch_pickle(self):
        """
        Tests pickling for Batch[].

        This is important when using with multiprocessing (e.g via
        torch...DataLoader).
        """
        # Ensure it can be pickled.
        obj = mut.Batch[Extra](x=torch.tensor([0]))
        pickle.dump(obj, BytesIO())

        @dc.dataclass
        class Example:
            # We create this class to exercise the instantiation, and ensure
            # that this has not been "batched" in this process.
            x: str

        def run_isolated(func):
            # Use multiprocessing to effectively fork this process and show an
            # example failure mode.
            queue = mp.Queue(maxsize=1)

            def target():
                queue.put(func())

            proc = mp.Process(target=target)
            proc.start()
            proc.join()
            return queue.get()

        # Show that we must "prime" batch instantiations when using it with
        # multiprocessing (e.g. when using `torch...DataLoader`).
        def deferred_batch_example():
            # N.B. `Example` is not yet an instantiated parameter of `Batch`.
            return mut.Batch[Example](x=["hello"])

        with self.assertRaises(AttributeError) as cm:
            run_isolated(deferred_batch_example)
        self.assertIn(
            "Can't get attribute 'Batch[Example]'", str(cm.exception)
        )

        # Now "prime" the batching.
        mut.Batch[Example]

        # Now it will work.
        value = run_isolated(deferred_batch_example)
        expected_value = mut.Batch[Example](x=["hello"])
        self.assert_equal_recursive(
            value, expected_value,
        )

    def test_batch_collate(self):
        rgb0 = np.zeros((5, 4, 3), dtype=np.float64)
        rgb1 = np.ones((5, 4, 3), dtype=np.float64)
        batch = [
            MyClass(
                a=0,
                b="b0",
                c=np.array([0.1, 0.2, 0.3]),
                d=rgb0,
                e=Extra(x=0),
                f={0: "f0"},
                g=RigidTransform([0.1, 0.2, 0.3]),
            ),
            MyClass(
                a=1,
                b="b1",
                c=np.array([1.1, 1.2, 1.3]),
                d=rgb1,
                e=Extra(x=1),
                f={0: "f1"},
                g=RigidTransform([1.1, 1.2, 1.3]),
            ),
        ]
        expected = mut.Batch[MyClass](
            a=torch.tensor([0, 1]),
            b=["b0", "b1"],
            c=torch.tensor(
                [[0.1, 0.2, 0.3], [1.1, 1.2, 1.3],], dtype=torch.float32
            ),
            d=torch.tensor([rgb0, rgb1], dtype=torch.float32),
            e=mut.Batch[Extra](x=torch.tensor([0, 1])),
            f={0: ["f0", "f1"]},
            g=[
                RigidTransform([0.1, 0.2, 0.3]),
                RigidTransform([1.1, 1.2, 1.3]),
            ],
        )
        actual = mut.batch_collate(batch)
        self.assert_equal_recursive(expected, actual)

    def assert_equal_recursive(self, a, b):
        T = type(a)
        self.assertEqual(type(b), T)
        if T == torch.Tensor:
            self.assertEqual(a.dtype, b.dtype)
            torch.testing.assert_allclose(a, b, rtol=0, atol=0)
        elif T == list:
            self.assertEqual(len(a), len(b))
            for ai, bi in zip(a, b):
                self.assert_equal_recursive(ai, bi)
        elif T == dict:
            self.assertEqual(len(a), len(b))
            self.assertEqual(a.keys(), b.keys())
            for k, ai in a.items():
                bi = b[k]
                self.assert_equal_recursive(ai, bi)
        elif dc.is_dataclass(T):
            fields = dc.fields(a)
            self.assertEqual(dc.fields(b), fields)
            for field in fields:
                ai = getattr(a, field.name)
                bi = getattr(b, field.name)
                self.assert_equal_recursive(ai, bi)
        elif T == RigidTransform:
            np.testing.assert_allclose(
                a.GetAsMatrix4(), b.GetAsMatrix4(), rtol=0, atol=0
            )
        else:
            self.assertEqual(a, b)

    def test_print_typing_info(self):
        mut.print_typing_array_info(mut)


if __name__ == "__main__":
    unittest.main()
