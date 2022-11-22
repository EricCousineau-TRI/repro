from importlib import reload
import sys
import unittest


class Test(unittest.TestCase):
    def test_all(self):
        # Do negative example first so that the error code paths get executed.
        with self.assertRaises(AssertionError) as cm:
            import multiprocessing_threading_workaround_negative_example
        self.assertIn("should only be imported", str(cm.exception))

        # Import positive example.
        import multiprocessing_threading_workaround as mut
        self.assertIn("numpy", sys.modules)
        self.assertIn("cv2", sys.modules)

        # Reloading the module should trigger a failure about numpy and cv2
        # already being loaded.
        with self.assertRaises(AssertionError) as cm:
            reload(mut)
        self.assertIn("should be imported *after*", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
