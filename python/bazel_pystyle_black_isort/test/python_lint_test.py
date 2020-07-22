import os
from subprocess import run, PIPE, STDOUT
import sys
from textwrap import dedent
import unittest

from runfiles import Rlocation


def _write_tempfile(relpath, text):
    with open(relpath, "w") as f:
        f.write(text)
    return relpath


class TestPythonLint(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        os.chdir(os.environ["TEST_TMPDIR"])

    def assert_file_equal(self, file, expected):
        with open(file, "r") as f:
            actual = f.read()
            self.assertEqual(actual, expected)

    def python_lint(self, args):
        binary = Rlocation("anzu/tools/lint/python_lint")
        return run(
            [binary] + args,
            stdout=PIPE,
            stderr=STDOUT,
            encoding="utf8",
        )

    def test_positive(self):
        good_1 = _write_tempfile(
            "good_1.py",
            dedent("""\
                import maybe_preload_pydrake_for_torch
                import torch.nn.functional as F
                """,
            ),
        )
        good_2 = _write_tempfile(
            "good_2.py",
            dedent("""\
                import maybe_preload_pydrake_for_torch
                from pyro import stuff
                """,
            ),
        )
        good_3 = _write_tempfile(
            "good_3.py",
            dedent("""\
                import nothing_to_do_with_torch
                """,
            ),
        )
        stat = self.python_lint([good_1, good_2, good_3])
        self.assertEqual(stat.stdout, "")
        self.assertEqual(stat.returncode, 0)

    def test_negative(self):
        bad_1 = _write_tempfile(
            "bad_1.py",
            dedent("""\
                import torch.nn.functional as F
                """,
            ),
        )
        bad_2 = _write_tempfile(
            "bad_2.py",
            dedent("""\
                if True:
                    from pyro import stuff
                """,
            ),
        )
        bad_3 = _write_tempfile(
            "bad_3.py",
            dedent("""\
                import maybe_preload_pydrake_for_torch
                import nothing_to_do_with_torch
                """,
            ),
        )
        stat = self.python_lint([bad_1, bad_2, bad_3])
        print(stat.stdout)
        self.assertEqual(stat.returncode, 1)
        self.assertIn("import maybe_preload_pydrake_for_torch", stat.stdout)
        self.assertIn("bad_1.py", stat.stdout)
        self.assertIn("bad_2.py", stat.stdout)
        self.assertIn("bad_3.py", stat.stdout)

        # Test fixing.
        stat = self.python_lint(["--fix", bad_1, bad_2, bad_3])
        self.assertEqual(stat.returncode, 0)
        self.assert_file_equal(
            bad_1,
            dedent("""\
                import maybe_preload_pydrake_for_torch
                import torch.nn.functional as F
                """,
            ),
        )
        self.assert_file_equal(
            bad_2,
            dedent("""\
                if True:
                    import maybe_preload_pydrake_for_torch
                    from pyro import stuff
                """,
            ),
        )
        self.assert_file_equal(
            bad_3,
            dedent("""\
                import nothing_to_do_with_torch
                """,
            ),
        )

        bad_4 = _write_tempfile(
            "bad_4.py",
            dedent("""\
                import torch
                import os
                """,
            ),
        )
        isort_settings_file = ""
        stat = self.python_lint([
            f"--isort_settings_file={isort_settings_file}",
            bad_4,
        ])
        print(stat.stdout)
        self.assertIn("maybe_preload_pydrake_for_torch", stat.stdout)
        self.assertIn("isort", stat.stdout)

        stat = self.python_lint([
            f"--isort_settings_file={isort_settings_file}",
            "--fix",
            bad_4,
        ])
        self.assertEqual(stat.returncode, 0)
        self.assert_file_equal(
            bad_4,
            dedent("""\
                import os

                import maybe_preload_pydrake_for_torch
                import torch
                """,
            ),
        )

        bad_5 = _write_tempfile(
            "bad_5.py",
            dedent("""\
                import subprocess
                import os
                print('Hello world, this will be reformatted by black + isort')
                """,
            ),
        )
        isort_settings_file = ""
        stat = self.python_lint([
            "--use_black",
            f"--isort_settings_file={isort_settings_file}",
            bad_5,
        ])
        print(stat.stdout)
        self.assertNotIn("maybe_preload_pydrake_for_torch", stat.stdout)
        self.assertIn("black", stat.stdout)
        self.assertIn("isort", stat.stdout)

        stat = self.python_lint([
            "--use_black",
            f"--isort_settings_file={isort_settings_file}",
            "--fix",
            bad_5,
        ])
        print(stat.stdout)
        self.assertEqual(stat.returncode, 0)
        self.assert_file_equal(
            bad_5,
            dedent("""\
                import os
                import subprocess

                print("Hello world, this will be reformatted by black + isort")
                """,
            ),
        )

    def test_edge_cases(self):
        # This should be fine...
        isort_settings_file = Rlocation("pyproject.toml")
        text = dedent("""\
            import time

            import numpy as np

            import bot_core
            """,
        )
        case_1 = _write_tempfile(
            "case_1.py",
            text,
        )
        stat = self.python_lint([
            f"--isort_settings_file={isort_settings_file}",
            "--fix",
            case_1,
        ])
        print(stat.stdout)
        self.assertEqual(stat.returncode, 0)
        self.assert_file_equal(case_1, text)
        print(text)

        # Test notebook.
        notebook = _write_tempfile(
            "notebook.ipynb",
            dedent(r"""
                {
                 "cells": [
                  {
                   "cell_type": "code",
                   "execution_count": null,
                   "metadata": {},
                   "outputs": [],
                   "source": [
                    "import yaml\n",
                    "import os\n",
                    "import matplotlib.pyplot as plt",
                   ]
                  },
                  {
                   "cell_type": "markdown",
                   "execution_count": null,
                   "metadata": {},
                   "outputs": [],
                   "source": [
                    "Hello world!!!"
                   ]
                  },
                  {
                   "cell_type": "code",
                   "execution_count": null,
                   "metadata": {},
                   "outputs": [],
                   "source": [
                    "with open('perception/stuff.yaml') as f:\n",
                    "    config = yaml.safe_load(f)"
                   ]
                  }
                 ],
                 "metadata": {
                  "kernelspec": {
                   "language": "python"
                  }
                 }
                }
                """),
        )

        expected = dedent(r"""
            {
              "cells": [
                {
                  "cell_type": "code",
                  "execution_count": null,
                  "metadata": {},
                  "outputs": [],
                  "source": [
                    "import os\n",
                    "\n",
                    "import matplotlib.pyplot as plt\n",
                    "import yaml\n",
                  ]
                },
                {
                  "cell_type": "markdown",
                  "execution_count": null,
                  "metadata": {},
                  "outputs": [],
                  "source": [
                    "Hello world!!!"
                  ]
                },
                {
                  "cell_type": "code",
                  "execution_count": null,
                  "metadata": {},
                  "outputs": [],
                  "source": [
                    "with open(\"perception/stuff.yaml\") as f:\n",
                    "    config = yaml.safe_load(f)\n"
                  ]
                }
              ],
              "metadata": {
                "kernelspec": {
                  "language": "python"
                }
              }
            }
            """).strip()

        stat = self.python_lint([
            "--use_black",
            f"--isort_settings_file={isort_settings_file}",
            "--fix",
            notebook,
        ])
        print(stat.stdout)
        self.assertEqual(stat.returncode, 0)
        self.assert_file_equal(notebook, expected)


if __name__ == "__main__":
    unittest.main()
