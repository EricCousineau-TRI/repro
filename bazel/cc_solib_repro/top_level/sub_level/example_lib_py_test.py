import subprocess
import example_lib_py as m

assert m.func() == 10

subprocess.check_call("ldd " + m.__file__, shell=True)
