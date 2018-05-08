import subprocess
import example_lib_py as m

print("Top level")
subprocess.check_call("ldd " + m.__file__, shell=True)
