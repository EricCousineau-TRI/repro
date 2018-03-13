import subprocess
subprocess.check_call("ldd noddy.so", shell=True)

from noddy import Noddy

print(Noddy.__doc__)
