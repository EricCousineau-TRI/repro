#!/bin/bash
# Goal: Get rid of relative RPaths.

<<EOF
PWD = /private/var/tmp/_bazel_eacousineau/ce74db83cc8c0c51e5d78a0122afde7b/bazel-sandbox/9042642985132448468/execroot/repro
$@ = -shared -o bazel-out/osx-opt/bin/python/bindings/pymodule/global_check/_consumer_2.so -Wl,-rpath,$ORIGIN/../../../../_solib_darwin/ -Lbazel-out/osx-opt/bin/_solib_darwin bazel-out/osx-opt/bin/python/bindings/pymodule/global_check/_objs/_consumer_2.so/python/bindings/pymodule/global_check/_consumer_2.pic.o -lpython_Sbindings_Spymodule_Sglobal_Ucheck_Slibproducer_Ulinkshared bazel-out/osx-opt/bin/python/bindings/pymodule/global_check/libproducer_linkstatic.pic.a -L/usr/local/opt/python/Frameworks/Python.framework/Versions/2.7/lib/python2.7/config -lpython2.7 -ldl -framework CoreFoundation -lstdc++ -undefined dynamic_lookup -headerpad_max_install_names -no-canonical-prefixes
$< = bazel-out/osx-opt/bin/_solib_darwin/libpython_Sbindings_Spymodule_Sglobal_Ucheck_Slibproducer_Ulinkshared.so
EOF

args_in='-shared -o bazel-out/osx-opt/bin/python/bindings/pymodule/global_check/_consumer_2.so -Wl,-rpath,$ORIGIN/../../../../_solib_darwin/ -Lbazel-out/osx-opt/bin/_solib_darwin bazel-out/osx-opt/bin/python/bindings/pymodule/global_check/_objs/_consumer_2.so/python/bindings/pymodule/global_check/_consumer_2.pic.o -lpython_Sbindings_Spymodule_Sglobal_Ucheck_Slibproducer_Ulinkshared bazel-out/osx-opt/bin/python/bindings/pymodule/global_check/libproducer_linkstatic.pic.a -L/usr/local/opt/python/Frameworks/Python.framework/Versions/2.7/lib/python2.7/config -lpython2.7 -ldl -framework CoreFoundation -lstdc++ -undefined dynamic_lookup -headerpad_max_install_names -no-canonical-prefixes'
output=bazel-out/osx-opt/bin/python/bindings/pymodule/global_check/_consumer_2.so

fix_rpath() {
    python - "$@" <<EOF
import sys, os
var = '\$ORIGIN'
pre = '-Wl,-rpath,'
out_dir=os.path.dirname("${output}")
args = []
for arg in sys.argv[1:]:
    if arg.startswith(pre + var):
        arg = pre + os.path.normpath(arg.replace(pre + var, out_dir))
    args.append(arg)
print(" ".join(args))
EOF
}

echo "Args: $(fix_rpath ${args_in})"
