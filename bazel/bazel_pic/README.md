# Bazel and PIC

References:

- https://bazel.build/reference/command-line-reference#flag--force_pic
- https://bazel.build/reference/be/c-cpp#cc_library
- https://github.com/RobotLocomotion/drake/blob/v1.11.0/tools/bazel.rc#L13

Relates:

- https://github.com/RobotLocomotion/drake-ros/issues/151#issuecomment-1373986939

Example for Ubuntu 22.04, Bazel 5.3.1, gcc 11.3.0:

```sh
$ bazel --nohome_rc clean
$ bazel --nohome_rc build --noforce_pic --announce_rc -s :empty
...
(cd ... \
  /usr/bin/gcc -U_FORTIFY_SOURCE -fstack-protector -Wall -Wunused-but-set-parameter -Wno-free-nonheap-object -fno-omit-frame-pointer '-std=c++0x' -MD -MF bazel-out/k8-fastbuild/bin/_objs/empty/empty.pic.d '-frandom-seed=bazel-out/k8-fastbuild/bin/_objs/empty/empty.pic.o' -fPIC -iquote . -iquote bazel-out/k8-fastbuild/bin -fno-canonical-system-headers -Wno-builtin-macro-redefined '-D__DATE__="redacted"' '-D__TIMESTAMP__="redacted"' '-D__TIME__="redacted"' -c empty.cc -o bazel-out/k8-fastbuild/bin/_objs/empty/empty.pic.o)
...
```

Note that `-fPIC` is there, even with `--noforce_pic` and
`cc_library(..., linkstatic=True)` :shrug:
