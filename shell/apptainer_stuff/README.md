https://gist.github.com/EricCousineau-TRI/df9777773c6a6c07c06ddef2f4f82fa3

https://gist.github.com/sloretz/074541edfe098c56ff42836118d94a8d
but to install Drake, just do binary
https://drake.mit.edu/from_binary.html#stable-releases

```sh
git clone https://github.com/sloretz/apptainer-ros

# build *.sif
apptainer build --fakeroot repro.sif \
    ./apptainer-ros/jammy-ros-humble-desktop/Apptainer
# generate writeable / persistent sandbox
apptainre build --sandbox repro.sif.sandbox repro.sif

# execute with minimal containment
# - as root
apptainer exec \
    --fakeroot --nv --writable \
    repro.sif.sandbox bash
# - as user
apptainer exec \
    --nv --writable \
    repro.sif.sandbox bash
```

### workarounds

as root

```sh
ln -s /root /home/<user>
alias sudo=""  # pty failure?!
```

as user
```sh
export PS1="(a) ${PS1}"
# sudo fails tho
```
