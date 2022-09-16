https://gist.github.com/EricCousineau-TRI/df9777773c6a6c07c06ddef2f4f82fa3

https://gist.github.com/sloretz/074541edfe098c56ff42836118d94a8d

```sh
git clone https://github.com/sloretz/apptainer-ros

apptainer build --fakeroot repro.sif \
    ./apptainer-ros/jammy-ros-humble-desktop/Apptainer
apptainer exec \
    --fakeroot --nv --writable \
    repro.sif bash
```
