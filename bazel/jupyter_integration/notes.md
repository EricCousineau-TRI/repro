```sh
$ python -m venv .
$ source bin/activate
$ pip install 'jupyterlab<=3.5'
$ pip show -f jupyterlab | grep 'bin/'
  ../../../bin/jlpm
  ../../../bin/jupyter-lab
  ../../../bin/jupyter-labextension
  ../../../bin/jupyter-labhub
$ jupyter lab paths
Application directory:   /home/eacousineau/tmp/jupyter-venv-pure/share/jupyter/lab
User Settings directory: /home/eacousineau/tmp/jupyter-venv-pure/etc/jupyter/lab/user-settings
Workspaces directory: /home/eacousineau/tmp/jupyter-venv-pure/etc/jupyter/lab/workspaces
```

Using `rules_python` does *not* generate stubs for entry points

may come from here?
https://github.com/jupyterlab/jupyterlab/blob/0673a5926be2c374c458e90c51ff90613040517c/pyproject.toml#L60
```py
jupyterlab.labapp:main
```
which seems to resolve to calling this func
https://github.com/jupyterlab/jupyterlab/blob/0673a5926be2c374c458e90c51ff90613040517c/jupyterlab/labapp.py#L946-L949
```
main = launch_new_instance = LabApp.launch_instance
```

ah, need deps;
```
file $(bazel info output_base)/external/external/pip_deps_jupyterlab/data/share/jupyter/lab/static/index.html
```

or can also access via this
```
./bazel-bin/tools/jupyter/example.runfiles/pip_deps_jupyterlab/site-packages/jupyterlab/static
```

weird


^^^ `/static` is under `Application directory` above


but now widgets do not work...
