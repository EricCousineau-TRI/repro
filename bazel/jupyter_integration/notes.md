```sh
$ python -m venv .
$ source bin/activate
$ pip install 'jupyterlab<=3.5'
$ pip show -f jupyterlab | grep 'bin/'
  ../../../bin/jlpm
  ../../../bin/jupyter-lab
  ../../../bin/jupyter-labextension
  ../../../bin/jupyter-labhub
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
