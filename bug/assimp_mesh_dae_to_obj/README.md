# to run

```sh
env PYTHONUNBUFFERED=1 \
    ./setup_and_run.py \
     2>&1 | sed 's#'${PWD}'#${src}#g' | tee ./output.txt
```