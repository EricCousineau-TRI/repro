#!/bin/bash
set -eux

python=${1}
src_dir=$(cd $(dirname $0) && pwd)
base_dir=${2}

for init_workaround in 0 1; do
    out_dir=${base_dir}/${init_workaround}
    env _INIT_WORKAROUND=${init_workaround} \
        python3 $(which sphinx-build) \
            -b html \
            -a -E \
            -W -N -q \
            ${src_dir} \
            ${out_dir}
    echo -e "\n\n\nHTML: file://${out_dir}/index.html\n\n\n"
done
