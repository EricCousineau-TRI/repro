#!/bin/bash
set -eux

# Only tested on Ubuntu 18.04.

# Install prereqs.
sudo apt-GET install \
    gir1.2-gtk-3.0 \
    python3-cairo \
    python3-gi \
    python3-venv \
    python3-xlib \

# Install isolated venv.
venv_dir=~/.local/opt/key-mon
rm -rf ${venv_dir}
python3 -m venv ${venv_dir} --system-site-packages
${venv_dir}/bin/pip install -U pip wheel
# Just pegged to a version; could be updated if so desired.
${venv_dir}/bin/pip install \
    git+https://github.com/scottkirkwood/key-mon@3785370d09a1a3168e2578c04857796b6c00fb9a

# Expose isolated binary to PATH.
mkdir -p ~/.local/bin/
ln -sf ${venv_dir}/bin/key-mon ~/.local/bin/key-mon

# Most implementations of `~/.bashrc` will add `~/.local/bin` on your PATH for
# interactive sessions if the folder exists.
# You may need to restart your sessions to have it added in.

# My (Eric) user customizations
#  - Run `key-mon`
#  - Right Click on its window, go to Settings
#  - "Buttons" tab
#    - Click "Meta" checkbox (to show Windows key)
#  - "Misc" Tab
#    - Scale: 1.6
#    - Theme: big-letters
