#!/bin/bash
set -eux

if ! which extundelete; then
  sudo apt-get install extundelete
fi

cd $(dirname ${BASH_SOURCE})

sudo umount -f ./mnt || :
sudo rm -rf ./mnt ./mnt.fs
rm -rf ./output

# Make dummy filesystem.
dd if=/dev/zero of=mnt.fs bs=1024 count=10240
mkfs.ext4 ./mnt.fs
# Mount.
mkdir mnt
sudo mount ./mnt.fs ./mnt
sudo chown -R ${USER}:${USER} ./mnt

# Create files.
echo "This is a file" > ./mnt/file.txt
gzip -c ./mnt/file.txt > ./mnt/file.txt.gz
# Remove file.
rm ./mnt/file.txt
rm ./mnt/file.txt.gz

sudo umount -f ./mnt || :

# Attempt recovery.
extundelete --restore-all ./mnt.fs

ls ./RECOVERED_FILES/
