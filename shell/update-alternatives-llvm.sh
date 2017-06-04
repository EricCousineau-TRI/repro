#!/bin/bash
set -e -u

# @ref https://bugs.launchpad.net/ubuntu/+source/llvm-3.1/+bug/991493

# @note This is a hack, and not really update-alternatives.
# @ref https://askubuntu.com/questions/681046/update-alternatives-change-version-of-llvm-tools

version_default=3.9
version=${1-${version_default}}
BIN_DIR=/usr/bin
LLVM_INSTALL_DIR=../lib/llvm-${version}

files=$(echo "
    bugpoint
    llc
    llvm-ar
    llvm-as
    llvm-bcanalyzer
    llvm-config
    llvm-cov
    llvm-diff
    llvm-dis
    llvm-dwarfdump
    llvm-extract
    llvm-link
    llvm-mc
    llvm-nm
    llvm-objdump
    llvm-ranlib
    llvm-rtdyld
    llvm-size
    llvm-tblgen
    macho-dump
    obj2yaml
    opt
    verify-uselistorder
    yaml2obj
    ")

set -x
cd ${BIN_DIR}
for file in $files; do
    ln -sf ${LLVM_INSTALL_DIR}/bin/${file} ${file}
done



# Example of showing what files are linked via `llvm` package
<<COMMENT
$ ls -l \$(dpkg -L llvm | grep '^/usr/bin/') | cut -d' ' -f 10-
/usr/bin/bugpoint -> ../lib/llvm-3.8/bin/bugpoint
/usr/bin/llc -> ../lib/llvm-3.8/bin/llc
/usr/bin/llvm-ar -> ../lib/llvm-3.8/bin/llvm-ar
/usr/bin/llvm-as -> ../lib/llvm-3.8/bin/llvm-as
/usr/bin/llvm-bcanalyzer -> ../lib/llvm-3.8/bin/llvm-bcanalyzer
/usr/bin/llvm-config -> ../lib/llvm-3.8/bin/llvm-config
/usr/bin/llvm-cov -> ../lib/llvm-3.8/bin/llvm-cov
/usr/bin/llvm-diff -> ../lib/llvm-3.8/bin/llvm-diff
/usr/bin/llvm-dis -> ../lib/llvm-3.8/bin/llvm-dis
/usr/bin/llvm-dwarfdump -> ../lib/llvm-3.8/bin/llvm-dwarfdump
/usr/bin/llvm-extract -> ../lib/llvm-3.8/bin/llvm-extract
/usr/bin/llvm-link -> ../lib/llvm-3.8/bin/llvm-link
/usr/bin/llvm-mc -> ../lib/llvm-3.8/bin/llvm-mc
/usr/bin/llvm-nm -> ../lib/llvm-3.8/bin/llvm-nm
/usr/bin/llvm-objdump -> ../lib/llvm-3.8/bin/llvm-objdump
/usr/bin/llvm-ranlib -> ../lib/llvm-3.8/bin/llvm-ranlib
/usr/bin/llvm-rtdyld -> ../lib/llvm-3.8/bin/llvm-rtdyld
/usr/bin/llvm-size -> ../lib/llvm-3.8/bin/llvm-size
/usr/bin/llvm-tblgen -> ../lib/llvm-3.8/bin/llvm-tblgen
/usr/bin/macho-dump -> ../lib/llvm-3.8/bin/macho-dump
/usr/bin/obj2yaml -> ../lib/llvm-3.8/bin/obj2yaml
/usr/bin/opt -> ../lib/llvm-3.8/bin/opt
/usr/bin/verify-uselistorder -> ../lib/llvm-3.8/bin/verify-uselistorder
/usr/bin/yaml2obj -> ../lib/llvm-3.8/bin/yaml2obj
COMMENT
