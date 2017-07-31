set -x
CXX=g++-4.9 make VERBOSE=1 -B

echo
CXX=g++-5 make VERBOSE=1 -B

echo
CXX=clang++ make VERBOSE=1 -B
