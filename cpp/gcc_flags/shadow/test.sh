set -x
CXX=g++ make VERBOSE=1 -B

echo
CXX=clang++ make VERBOSE=1 -B
