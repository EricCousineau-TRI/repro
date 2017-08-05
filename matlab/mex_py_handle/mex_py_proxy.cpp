#include <dlfcn.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

using std::cout;
using std::endl;
using std::string;

// For GIL
// <pybind11>
#include <Python.h>

namespace pybind11 {

class gil_scoped_acquire {
    PyGILState_STATE state;
public:
    gil_scoped_acquire() {
      cout << "c: gil_acquire start" << endl;
      state = PyGILState_Ensure();
    }
    ~gil_scoped_acquire() {
      PyGILState_Release(state);
      cout << "c: gil_acquire end" << endl;
    }
};

class gil_scoped_release {
    PyThreadState *state;
public:
    gil_scoped_release() { state = PyEval_SaveThread(); }
    ~gil_scoped_release() { PyEval_RestoreThread(state); }
};

}  // namespace pybind11

namespace py = pybind11;

// </pybind11>

// https://stackoverflow.com/questions/15011674/is-it-possible-to-dereference-variable-ids
// Can wrap around pythonic types?

#include <mex.h>
#include <matrix.h>

typedef uint64_T mx_raw_t;
typedef uint64_T py_raw_t;

mxArray* mxCreateUint64Value(mx_raw_t value) {
  mxArray* mx_out = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
  mx_raw_t* pvalue = static_cast<mx_raw_t*>(mxGetData(mx_out));
  *pvalue = value;
  return mx_out;
}

mx_raw_t mxGetUint64(const mxArray* in) {
  return *static_cast<uint64_T*>(mxGetData(in));
}

// static_cast will not work, as MATLAB shifts memory around between calls.
mx_raw_t mx_to_mx_raw(mxArray* mx) {
  cout << "keep going" << endl;
  mxArray* argin[1] = {mx};
  mxArray* argout[1] = {nullptr};
  mexCallMATLAB(1, argout, 1, argin, "MexPyProxy.mx_to_mx_raw");
  mx_raw_t mx_raw = mxGetUint64(argout[0]);
  cout << "mer: " << mx_raw << endl;
  // mxFree(argout[0]);
  return mx_raw;
}

mxArray* mx_raw_to_mx(mx_raw_t mx_raw) {
  mxArray* argin[1] = {mxCreateUint64Value(mx_raw)};
  mxArray* argout[1] = {nullptr};
  mexCallMATLAB(1, argout, 1, argin, "MexPyProxy.mx_raw_to_mx");
  mxArray* mx = argout[0];
  // mxFree(argin[0]);
  return mx;
}

// Use internal linkage to prevent collision when dynamically linking.
namespace {

void* c_void_p_pass_thru(void* in) {
  // Leverage Python's ctypes marhsalling of py_object is a (hopefully)
  // robust mechanism to pass stuff around.
  return in;
}

void* c_mx_feval_py_raw(mx_raw_t mx_raw_handle, int nout, py_raw_t py_raw_in) {
  cout << "c: c_mx_feval_py_raw" << endl;

  py::gil_scoped_acquire py_gil;

  mxArray* mx_mx_raw_handle = mxCreateUint64Value(mx_raw_handle);
  // mxArray* mx_nout = mxCreateUint64Value(nout);
  // mxArray* mx_py_raw_in = mxCreateUint64Value(py_raw_in);

  // const int nrhs = 3;
  // mxArray* mx_in[nrhs] = {mx_mx_raw_handle, mx_nout, mx_py_raw_in};
  // const int nlhs = 1;
  // mxArray* mx_out[nlhs] = {NULL};
  mxArray* mx_in[] = {mx_mx_raw_handle};
  cout << "c: call matlab - start" << endl;
  // mexCallMATLAB(nlhs, mx_out, nrhs, mx_in, "MexPyProxy.mx_feval_py_raw");
  mexCallMATLAB(0, nullptr, 1, mx_in, "simple");
  // py_raw_t py_raw_out = reinterpret_cast<py_raw_t>(mxGetUint64(mx_out[0]));

  cout << "c: call matlab - finish" << endl;

  // mxFree(mx_mx_raw_handle);

  // mxFree(mx_nout);
  // mxFree(mx_py_raw_in);
  // mxFree(mx_out);

  return 0;
  // return py_raw_out;
}

int c_simple() {
  py::gil_scoped_acquire py_gil;
  cout << "c: call matlab - start" << endl;
  mxArray* mx_in[] = {mxCreateUint64Value(1), mxCreateUint64Value(2)};
  mxArray* mx_out[] = {nullptr};
  mexCallMATLAB(1, mx_out, 2, mx_in, "simple");
  cout << "  y = " << mxGetUint64(mx_out[0]) << endl;
  cout << "c: call matlab - finish" << endl;
  // mxFree(mx_in[0]);
  return 0;
}

}  // namespace

string mxToStdString(mxArray* mx_in) {
  // From: mxmalloc.c
  const int len = mxGetN(mx_in) + 1;
  char* buffer = mxMalloc(len);
  mxGetString(mx_in, buffer, (mwSize)len);
  string out = buffer;
  mxFree(buffer);
  return out;
}

#define ex_assert(cond, msg_expr) \
  if (!(cond)) { \
    std::ostringstream os; \
    os << #cond << endl; \
    os << msg_expr; \
    throw std::runtime_error(os.str()); \
  }

const string usage =
    "[varargout{:}] = mex_py_proxy(op, varargin)\n"
    "  op\n" \
    "    'mx_to_mx_raw' Convert mxArray* to uint64 to be passed opaquely.\n" \
    "        [mx_raw] = mex_py_proxy('mx_to_mx_raw', mx)\n" \
    "    'mx_raw_to_mx' Unpack opaque value.\n" \
    "        [mx] = mex_py_proxy('mx_raw_to_mx', mx_raw)\n" \
    "    'get_c_func_ptrs' Get C pointers, using opaque types, to pass to Python.\n" \
    "        [c_func_ptrs_struct] = mex_py_proxy('get_c_func_ptrs')\n" \
    "    'simple' Call Python from MATLAB from C\n" \
    "        [] = mex_py_proxy('simple')\n" \
    "    'help' Show usage.";

// Create MATLAB struct containing raw values pointing to functions.
mxArray* get_c_func_ptrs() {
  // Reference example: mxcreatestructarray.c
  const int n = 4;
  const char* names[n] = {
    "c_py_to_py_raw",
    "c_py_raw_to_py",
    "c_mx_feval_py_raw",
    "c_simple",
  };
  // TODO: Will this work???
  void* ptrs[n] = {
    &c_void_p_pass_thru,
    &c_void_p_pass_thru,
    &c_mx_feval_py_raw,
    &c_simple,
  };
  mwSize dims[2] = {1, 1};
  mxArray* s = mxCreateStructArray(2, dims, n, names);
  for (int i = 0; i < n; ++i) {
    const char* name = names[i];
    void* ptr = ptrs[i];
    int field_index = mxGetFieldNumber(s, name);
    mx_raw_t ptr_raw = reinterpret_cast<mx_raw_t>(ptr);
    mxArray* value = mxCreateUint64Value(ptr_raw);
    mxSetFieldByNumber(s, 0, field_index, value);
  }
  return s;
}

// Wrap MEX function call.
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  try {
    ex_assert(nrhs >= 1, usage);

    string op = mxToStdString(prhs[0]);
    if (op == "mx_to_mx_raw") {
      cout << "start" << endl;
      // Return a uint64 representing point to mxArray* item.
      ex_assert(nrhs == 2, usage);
      ex_assert(nlhs == 1, usage);
      mxArray* mx = prhs[1];
      mx_raw_t mx_raw = mx_to_mx_raw(mx);
      plhs[0] = mxCreateUint64Value(mx_raw);
    } else if (op == "mx_raw_to_mx") {
      // Opposite direction.
      ex_assert(nrhs == 3, usage);
      ex_assert(nlhs == 1, usage);
      mx_raw_t mx_raw = mxGetUint64(prhs[1]);
      mxArray* mx = mx_raw_to_mx(mx_raw);
      plhs[0] = mx;
    } else if (op == "get_c_func_ptrs") {
      ex_assert(nrhs == 1, usage);
      ex_assert(nlhs == 1, usage);
      // Using example: 
      plhs[0] = get_c_func_ptrs();
    } else if (op == "help") {
      mexPrintf("%s\n", usage.c_str());
    } else if (op == "simple") {
      ex_assert(nrhs == 1, "");
      ex_assert(nlhs == 0, "");
      c_simple();
    } else if (op == "py_so_reload") {
      dlopen("/usr/lib/x86_64-linux-gnu/libpython2.7.so.1", RTLD_NOLOAD | RTLD_GLOBAL);
      cout << "Reload python.so with global symbol table" << endl;
    }
  }
  catch (const std::exception& e) {
    mexErrMsgIdAndTxt("mex_py_proxy:exception", e.what());
  }
}
