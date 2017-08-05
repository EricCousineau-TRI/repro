#include <dlfcn.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <mex.h>
#include <matrix.h>

/*
Reference counting:
For MEX, consider `mexMakeArrayPersistent`, or undocumented
https://stackoverflow.com/questions/19813718/mex-files-how-to-return-an-already-allocated-matlab-array
Cannot return directly from this to `plhs`, though...
Can tinker with `format debug` to see addresses.

(Hopefully this is not a deep copy.)

For Python objects in MATLAB, since MATLAB treats Python objects well, we
not actually need reference counting.
If we do, __del__ should be sufficient.
*/

using std::cout;
using std::endl;
using std::string;

namespace {

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

string mxToStdString(const mxArray* mx_in) {
  // From: mxmalloc.c
  const int len = mxGetN(mx_in) + 1;
  char* buffer = static_cast<char*>(mxMalloc(len));
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
    "    'get_c_func_ptrs' Get C pointers, using opaque types, to pass to Python.\n" \
    "        [c_func_ptrs_struct] = mex_py_proxy('get_c_func_ptrs')\n" \
    "    'simple' Call Python from MATLAB from C\n" \
    "        [] = mex_py_proxy('simple')\n" \
    "    'help' Show usage.";

// <c_func_ptrs>
// These functions will be passed from MEX to MATLAB to Python.

// TODO(eric.cousineau): Consider testing ctypes.py_object and void* combos, per
// this example:
// https://stackoverflow.com/questions/15011674/is-it-possible-to-dereference-variable-ids  

int c_simple() {
  cout << "c: call matlab - start" << endl;
  mxArray* mx_in[] = {mxCreateUint64Value(1), mxCreateUint64Value(2)};
  mxArray* mx_out[] = {nullptr};
  mexCallMATLAB(1, mx_out, 2, mx_in, "simple");
  cout << "  y = " << mxGetUint64(mx_out[0]) << endl;
  cout << "c: call matlab - finish" << endl;
  mxDestroyArray(mx_in[0]);
  mxDestroyArray(mx_in[1]);
  return 0;
}

int c_mx_raw_ref_incr(mx_raw_t mx_raw) {
  mxArray* mx_mx_raw = mxCreateUint64Value(mx_raw);
  mxArray* mx_in[] = {mx_mx_raw};
  mexCallMATLAB(0, nullptr, 1, mx_in, "MexPyProxy.mx_raw_ref_incr");
  mxDestroyArray(mx_mx_raw);
}

int c_mx_raw_ref_decr(mx_raw_t mx_raw) {
  mxArray* mx_mx_raw = mxCreateUint64Value(mx_raw);
  mxArray* mx_in[] = {mx_mx_raw};
  mexCallMATLAB(0, nullptr, 1, mx_in, "MexPyProxy.mx_raw_ref_decr");
  mxDestroyArray(mx_mx_raw);
}

py_raw_t c_mx_feval_py_raw(mx_raw_t mx_raw_handle, int nargout, py_raw_t py_raw_in) {
  mxArray* mx_mx_raw_handle = mxCreateUint64Value(mx_raw_handle);
  mxArray* mx_nargout = mxCreateUint64Value(nargout);
  mxArray* mx_py_raw_in = mxCreateUint64Value(py_raw_in);

  const int nrhs = 3;
  mxArray* mx_in[nrhs] = {mx_mx_raw_handle, mx_nargout, mx_py_raw_in};
  const int nlhs = 1;
  mxArray* mx_out[nlhs] = {NULL};
  // TODO(eric.cousineau): Handle MATLAB exceptions.
  mexCallMATLAB(nlhs, mx_out, nrhs, mx_in, "MexPyProxy.mx_feval_py_raw");
  py_raw_t py_raw_out = reinterpret_cast<py_raw_t>(mxGetUint64(mx_out[0]));

  mxDestroyArray(mx_mx_raw_handle);
  mxDestroyArray(mx_nargout);
  mxDestroyArray(mx_py_raw_in);

  return py_raw_out;
}

// </c_func_ptrs>

// Create MATLAB struct containing raw values pointing to functions.
mxArray* get_c_func_ptrs() {
  // Reference example: mxcreatestructarray.c
  const int n = 4;
  const char* names[n] = {
    "c_mx_feval_py_raw",
    "c_simple",
    "c_mx_raw_ref_incr",
    "c_mx_raw_ref_decr",
  };
  // Store easily-accessible pointers.
  auto raw_cast = [](auto x) {
    return reinterpret_cast<mx_raw_t>(x);
  };
  mx_raw_t ptrs_raw[n] = {
    raw_cast(&c_mx_feval_py_raw),
    raw_cast(&c_simple),
    raw_cast(&c_mx_raw_ref_incr),
    raw_cast(&c_mx_raw_ref_decr)
  };
  mwSize dims[2] = {1, 1};
  mxArray* s = mxCreateStructArray(2, dims, n, names);
  for (int i = 0; i < n; ++i) {
    const char* name = names[i];
    mx_raw_t ptr_raw = ptrs_raw[i];
    int field_index = mxGetFieldNumber(s, name);
    mxArray* value = mxCreateUint64Value(ptr_raw);
    mxSetFieldByNumber(s, 0, field_index, value);
  }
  return s;
}

}  // namespace

// Wrap MEX function call.
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  try {
    ex_assert(nrhs >= 1, usage);
    string op = mxToStdString(prhs[0]);
    if (op == "get_c_func_ptrs") {
      ex_assert(nrhs == 1, usage);
      ex_assert(nlhs == 1, usage);
      plhs[0] = get_c_func_ptrs();
    } else if (op == "help") {
      mexPrintf("%s\n", usage.c_str());
    } else if (op == "simple") {
      ex_assert(nrhs == 1, usage);
      ex_assert(nlhs == 0, usage);
      c_simple();
    } else {
      throw std::runtime_error("Invalid op: " + op);
    }
  }
  catch (const std::exception& e) {
    mexErrMsgIdAndTxt("mex_py_proxy:exception", e.what());
  }
}
