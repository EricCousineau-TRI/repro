#include <iostream>

// https://stackoverflow.com/questions/15011674/is-it-possible-to-dereference-variable-ids
// Can wrap any pythonic types...

#include <mex.h>
#include <matrix.h>

#include "func.h"

typedef mwSize mxInt64;

double call_mex(void* mex_func_raw, double in) {
  const mxArray* mex_func_handle = static_cast<const mxArray*>(mex_func_raw);
  const mxArray* mex_in = mxCreateDoubleScalar(in);

  const int nrhs = 2;
  const mxArray* prhs[nrhs] = {mex_func_handle, mex_in};

  const int nlhs = 1;
  mxArray* plhs[1];

  // mexCallMATLABWithTrap
  mexCallMATLAB(nlhs, plhs, nrhs, prhs, "feval");
  double out = *mxGetPr(plhs[0]);

  for (int i = 0; i < nlhs; ++i) {
    mxDestroyArray(plhs[i]);
  }
  for (int i = 0; i < nrhs; ++i) {
    mxDestroyArray(prhs[i]);
  }
  return out;
}

// Wrap MEX function call.
void mexFunction(int nlhs,mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nlhs != 2) {
    mexErrMsgTxt("nlhs != 2 : [a, b] = f(handle)");
    return;
  }
  if (nrhs != 1) {
    mexErrMsgTxt("nrhs != 1 : [a, b] = f(handle)");
    return;
  }

  if (!mxIsClass(prhs[0] , "function_handle")) {
    mexErrMsgTxt("First input argument is not a function handle.");
  }

  mxArray* mex_func_handle = prhs[0];

  // [mx_caller, func_handle] = ...
  plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
  mxInt64& mex_caller_raw = *static_cast<mxInt64*>(mxGetData(plhs[0]));
  mex_caller_raw = static_cast<mxInt64>(&call_mex);

  plhs[1] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
  mxInt64& mex_func_handle_raw = *static_cast<mxInt64*>(mxGetData(plhs[1]));
  mex_func_handle_raw = static_cast<mxInt64>(mex_func_handle);
}
