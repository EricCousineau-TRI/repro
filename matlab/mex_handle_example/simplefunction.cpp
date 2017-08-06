// From: https://www.mathworks.com/matlabcentral/answers/96775-how-do-i-pass-function-handles-to-c-mex-function-in-matlab-7-8-r2009a

// A simplefunction.cpp that takes 2 arguments
// 1) A function handle
// 2) A double value
// and returns the output of the same

#include <iostream>
#include "mex.h"
#include "matrix.h"

using namespace std;
extern void _main();

void mexFunction(int nlhs,mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  mxArray *lhs,*rhs[2];
  double A;
    
  // Argument Checking:
  // First Argument should be a Function Handle
  if( !mxIsClass( prhs[0] , "function_handle")) {
    mexErrMsgTxt("First input argument is not a function handle.");
  }
  // Second Argument is a Double Scalar
  if (!mxIsClass(prhs[1], "double")||(mxGetM(prhs[1])>1)||(mxGetN(prhs[1])>1)) {
    mexErrMsgTxt("Second input argument is not a double scalar.");
  }
  
  //processing on input arguments
  rhs[0] = const_cast<mxArray *>(prhs[0]); 
  A = *mxGetPr(prhs[1]);
  rhs[1] = mxCreateDoubleScalar(A);
  mexCallMATLAB(1,&lhs,2,rhs,"feval");
  mexPrintf("Output of the function handle %lf\n", *mxGetPr(lhs));
  
  // Output Argument
  plhs[0] = mxCreateDoubleScalar(*mxGetPr(lhs));
    
  // Clean UP
  mxDestroyArray(lhs);
  mxDestroyArray(rhs[1]);
}
