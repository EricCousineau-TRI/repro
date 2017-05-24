% Adapted from: drake:96c2ca1:drake/matlab/solvers/test/testMathematicalProgram.m

% Simple example of calling MathematicalProgram through the python
% interface.

py_mp = pyimport('pydrake.solvers.mathematicalprogram');
mp = py_proxy(py_mp);

% QP test
prog = mp.MathematicalProgram();
x = prog.NewContinuousVariables(int32(2),'x'); % Note: py_relax_overload not used
% TODO: Consider making PyProxy array-able, or a separate item for numpy
% stuff.
% subsref() seems to throw a slew of complications into the mix.
prog.AddLinearConstraint(x.item(0) >= 1);
prog.AddLinearConstraint(x.item(1) >= 1);
prog.AddQuadraticCost(eye(2), zeros(2, 1), x);
result = prog.Solve();

% Need to use Python to refer to static variables...
% Not great...
assert(result == py_mp.SolutionResult.kSolutionFound);

x_sol = prog.GetSolution(x);
x_expected = [1; 1];

maxabs = @(x) max(abs(x(:)));
tol = sqrt(eps);
assert(maxabs(x_sol - x_expected) < tol);
