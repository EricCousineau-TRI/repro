% Adapted from: drake:96c2ca1:drake/matlab/solvers/test/testMathematicalProgram.m

% Simple example of calling MathematicalProgram through the python
% interface.

mp = pyimport_proxy('pydrake.solvers.mathematicalprogram');

% QP test
prog = mp.MathematicalProgram();
x = prog.NewContinuousVariables(int32(2),'x');
% TODO: Consider making PyProxy array-able, or a separte item for numpy
% stuf.

% TODO: Make indexing work, somehow...
prog.AddLinearConstraint(x.item(0) >= 1); % Hard to implement subsref
prog.AddLinearConstraint(x.item(1) >= 1);
% TODO: Presently interpreted as dtype=object. See if we can get pybind11
% to use the actual type.
prog.AddQuadraticCost(eye(2), zeros(2, 1), x);
result = prog.Solve();

% Should extract module from Python to refer to static variables...
% Not great...
assert(result == mp.py.SolutionResult.kSolutionFound);

x_sol = prog.GetSolution(x);
assert(abs(x_sol(1)-1.0)<1e-6);
assert(abs(x_sol(2)-1.0)<1e-6);
