% Adapted from: drake:96c2ca1:drake/matlab/solvers/test/testMathematicalProgram.m

% Simple example of calling MathematicalProgram through the python
% interface.

mp = pyimport_proxy('pydrake.solvers.mathematicalprogram');

% QP test
prog = mp.MathematicalProgram();
% TODO: Change this to a proper Eigen array.
x = prog.NewContinuousVariables(int32(2),'x');
prog.AddLinearConstraint(x.item(0) >= 1);
prog.AddLinearConstraint(x.item(1) >= 1);
% TODO: Presently interpreted as dtype=object. See if we can get pybind11
% to use the actual type.
prog.AddQuadraticCost(eye(2), zeros(2), x);
result = prog.Solve()

% Note: int32(0) is kSolutionFound; can't reference it directly.
assert(result == mp.SolutionResult(int32(0)));

x_sol = prog.GetSolution(x);
assert(abs(x_sol(1)-1.0)<1e-6);
assert(abs(x_sol(2)-1.0)<1e-6);
