mp = pyimport_proxy('pydrake.solvers.mathematicalprogram');

% QP test
prog = mp.MathematicalProgram();
x = prog.NewContinuousVariables(int32(2),'x'); % Note: py_relax_overload not used
prog.AddLinearConstraint(x(1) >= 1);
prog.AddLinearConstraint(x(2) >= 1);
prog.AddQuadraticCost(eye(2), zeros(2, 1), x);
result = prog.Solve();

assert(result == mp.SolutionResult.kSolutionFound);

x_sol = double(prog.GetSolution(x));
x_expected = [1; 1];

maxabs = @(x) max(abs(x(:)));
tol = sqrt(eps);
assert(maxabs(x_sol - x_expected) < tol);

%% Additional stuff
size(x)
x(:)
(2 + x^2) * (x(1) + x(2))
