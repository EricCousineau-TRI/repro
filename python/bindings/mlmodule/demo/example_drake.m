% Raw Python

% QP test
prog = py.pydrake.solvers.mathematicalprogram.MathematicalProgram();
x = prog.NewContinuousVariables(int32(2),'x'); % Note: py_relax_overload not used
prog.AddLinearConstraint(x.item(0) >= 1);
prog.AddLinearConstraint(x.item(1) >= 1);
prog.AddQuadraticCost(py.numpy.eye(2), py.numpy.zeros(2), x);
result = prog.Solve();

assert(result == py.pydrake.solvers.mathematicalprogram.SolutionResult(int32(0)));

x_sol_py = prog.GetSolution(x);
x_sol = [x_sol_py.item(0); x_sol_py.item(1)];
x_expected = [1; 1];

maxabs = @(x) max(abs(x(:)));
tol = sqrt(eps);
assert(maxabs(x_sol - x_expected) < tol);

%% Additional stuff
(2 + x^2) * (x.item(0) + x.item(1))
