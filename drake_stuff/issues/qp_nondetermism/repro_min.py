#!/usr/bin/env python
import pickle

import numpy as np

from pydrake.solvers import (
    MathematicalProgram,
    OsqpSolver,
    SolverOptions,
)


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def check(adaptive_rho):
    print(f"[ adaptive_rho={adaptive_rho} ]")
    workspace = load_pickle("./workspace_min.pkl")
    u_set = set()

    for k in range(100):
        solver = OsqpSolver()
        prog = MathematicalProgram()

        num_v = 7
        num_u = 7

        M = workspace["M"]
        C = workspace["C"]
        tau_g = workspace["tau_g"]

        Iv = np.eye(num_v)

        vd_star = prog.NewContinuousVariables(num_v, "vd_star")
        u_star = prog.NewContinuousVariables(num_u, "u_star")
        vd_u_star = np.concatenate([vd_star, u_star])

        prog.AddLinearEqualityConstraint(
            np.hstack([M, -Iv]), -C + tau_g, vd_u_star
        )

        # For sake of brevity, only add one task, and ignore adding limits.
        task = workspace["task"]
        task_cost_A, task_cost_b, task_cost_proj = map(
            task.get, ("task_cost_A", "task_cost_b", "task_cost_proj")
        )
        prog.Add2NormSquaredCost(
            task_cost_proj @ task_cost_A,
            task_cost_proj @ task_cost_b,
            vd_star,
        )

        solver_options = SolverOptions()
        solver_options.SetOption(
            solver.solver_id(), "adaptive_rho", adaptive_rho
        )

        result = solver.Solve(prog, None, solver_options)
        assert result.is_success()
        u = result.GetSolution(u_star)
        u_set.add(tuple(u))

    print(f"num unique: {len(u_set)}")

    # Print diffs.
    us = [np.array(u) for u in u_set]
    u0 = us[0]
    for u in us[1:]:
        print(u - u0)


def main():
    count = 10
    for i in range(count):
        check(adaptive_rho=1)
    print()
    for i in range(count):
        check(adaptive_rho=0)


if __name__ == "__main__":
    main()
