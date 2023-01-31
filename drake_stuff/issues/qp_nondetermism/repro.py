#!/usr/bin/env python
import pickle

import numpy as np

from pydrake.solvers import MathematicalProgram, OsqpSolver

    
def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def main():
    workspace = load_pickle("./workspace.pkl")
    u_set = set()

    for k in range(1000):
        solver = OsqpSolver()
        prog = MathematicalProgram()

        M = workspace["M"]
        C = workspace["C"]
        tau_g = workspace["tau_g"]
        vd_min, vd_max = map(workspace.get, ("vd_min", "vd_max"))
        u_min, u_max = map(workspace.get, ("u_min", "u_max"))

        num_v = len(vd_min)
        num_u = len(u_min)

        Iv = np.eye(num_v)

        vd_star = prog.NewContinuousVariables(num_v, "vd_star")
        u_star = prog.NewContinuousVariables(num_u, "u_star")
        vd_u_star = np.concatenate([vd_star, u_star])

        prog.AddLinearEqualityConstraint(
            np.hstack([M, -Iv]), -C + tau_g, vd_u_star
        )

        for task in workspace["tasks"].values():
            task_cost_A, task_cost_b, task_cost_proj = map(
                task.get, ("task_cost_A", "task_cost_b", "task_cost_proj")
            )
            prog.Add2NormSquaredCost(
                task_cost_proj @ task_cost_A,
                task_cost_proj @ task_cost_b,
                vd_star,
            )

        prog.AddBoundingBoxConstraint(vd_min, vd_max, vd_star)
        prog.AddBoundingBoxConstraint(u_min, u_max, u_star)

        result = solver.Solve(prog)
        assert result.is_success()
        u = result.GetSolution(u_star)
        u_set.add(tuple(u))

    print(f"num unique: {len(u_set)}")

    # Print diffs.
    us = [np.array(u) for u in u_set]
    u0 = us[0]
    for u in us[1:]:
        print(u - u0)


if __name__ == "__main__":
    main()
