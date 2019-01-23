import numpy as np

from threading import Thread

from pydrake.all import MathematicalProgram

def cost(x):
    c = 1.0 * x[0]
    return c

def constraint(x):
    return np.array([x[0]])

def quad_constraint(x):
    return np.array([x[0]**2])


def run_simple_mathematical_program():
    print "\n\nsimple mathematical program"
    mp = MathematicalProgram()
    x = mp.NewContinuousVariables(1, "x")
    mp.AddLinearCost(x[0] * 1.0)
    mp.AddLinearConstraint(x[0] >= 1)
    print mp.Solve()
    print mp.GetSolution(x)
    print "(finished) simple mathematical program"

def run_complex_mathematical_program():
    print "\n\ncomplex mathematical program"
    mp = MathematicalProgram()
    x = mp.NewContinuousVariables(1, 'x')
    mp.AddCost(cost, x)
    mp.AddConstraint(quad_constraint, [1.0], [2.0], x)
    mp.SetInitialGuess(x, [1.1])
    print mp.Solve()
    res = mp.GetSolution(x)
    print res

    print "(finished) complex mathematical program"


def run_mathematical_program_in_thread():
    t = Thread(target=run_simple_mathematical_program)
    t.start()
    t.join()

    # doesn't work inside ROS callback
    t = Thread(target=run_complex_mathematical_program)
    t.start()
    t.join()


if __name__ == "__main__":

    run_mathematical_program_in_thread()

print "finished cleanly"
