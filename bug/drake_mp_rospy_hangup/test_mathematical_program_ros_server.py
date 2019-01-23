import rospy
import actionlib
import actionlib_tutorials.msg

import numpy as np


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


def test_mathematical_program(goal=None):
    print "\n\n\nstarting mathematical program test"

    if goal is not None:
        print "inside ROS callback"

    # does work inside ROS callback
    run_simple_mathematical_program()

    # doesn't work inside ROS callback
    run_complex_mathematical_program()

    print "finished cleanly"
#
    if goal is not None:
        result = actionlib_tutorials.msg.FibonacciResult()
        action_server.set_succeeded(result)
#
# #
rospy.init_node("test")
action_name = "MathematicalProgram"
action_server = actionlib.SimpleActionServer(action_name, actionlib_tutorials.msg.FibonacciAction, execute_cb=test_mathematical_program, auto_start = False)

action_server.start()

if __name__ == "__main__":
    print "initializing node"

    test_mathematical_program()

    rospy.spin()

print "finished cleanly"
