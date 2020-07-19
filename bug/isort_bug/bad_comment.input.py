"""
Example edge case for isort==5.1.2:

    isort --stdout ./bad_comment.input.py | tee ./bad_comment.output.py

See: https://github.com/RobotLocomotion/drake/pull/13709#pullrequestreview-451092149
"""

# Comment for A.
import a
# Comment for B - not A!
import b
