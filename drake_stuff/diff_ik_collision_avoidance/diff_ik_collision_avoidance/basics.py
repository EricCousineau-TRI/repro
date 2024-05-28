from pathlib import Path

import numpy as np

from pydrake.all import (
    PackageMap,
    Parser,
    ProcessModelDirectives,
    RobotDiagramBuilder,
    SceneGraphCollisionChecker,
)

import diff_ik_collision_avoidance as _me


def MakeMyDefaultPackageMap():
    project_dir = Path(_me.__file__).parent
    package_map = PackageMap()
    package_map.Add("diff_ik_collision_avoidance", str(project_dir))
    return package_map


def ProcessMyModelDirectives(model_directives, plant):
    parser = Parser(plant)
    parser.package_map().AddMap(MakeMyDefaultPackageMap())
    added_models = ProcessModelDirectives(
        model_directives, plant, parser=parser
    )
    return added_models


def MakeRobotDiagramBuilder(directives, time_step):
    robot_builder = RobotDiagramBuilder(time_step=time_step)
    ProcessMyModelDirectives(directives, robot_builder.plant())
    return robot_builder


def configuration_distance(q1, q2):
    """A boring implementation of ConfigurationDistanceFunction."""
    # TODO(eric.cousineau): Is there a way to ignore this for just a config
    # distance function?
    # What about a C++ function, without Python indirection, for better speed?
    return np.linalg.norm(q1 - q2)


def MakeCollisionChecker(robot_diagram, model_instances):
    return SceneGraphCollisionChecker(
        model=robot_diagram,
        configuration_distance_function=configuration_distance,
        edge_step_size=0.05,
        env_collision_padding=0.0,
        self_collision_padding=0.0,
        robot_model_instances=model_instances,
    )
