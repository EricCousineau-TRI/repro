r"""
DISCLAIMER: This was ported from TRI's Anzu codebase, with code review from
other TRI employees.

Simple sink clutter generator using pydrake.

Example of running a large amount of scenes:

    bazel run  :generate_poses_sink_clutter -- \
        --start_index=0 --stop_index=100 \
        --max_sim_time=0.5 \
        --projection_mode=height_heuristic \
        --process_count=5

Example of visualizing generated scenes:

    # Terminal 1
    bazel run //tools:drake_visualizer  # TODO: Not accessible!

    # Terminal 2
    bazel run  :generate_poses_sink_clutter -- \
        --start_index=0 --stop_index=10 \
        --max_sim_time=0.5 \
        --projection_mode=height_heuristic \
        --visualize --visualize_height_heuristic

Note that 'height_heuristic' will report longer simulation times, but that
is because it has bodies in contact sooner than without this heuristic. The
poses generated using this will look "better" (e.g. more settled into the
sink).

Taking cues from Naveen's setup:
    https://github.com/RobotLocomotion/drake/tree/f2808c7a/attic/manipulation/scene_generation
"""

# TODO(eric.cousineau): Follow suggestions for other techniques from Stack
# Overflow: https://stackoverflow.com/q/61841013/7829525. See notes in
# `height_heuristic` about this.

import argparse
from functools import partial
from textwrap import indent
import time

import numpy as np
import pandas as pd
import tqdm

from pydrake.geometry import ConnectDrakeVisualizer
from pydrake.math import RigidTransform
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder

from TODO_THIS_DOESNT_ACTUALLY_EXIST_YET import (
    generate_random_uniform_quaternion,
    seed,
)


class AddModelByReparsing:
    """Adds a model by re-parsing every time.

    WARNING: This makes plant construction slow -- it can take as long as
    simulation. Storing parsed results in a subgraph makes construction faster by
    # 7-10x.
    """
    def __init__(self, model_file):
        self._model_file = model_file

    def add_to(self, plant, scene_graph, name):
        return Parser(plant).AddModelFromFile(self._model_file, name)


class ModelManifest:
    def __init__(self):
        self.sink = AddModelByReparsing(TODO_SINK_MODEL)
        self.manipulands = [
            AddModelByReparsing(TODO_PLATE_MODEL),
            AddModelByReparsing(TODO_MUG1_MODEL),
            AddModelByReparsing(TODO_MUG2_MODEL),
        ]


def add_sink_scene(
        model_manifest, plant, scene_graph, get_num_model_instances):
    """Adds model instances of a sink and a given set of manipulands."""
    sink_model = model_manifest.sink.add_to(plant, scene_graph, "sink")
    frame_Sink = plant.GetFrameByName("__model__", sink_model)
    X_WSink = RigidTransform()
    frame_W = plant.world_frame()
    plant.WeldFrames(frame_W, frame_Sink, X_WSink)
    # Add manipulands.
    frame_O_list = []
    index = 0
    for model_spec in model_manifest.manipulands:
        for _ in range(get_num_model_instances(model_spec)):
            new_name = f"model_{index}"
            model_instance = model_spec.add_to(plant, scene_graph, new_name)
            frame_O = plant.GetFrameByName("__model__", model_instance)
            frame_O_list.append(frame_O)
            index += 1
    return frame_O_list, frame_Sink


class GenerateSinkPoseParam:
    """Parameters for genrating poses in the sink."""
    def __init__(self):
        # These represent the two corners of an axis-aligned bounding box that
        # approximates the inside of the sink. These values were taken from the
        # central_square/camera_manifest.yaml, for the sink.
        self.xy_SinkO_lower = np.array([-0.2, -0.2])
        self.xy_SinkO_upper = np.array([0.15, 0.2])
        # N.B. This should ideally be the radius of the largest object.
        self.margin = [0.1, 0.1]
        # Displacement in zs. The higher you put 'em, the longer it takes to
        # settle.
        self.dz = 0.2


def generate_manipuland_sink_poses(
        num_manipulands, param=GenerateSinkPoseParam()):
    """Generates manipuland poses in sink."""
    X_SinkO_list = []
    for i in range(num_manipulands):
        lower = param.xy_SinkO_lower.copy() + param.margin
        upper = param.xy_SinkO_upper.copy() - param.margin
        # Alternate sides to further avoid collisions.
        if i % 2 == 0:
            # Generate pose in left side of sink's coordinate frame
            # (+x forward, +y left).
            lower[1] = 0.
        else:
            # Generate pose in right side of sink's coordinate frame
            # (+x forward, -y right).
            upper[1] = 0.
        xy_SinkO = np.random.uniform(lower, upper)
        # Bump it up one "level" higher.
        z_SinkO = param.dz * (i + 1)
        p_SinkO = np.hstack((xy_SinkO, z_SinkO))
        q_SinkO = generate_random_uniform_quaternion()
        X_SinkO = RigidTransform(quaternion=q_SinkO, p=p_SinkO)
        X_SinkO_list.append(X_SinkO)
    return X_SinkO_list


def set_manipuland_poses(
        plant, context, frame_Sink, frame_O_list, X_SinkO_list):
    """Sets manipulands poses in the plant."""
    frame_W = plant.world_frame()
    X_WSink = plant.CalcRelativeTransform(context, frame_W, frame_Sink)
    for O, X_SinkO in zip(frame_O_list, X_SinkO_list):
        X_WO = X_WSink @ X_SinkO
        set_frame_pose(plant, context, O, X_WO)


class RetryError(RuntimeError):
    """Indicates that we needed to retry / resample."""
    def __init__(self, timing, message):
        self.timing = timing
        super().__init__(message)


# Dtype for recording different sections for time.
timing_dtype = np.dtype([
    ('construction', float),
    ('posturing', float),
    ('projection', float),
    ('simulation', float),
    ])


def make_timer(timing, field):
    """Returns function that, when called, will set `timing[field]` with the
    time elapsed since the function was made."""
    t0 = time.time()

    def end_and_record():
        timing[field] = time.time() - t0
        return timing

    return end_and_record


def generate(
        model_manifest,
        seed_value,
        get_num_model_instances,
        use_height_heuristic=False,
        min_realtime_rate=1.,
        max_sim_time=1.,
        visualize=False,
        visualize_height_heuristic=False,
        sim_rate=0.,
        sim_dt=0.05,
        ):
    """Generates poses for a given scene."""
    # TOOD(eric.cousineau): This may not be fully deterministic? Random state
    # may be leaking in from somewhere else?
    seed(seed_value)
    timing = pd.DataFrame(np.zeros(1, timing_dtype))

    time_construction = make_timer(timing, 'construction')
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=5e-4)
    frame_O_list, frame_Sink = add_sink_scene(
        model_manifest, plant, scene_graph, get_num_model_instances)
    num_manipulands = len(frame_O_list)
    if num_manipulands == 0:
        raise RetryError(time_construction(), "No manipulands (empty sink)")
    plant.Finalize()

    if visualize:
        ConnectDrakeVisualizer(builder, scene_graph)

    diagram = builder.Build()
    d_context = diagram.CreateDefaultContext()
    context = plant.GetMyContextFromRoot(d_context)
    time_construction()

    time_posturing = make_timer(timing, 'posturing')
    # Try to sample different configurations that are collision-free.
    X_SinkO_list = generate_manipuland_sink_poses(num_manipulands)
    set_manipuland_poses(
        plant, context, frame_Sink, frame_O_list, X_SinkO_list)
    # If there are any existing collisions, reject this sample.
    query_object = plant.get_geometry_query_input_port().Eval(context)
    if query_object.HasCollisions():
        raise RetryError(time_posturing(), f"Collision in initial config")
    time_posturing()

    # Initilialize simulator here so that the visualization can be initialized
    # (if enabled).
    simulator = Simulator(diagram, d_context)
    simulator.Initialize()

    time_projection = make_timer(timing, 'projection')
    if use_height_heuristic:

        def height_heuristic_show(debug_name):
            if visualize_height_heuristic:
                print(f"  height_heuristic_show: {debug_name}")
                diagram.Publish(d_context)
                input(f"    Press Enter...\n")

        # All objects in this pose should be collision free. Record their
        # heights.
        zs_free = get_frame_heights(plant, context, frame_O_list)
        # Specify a configuration that should have collisions. We use
        # something slightly below zero because if there are only plates, their
        # frame is defined such that z=0 may not collide with the sink.
        # We purposely make things collide so that we can briefly search for
        # the closest collision free configuration.
        zs_colliding = np.full(num_manipulands, -0.01)
        zs = height_heuristic(
            plant, context, frame_O_list, zs_colliding, zs_free,
            show=height_heuristic_show)
        # Use returned heights in our context.
        set_frame_heights(plant, context, frame_O_list, zs)
        diagram.Publish(d_context)
    time_projection()

    time_simulation = make_timer(timing, 'simulation')
    simulator.set_target_realtime_rate(sim_rate)

    try:
        # TODO(eric.cousineau): Also use settling velocities / height as
        # terminating criteria (to reject or use a simulation)?
        realtime_rates = []
        while d_context.get_time() < max_sim_time:
            t_real_start = time.time()
            simulator.AdvanceTo(d_context.get_time() + sim_dt)
            dt_real = time.time() - t_real_start

            realtime_rates.append(sim_dt / dt_real)
            if np.mean(realtime_rates) < min_realtime_rate:
                raise RetryError(time_simulation(), "Sim too slow")

            zs = get_frame_heights(plant, context, frame_O_list)
            if np.any(zs < 0.):
                raise RetryError(time_simulation(), "Manipuland(s) fell")
    except RuntimeError as e:
        # Try to mitigate solver failures.
        is_solver_failure = (
           "discrete update solver failed to converge" in str(e))
        if is_solver_failure:
            raise RetryError(time_simulation(), "Solver failure")
        else:
            raise
    time_simulation()

    # Compute poses w.r.t. sink and return.
    X_SinkO_list = []
    for i, frame_O in enumerate(frame_O_list):
        X_SinkO = plant.CalcRelativeTransform(context, frame_Sink, frame_O)
        X_SinkO_list.append(X_SinkO)
    return X_SinkO_list, timing


def height_heuristic(
        plant, context, frame_O_list, zs_colliding, zs_free,
        iter_max=20, dz=0.05, show=lambda x: None):
    """Tries to find a minimum collisino-free height by incrementing the height
    for a set of objects that should already be staggered in height.

    Arguments:
        plant: Plant with attached SceneGraph).
        context: Context that can be used with query port.
        frame_O_list: List of frames corresponding to objects to position.
        zs_colliding: Heights that should have collision(s).
        zs_free: Heights that should be collision-free.
        iter_max: Maximum number of iterations to take for each object.
        dz: Change in height to increase for each object.
        show: Function to visualize iteration (using context). Of the form
            show(debug_name).
    """
    # TODO(eric.cousineau): Alternative, project along ray.
    # TODO(eric.cousineau): Alternative, just get penetration depth along z,
    # and just move up by that depth plus a buffer.
    # N.B. CollisionRemover is generally too slow to be useful (~1-10s).
    # N.B. I (Eric) tried writing a brief MathematicalProgram that optimized on
    # height. Did not work well.

    num_manipulands = len(frame_O_list)
    assert len(zs_colliding) == num_manipulands
    assert len(zs_free) == num_manipulands
    # Ensure all non-colliding heights are above the colliding heights.
    assert np.all(zs_colliding <= zs_free)
    # Ensure all collision-free heights are strictly monotonically increasing.
    assert np.all(np.diff(zs_free) > 0.)
    query_object = plant.get_geometry_query_input_port().Eval(context)
    set_frame_heights(plant, context, frame_O_list, zs_colliding)
    show("colliding")
    assert query_object.HasCollisions()
    set_frame_heights(plant, context, frame_O_list, zs_free)
    assert not query_object.HasCollisions()
    zs = np.zeros(num_manipulands)
    # Initialize "decision" variable with all manipulands being collision-free.
    zs[:] = zs_free
    z_colliding_min = np.min(zs_colliding)
    z_prev = z_colliding_min
    for i in range(num_manipulands):
        # Reset near the height of the previous object.
        zs[i] = (z_colliding_min + z_prev) / 2
        for k in range(iter_max):
            zs_readable = ", ".join(f"{z:0.2g}" for z in zs)
            show(f"manipuland[{i}], iter[{k}], zs = [{zs_readable}]")
            set_frame_heights(plant, context, frame_O_list, zs)
            if not query_object.HasCollisions():
                break
            zs[i] += dz
        else:
            # Stop trying and use collision-free positions.
            zs[i] = zs_free[i]
        z_prev = zs[i]
    return zs


def set_frame_pose(plant, context, frame_O, X_WO):
    """Sets the pose of a frame attached to floating body."""
    body = frame_O.body()
    assert body.is_floating()
    X_OB = frame_O.GetFixedPoseInBodyFrame().inverse()
    X_WB = X_WO @ X_OB
    plant.SetFreeBodyPose(context, body, X_WB)


def set_frame_heights(plant, context, frame_O_list, zs):
    """Sets the height of a list of frames."""
    frame_W = plant.world_frame()
    for frame_O, z_WO in zip(frame_O_list, zs):
        X_WO = plant.CalcRelativeTransform(context, frame_W, frame_O)
        p_WO = X_WO.translation().copy()
        p_WO[2] = z_WO
        X_WO.set_translation(p_WO)
        set_frame_pose(plant, context, frame_O, X_WO)


def get_frame_heights(plant, context, frame_O_list):
    """Gets the height from a list of frames."""
    zs = []
    for frame_O in frame_O_list:
        frame_W = plant.world_frame()
        X_WO = plant.CalcRelativeTransform(context, frame_W, frame_O)
        z_WO = X_WO.translation()[2]
        zs.append(z_WO)
    return np.array(zs)


def worker(indices, retry_count=10, **kwargs):
    """Generator that consumes indices and dispatches to `generate`, while
    allowing for retries."""
    model_manifest = ModelManifest()
    # N.B. Choose a retry offset to avoid colliding with the seeds for other
    # (practical) values of `index`.
    retry_offset = 10000001
    for index in indices:
        timing = pd.DataFrame(np.zeros(1, timing_dtype))
        retries = []
        seed_value = index
        for attempt in range(retry_count):
            try:
                X_SinkO_list, timing_i = generate(
                    model_manifest, seed_value, **kwargs)
                timing += timing_i
                break
            except RetryError as e:
                timing += e.timing
                retries.append(
                    f"Attempt {attempt + 1} / {retry_count} needs retry: {e}")
                seed_value += retry_offset
        else:
            raise RetryError(
                timing,
                f"Index {index}: Too many retries:\n  " + "\n  ".join(retries))
        yield X_SinkO_list, timing


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--start_index", type=int, default=0,
        help="Starting index for the scenes (inclusive).")
    parser.add_argument(
        "--stop_index", type=int, default=10,
        help="Stopping index for the scenes (non-inclusive).")
    parser.add_argument(
        "--max_num_model_instances", type=int, default=3,
        help="Maximum number of instances per model.")
    parser.add_argument(
        "--max_sim_time", type=float, default=0.5,
        help="Maximum amount of time to simulate.")
    parser.add_argument(
        "--min_realtime_rate", type=float, default=0.9,
        help="Minimum realtime rate for sim. If the sim is lower than this, "
             "the scene is rejected.")
    parser.add_argument(
        "--projection_mode", type=str, default='height_heuristic',
        choices=['none', 'height_heuristic'],
        help="Choose a method to project the initial poses closer to the "
             "sink.")
    parser.add_argument(
        "--visualize", action="store_true",
        help="Publish to Drake Visualizer to see each scene being generated.")
    parser.add_argument(
        "--visualize_height_heuristic", action="store_true",
        help="Show iterations of height heuristic.")
    parser.add_argument(
        "--sim_rate", type=float, default=0.,
        help="Simulation rate. Only used with --visualize.")
    parser.add_argument(
        "--process_count", type=int, default=0,
        help="Number of processes to use with `parallel_work(...)`")
    args = parser.parse_args()

    indices = list(range(args.start_index, args.stop_index))
    if args.projection_mode == 'height_heuristic':
        use_height_heuristic = True
    else:
        use_height_heuristic = False

    def get_num_model_instances(model_spec):
        return np.random.randint(0, args.max_num_model_instances + 1)

    worker_bind = partial(
        worker,
        use_height_heuristic=use_height_heuristic,
        max_sim_time=args.max_sim_time,
        min_realtime_rate=args.min_realtime_rate,
        get_num_model_instances=get_num_model_instances,
    )

    if args.visualize:
        assert args.process_count == 0
        worker_gen = worker_bind(
            indices,
            visualize=True,
            visualize_height_heuristic=args.visualize_height_heuristic,
            sim_rate=args.sim_rate,
        )
        timings = []
        for index in indices:
            print("\n")
            print(f"Index: {index}")
            print()
            X_WO_list, timing = next(worker_gen)
            print("  Timing:")
            print(indent(str(timing), '  '))
            print()
            timings.append(timing)
            input("  Press Enter...")
    else:
        assert args.sim_rate == 0.
        assert not args.visualize_height_heuristic
        pairs = parallel_work(
            worker_bind,
            indices,
            process_count=args.process_count,
            progress_cls=tqdm.tqdm,
        )
        X_WO_list_set, timings = zip(*pairs)

    # Show timing.
    timings = pd.concat(timings)
    timings["total"] = np.sum(timings, axis=1)
    print()
    print(f"Timing:")
    print(pd.DataFrame({
        "sum": timings.sum(),
        "mean": timings.mean(),
        "std": timings.std(),
    }))


if __name__ == "__main__":
    main()
