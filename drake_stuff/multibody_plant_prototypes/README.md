# Prototype `MultibodyPlant` Functionality for Drake

## `MultibodyPlantSubgraph`

Taken from prototype code in Anzu. See:

- `multibody_plant_subgraph.py`
- `multibody_plant_subgraph_test.py`

Prototyping towards the following issues:

- <https://github.com/RobotLocomotion/drake/issues/12203>
- <https://github.com/RobotLocomotion/drake/issues/13074>
- <https://github.com/RobotLocomotion/drake/issues/13177>

To visualize the test cases:

```sh
cd .../multibody_plant_prototypes

# Build
bazel build //... @drake_artifacts//...

# Terminal 1
./bazel-bin/external/drake_artifacts/drake_visualizer

# Terminal 2
./bazel-bin/multibody_plant_subgraph_test --visualize --verbose
```

## `generate_poses_sink_clutter.py`

Prototyping towards:

<https://stackoverflow.com/questions/61841013/how-to-dampen-multibodyplants-compliant-contact-model-in-a-simulation>

Generates objects in a sink for clutter; trying to use heuristics to be "fast".

See also:

- `multibody_plant_energy_hacks.py`
