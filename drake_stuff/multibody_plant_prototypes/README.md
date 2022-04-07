# Prototype `MultibodyPlant` Functionality for Drake

Tested on Ubuntu 20.04, CPython 3.8, requires `sudo apt install python3-venv`.

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

# Source setup (run `pip install` insdie virtualenv if necessary)
# Run for each terminal
source ./setup.sh

# Terminal 1
python -m pydrake.visualization.meldis -w

# Terminal 2
python ./test/multibody_plant_subgraph_test --visualize --verbose
```

WARNING: The visualization in `meshcat` (via `meldis`) is presently
(2022-04-07) a bit awkward when reloading due to over-the-wire delay.

## `generate_poses_sink_clutter.py`

Prototyping towards:

<https://stackoverflow.com/questions/61841013/how-to-dampen-multibodyplants-compliant-contact-model-in-a-simulation>

Generates objects in a sink for clutter; trying to use heuristics to be "fast".

See also:

- `multibody_plant_energy_hacks.py`
