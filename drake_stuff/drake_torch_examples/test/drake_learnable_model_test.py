import unittest

import numpy as np
import torch

from pydrake.multibody.parsing import LoadModelDirectives, Parser
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import BodyIndex

from my_package.not_exported_soz import (
    ProcessMyModelDirectives,
    resolve_path,
)
from my_package.drake_learnable_model import (
    DrakeLearnableModel,
    LogCholeskyInertialParameter,
    RawInertialParameter,
    VectorInertialParameter,
    get_identifiable_bodies,
    get_plant_inertial_param,
    plant_inverse_dynamics,
)
from my_package.drake_torch_autodiff import (
    autodiff_to_value_and_grad,
    torch_to_initialize_autodiff_list,
)


def add_model_panda_j7(plant):
    model_file = resolve_path("package://my_package/not_exported_soz/haptic/panda_j7.urdf")
    model = Parser(plant).AddModelFromFile(model_file)
    # Posture the link with rotary axis aligned with gravity.
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("panda_joint7_inboard"),
    )
    bodies = get_identifiable_bodies(plant)
    return model, bodies


def add_model_acrobot(plant):
    model_file = resolve_path(
        "package://drake/multibody/benchmarks/acrobot/acrobot.sdf"
    )
    model = Parser(plant).AddModelFromFile(model_file)
    bodies = get_identifiable_bodies(plant)
    return model, bodies


def add_model_panda(plant):
    model_file = resolve_path(
        "package://my_package/not_exported_soz/haptic/add_haptic_test_panda.yaml"
    )
    directives = LoadModelDirectives(model_file)
    ProcessMyModelDirectives(directives, plant)
    model = plant.GetModelInstanceByName("right::panda")
    bodies = get_identifiable_bodies(plant)
    return model, bodies


def ensure_zero_grad(params):
    for param in params:
        param.grad = None


class Test(unittest.TestCase):
    @torch.no_grad()
    def test_inertial(self):
        """
        Ensure inertial parmaterizations are invertible.

        Checks via round-trip ("rt").
        """
        plant = MultibodyPlant(time_step=0.0)
        # _, bodies = add_model_panda(plant)
        _, bodies = add_model_panda_j7(plant)
        plant.Finalize()
        context = plant.CreateDefaultContext()

        def to_torch(x):
            return torch.from_numpy(x).to(torch.float32)

        masses, coms, rot_inertias = map(
            to_torch, get_plant_inertial_param(plant, context, bodies)
        )

        cls_list = [
            RawInertialParameter,
            VectorInertialParameter,
            LogCholeskyInertialParameter,
        ]
        for cls in cls_list:
            with self.subTest(cls=cls):
                inertial = cls(masses, coms, rot_inertias)
                masses_rt, coms_rt, rot_inertias_rt = inertial()
                torch.testing.assert_close(masses, masses_rt)
                torch.testing.assert_close(coms, coms_rt)
                torch.testing.assert_close(rot_inertias, rot_inertias_rt)

    def check_forward_and_backward(
        self,
        add_model,
        inertial_param_cls,
        *,
        batch_shape=(),
    ):
        plant = MultibodyPlant(time_step=0.0)
        _, bodies = add_model(plant)
        plant.Finalize()

        inverse_dynamics = DrakeLearnableModel(
            plant, bodies, inertial_param_cls=inertial_param_cls
        )
        num_q = plant.num_positions()
        # For now, simplify expectations.
        assert plant.num_velocities() == num_q
        dim_q = batch_shape + (num_q,)
        q0 = torch.zeros(dim_q)
        v0 = torch.zeros(dim_q)
        vd0 = torch.ones(dim_q)

        # Ensure we can backprop a few times.
        for i in range(5):
            ensure_zero_grad(inverse_dynamics.parameters())
            tau = inverse_dynamics(q0, v0, vd0)
            fake_loss = tau.sum()
            fake_loss.backward()

        return inverse_dynamics.inertial

    def check_model(self, add_model):
        # Unbatched.
        inertial_unc = self.check_forward_and_backward(
            add_model,
            RawInertialParameter,
        )
        inertial_chol = self.check_forward_and_backward(
            add_model,
            LogCholeskyInertialParameter,
        )
        # Batched.
        self.check_forward_and_backward(
            add_model, RawInertialParameter, batch_shape=(2,),
        )
        # Only return unbatched.
        return inertial_unc, inertial_chol

    def test_1_panda_j7(self):
        self.check_model(add_model_panda_j7)

    def test_2_acrobot(self):
        self.check_model(add_model_acrobot)

    def test_3_panda(self):
        self.check_model(add_model_panda)


if __name__ == "__main__":
    unittest.main()
