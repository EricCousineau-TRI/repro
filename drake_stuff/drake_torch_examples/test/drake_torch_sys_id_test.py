import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from pydrake.multibody.plant import MultibodyPlant

from not_exported_soz.containers import dict_items_zip
from anzu.not_exported_soz.dair_pll_inertia import parallel_axis_theorem
from anzu.drake_torch_dynamics import (
    InertialEntropicDivergence,
    LogCholeskyComInertialParameter,
    LogCholeskyInertialParameter,
    LogCholeskyLinAlgInertialParameter,
    PseudoInertiaCholeskyInertialParameter,
    RawInertialParameter,
    VectorInertialParameter,
    get_plant_inertial_params,
)
from anzu.drake_torch_sys_id import (
    DynamicsModel,
    DynamicsModelTrajectoryLoss,
)
from anzu.test.drake_torch_dynamics_test import (
    add_model_acrobot,
    add_model_panda,
    add_model_panda_j7,
    ensure_zero_grad,
    to_torch,
    torch_uniform,
)

VISUALIZE = False
DEFAULT_INERTIA_CLS = LogCholeskyInertialParameter


def copy_parameters(*, dest, src):
    dest_params = dict(dest.named_parameters())
    src_params = dict(src.named_parameters())
    items_iter = dict_items_zip(dest_params, src_params)
    for _, (dest_param, src_param) in items_iter:
        dest_param.data[:] = src_param.data


def perturb_inertial_params(inertial_params, *, perturb_scale):
    masses, coms, rot_inertias = inertial_params()
    rot_inertias_cm = parallel_axis_theorem(rot_inertias, masses, coms)
    # TODO(eric.cousineau): Shift CoM?
    N = masses.shape[0]
    mass_scale = 1.0 + perturb_scale * torch_uniform(N)
    masses *= mass_scale
    # TODO(eric.cousineau): Should rotate inertia as well? Add random
    # point-mass noise per Lee et al?
    rot_inertias_cm *= mass_scale.reshape((-1, 1, 1))
    rot_inertias = parallel_axis_theorem(
        rot_inertias_cm,
        masses,
        coms,
        Ba_is_Bcm=False,
    )
    # Reconstruct and remap parameters.
    inertial_param_cls = type(inertial_params)
    perturbed = inertial_param_cls(masses, coms, rot_inertias)
    copy_parameters(dest=inertial_params, src=perturbed)


def param_in(p_check, params):
    # Possible bug in PyTorch. See testing below.
    ids = [id(p) for p in params]
    return id(p_check) in ids


def make_dyn_model(add_model, inertial_param_cls=DEFAULT_INERTIA_CLS):
    plant = MultibodyPlant(time_step=0.0)
    _, bodies = add_model(plant)
    plant.Finalize()
    dyn_model = DynamicsModel.from_plant(
        plant, bodies, inertial_param_cls=inertial_param_cls
    )
    masses_gt, coms_gt, rot_inertias_gt = dyn_model.inertial_params()
    inertial_params_dist_gt = InertialEntropicDivergence(
        masses_gt, coms_gt, rot_inertias_gt
    )

    @torch.no_grad()
    def calc_mean_inertial_params_dist():
        masses, coms, rot_inertias = dyn_model.inertial_params()
        return inertial_params_dist_gt(masses, coms, rot_inertias).mean()

    return dyn_model, calc_mean_inertial_params_dist


class Test(unittest.TestCase):
    def test_param_in(self):
        param = nn.Parameter(torch.zeros(2))
        others = [nn.Parameter(torch.zeros(3, 4))]
        with self.assertRaises(RuntimeError) as cm:
            param in others
        # Possible PyTorch bug?
        # https://github.com/pytorch/pytorch/issues/97823
        self.assertIn("non-singleton dimension", str(cm.exception))
        self.assertFalse(param_in(param, others))
        self.assertTrue(param_in(others[0], others))

    def make_dyn_model(self, add_model):
        plant = MultibodyPlant(time_step=0.0)
        _, bodies = add_model(plant)
        plant.Finalize()
        dyn_model = DynamicsModel.from_plant(plant, bodies)
        masses_gt, coms_gt, rot_inertias_gt = dyn_model.inertial_params()
        inertial_params_dist_gt = InertialEntropicDivergence(
            masses_gt, coms_gt, rot_inertias_gt
        )

        @torch.no_grad()
        def calc_mean_inertial_params_dist():
            masses, coms, rot_inertias = dyn_model.inertial_params()
            return inertial_params_dist_gt(masses, coms, rot_inertias).mean()

        return dyn_model, calc_mean_inertial_params_dist

    @torch.no_grad()
    def test_reparameterize(self):
        dyn_model, _ = make_dyn_model(add_model_panda)
        self.assertIsInstance(dyn_model.inertial_params, DEFAULT_INERTIA_CLS)
        new_cls = PseudoInertiaCholeskyInertialParameter
        self.assertIsNot(new_cls, DEFAULT_INERTIA_CLS)
        dyn_model.reparameterize_inertial(new_cls)
        self.assertIsInstance(dyn_model.inertial_params, new_cls)

    @torch.no_grad()
    def test_perturb_inertial_params(self):
        dyn_model, calc_mean_inertial_params_dist = self.make_dyn_model(
            add_model_panda
        )
        # Brief sanity checks.
        self.assertLess(calc_mean_inertial_params_dist(), 1e-7)
        # Ensure zero scale perturbation works as expected.
        perturb_inertial_params(dyn_model.inertial_params, perturb_scale=0.0)
        self.assertLess(calc_mean_inertial_params_dist(), 1e-6)

    @torch.no_grad()
    def test_perturb_inertial_params(self):
        dyn_model, calc_mean_inertial_params_dist = make_dyn_model(
            add_model_panda
        )
        # Brief sanity checks.
        self.assertLess(calc_mean_inertial_params_dist(), 1e-6)
        # Ensure zero scale perturbation works as expected.
        perturb_inertial_params(dyn_model.inertial_params, perturb_scale=0.0)
        self.assertLess(calc_mean_inertial_params_dist(), 1e-6)

    @torch.no_grad()
    def check_sys_id(
        self,
        add_model,
        *,
        loss_perturb_min,
        lr,
        inertial_param_cls=DEFAULT_INERTIA_CLS,
        max_grad_tol=1e-6,
    ):
        dyn_model, calc_mean_inertial_params_dist = make_dyn_model(
            add_model,
            inertial_param_cls=inertial_param_cls,
        )
        gamma = 1e-2
        dyn_model_loss = DynamicsModelTrajectoryLoss(
            model=dyn_model,
            gamma=gamma,
        )

        torch.manual_seed(0)

        num_q = dyn_model.inverse_dynamics.num_velocities()
        N = 100
        dim_q = (N, num_q)
        q = 1.0 * torch_uniform(dim_q)
        v = 3.0 * torch_uniform(dim_q)
        vd = 5.0 * torch_uniform(dim_q)
        tau_gt = dyn_model(q, v, vd)

        with torch.set_grad_enabled(True):
            ensure_zero_grad(dyn_model_loss.parameters())
            loss, _ = dyn_model_loss(q, v, vd, tau_gt)
            np.testing.assert_allclose(
                loss.detach().numpy(), 0.0, atol=1.0e-6, rtol=0.0
            )
            loss.backward()
            # Ensure we have near zero gradients everywhere.
            for param in dyn_model.parameters():
                self.assertLess(param.grad.abs().max(), max_grad_tol)

        # Perturb all model parameters a small amount.
        perturb_scale = 0.5
        perturb_inertial_params(
            dyn_model.inertial_params, perturb_scale=perturb_scale
        )
        inertial_params_list = list(dyn_model.inertial_params.parameters())
        for param in dyn_model.parameters():
            if param_in(param, inertial_params_list):
                continue
            param.data += perturb_scale * torch_uniform(param.shape)

        # Show that our loss is now non-zero.
        loss_perturb, _ = dyn_model_loss(q, v, vd, tau_gt)
        self.assertGreater(loss_perturb, loss_perturb_min)

        if lr is None:
            return

        # Reconstruct loss so our regularizer is reinitialized.
        dyn_model_loss = DynamicsModelTrajectoryLoss(
            model=dyn_model,
            gamma=gamma,
        )

        # Add some slight (arbitrary) noise.
        q += 0.001 * torch_uniform(dim_q)
        v += 0.01 * torch_uniform(dim_q)
        vd += 0.05 * torch_uniform(dim_q)

        # Show that we can decrease from here.
        num_epochs = 20
        opt = torch.optim.Adam(dyn_model.parameters(), lr=lr)
        losses = []
        loss_dicts = []
        dists = []

        with torch.set_grad_enabled(True):
            for i in tqdm(range(num_epochs)):
                # Record loss.
                loss, loss_dict = dyn_model_loss(q, v, vd, tau_gt)
                losses.append(loss.detach().item())
                loss_dicts.append(loss_dict)
                # Record distance from ground-truth parameters.
                dist = calc_mean_inertial_params_dist().item()
                dists.append(dist)
                # Optimize.
                loss.backward()
                opt.step()
                opt.zero_grad()
        # Analyze basic trends.
        final_loss, final_loss_dict = dyn_model_loss(q, v, vd, tau_gt)
        losses.append(final_loss.item())
        loss_dicts.append(final_loss_dict)
        final_dist = calc_mean_inertial_params_dist()
        dists.append(final_dist.item())

        if VISUALIZE:
            _, axs = plt.subplots(nrows=2)

            plt.sca(axs[0])
            loss_keys = list(final_loss_dict.keys())
            plt.plot(losses, linewidth=3)
            for key in loss_keys:
                plt.plot([loss_dict[key] for loss_dict in loss_dicts])
            plt.legend(["sum"] + loss_keys)
            plt.ylabel("Loss")

            plt.sca(axs[1])
            plt.plot(dists)
            plt.ylabel("Entropic Divergence")
            plt.xlabel("Epoch")
            plt.show()

        first_loss = losses[0]
        # N.B. Choice of 0.5 is relatively arbitrary for now.
        # Our loss should decrease.
        self.assertLess(final_loss, 0.5 * first_loss)
        # TODO(eric.cousineau): See if there's some way we can intelligently
        # reason about ground truth, such as getting the proper null-space of
        # what is identifiable and check in that space (aside from checking
        # inverse dynamics itself).

    def test_bad_inertial_parameterizations(self):
        bad_cls_list = [
            RawInertialParameter,
            VectorInertialParameter,
        ]
        for bad_cls in bad_cls_list:
            with self.assertRaises(RuntimeError) as cm:
                self.check_sys_id(
                    add_model_panda_j7,
                    loss_perturb_min=0.002,
                    lr=1e-2,
                    inertial_param_cls=bad_cls,
                    max_grad_tol=1e-4,
                )
            self.assertIn("PhysicallyValid", str(cm.exception))

    def test_1_panda_j7(self):
        self.check_sys_id(add_model_panda_j7, loss_perturb_min=0.002, lr=1e-2)

    def test_2_acrobot(self):
        self.check_sys_id(add_model_acrobot, loss_perturb_min=9.5, lr=1e-2)

    def run_panda(self, inertial_param_cls):
        self.check_sys_id(
            add_model_panda,
            loss_perturb_min=0.6,
            lr=5e-3,
            inertial_param_cls=inertial_param_cls,
        )

    def test_3_panda(self):
        self.assertIs(DEFAULT_INERTIA_CLS, LogCholeskyInertialParameter)
        self.run_panda(LogCholeskyInertialParameter)

    def test_3_panda_other_parameterizations(self):
        self.run_panda(LogCholeskyLinAlgInertialParameter)
        self.run_panda(LogCholeskyComInertialParameter)
        # N.B. These parameterizations seems to evolve with a similar loss
        # landscape, but the entropic divergence evolution seems more "choppy".
        self.run_panda(PseudoInertiaCholeskyInertialParameter)


if __name__ == "__main__":
    unittest.main()
