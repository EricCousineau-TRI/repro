import hashlib
import unittest

import numpy as np
import torch

from pydrake.multibody.parsing import LoadModelDirectives, Parser
from pydrake.multibody.plant import MultibodyPlant

from anzu.not_exported_soz.cc import ProcessAnzuModelDirectives
from anzu.not_exported_soz.dair_pll_inertia import parallel_axis_theorem
from anzu.drake_torch_dynamics import (
    DrakeInverseDynamics,
    InertialEntropicDivergence,
    LogCholeskyComInertialParameter,
    LogCholeskyInertialParameter,
    LogCholeskyLinAlgInertialParameter,
    MassAndComInertialParameter,
    PseudoInertiaCholeskyInertialParameter,
    RawInertialParameter,
    UnitLogCholeskyInertialParameter,
    UnitRotationalInertialParameter,
    VectorInertialParameter,
    cov_matrix_to_rot_inertia,
    from_triangular_number,
    get_candidate_sys_id_bodies,
    get_plant_inertial_params,
    inertial_param_to_pseudo_inertia,
    log_cholesky_vector_to_lower_triangle,
    log_cholesky_vector_to_matrix,
    lower_triangle_to_log_cholesky_vector,
    matrix_to_log_cholesky_vector,
    pseudo_inertia_to_inertial_param,
    rot_inertia_to_cov_matrix,
    to_triangular_number,
)
from not_exported_soz.path_util import resolve_path


def add_model_panda_j7(plant):
    model_file = resolve_path("package://anzu/models/haptic/panda_j7.urdf")
    model = Parser(plant).AddModelFromFile(model_file)
    # Posture the link with rotary axis aligned with gravity.
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("panda_joint7_inboard"),
    )
    bodies = get_candidate_sys_id_bodies(plant)
    return model, bodies


def add_model_acrobot(plant):
    model_file = resolve_path(
        "package://drake/multibody/benchmarks/acrobot/acrobot.sdf"
    )
    model = Parser(plant).AddModelFromFile(model_file)
    bodies = get_candidate_sys_id_bodies(plant)
    return model, bodies


def add_model_panda(plant):
    model_file = resolve_path(
        "package://anzu/models/haptic/add_haptic_test_panda.yaml"
    )
    directives = LoadModelDirectives(model_file)
    ProcessAnzuModelDirectives(directives, plant)
    model = plant.GetModelInstanceByName("right::panda")
    bodies = get_candidate_sys_id_bodies(plant)
    return model, bodies


def ensure_zero_grad(params):
    for param in params:
        param.grad = None


def to_torch(x):
    return torch.from_numpy(x).to(torch.float32)


def numpy_hash(x):
    # https://stackoverflow.com/a/16592241/7829525, answer from Jann Poppinga.
    return hashlib.sha256(x.data).hexdigest()


def torch_hash(x):
    return numpy_hash(x.numpy())


def num_unique_tensors(x):
    hashes = [torch_hash(x_i) for x_i in x]
    return len(set(hashes))


def torch_uniform(shape, low=-1.0, high=1.0):
    y = torch.zeros(shape)
    y.uniform_(low, high)
    return y


def assert_positive_definite(test, Ms):
    eigs = torch.linalg.eigvals(Ms)
    test.assertTrue((eigs.imag == 0.0).all())
    test.assertTrue((eigs.real > 0).all())


class Test(unittest.TestCase):
    def make_inertial_params(self):
        plant = MultibodyPlant(time_step=0.0)
        _, bodies = add_model_panda(plant)
        plant.Finalize()
        context = plant.CreateDefaultContext()
        masses, coms, rot_inertias = map(
            to_torch, get_plant_inertial_params(plant, context, bodies)
        )
        return masses, coms, rot_inertias

    def test_tri_sum(self):
        for N in range(10):
            count = to_triangular_number(N)
            self.assertEqual(from_triangular_number(count), N)
        with self.assertRaises(AssertionError):
            from_triangular_number(2)

    @torch.no_grad()
    def test_cov_matrix(self):
        masses, coms, rot_inertias = self.make_inertial_params()
        assert_positive_definite(self, rot_inertias)
        cov_matrix = rot_inertia_to_cov_matrix(rot_inertias)
        assert_positive_definite(self, cov_matrix)
        rot_inertias_rt = cov_matrix_to_rot_inertia(cov_matrix)
        torch.testing.assert_close(rot_inertias, rot_inertias_rt)

    @torch.no_grad()
    def test_pseudo_inertia(self):
        masses, coms, rot_inertias = self.make_inertial_params()
        # Ensure computed pseudo inertias are positive definite.
        pseudo_inertias = inertial_param_to_pseudo_inertia(
            masses, coms, rot_inertias
        )
        assert_positive_definite(self, pseudo_inertias)
        # Ensure we can round-trip.
        masses_rt, coms_rt, rot_inertias_rt = pseudo_inertia_to_inertial_param(
            pseudo_inertias
        )
        torch.testing.assert_close(masses, masses_rt)
        torch.testing.assert_close(coms, coms_rt)
        torch.testing.assert_close(rot_inertias, rot_inertias_rt)

    def check_log_cholesky(self, M):
        # For debugging, ensure we can recover lower triangle itself.
        L = torch.linalg.cholesky(M)
        log_vec = lower_triangle_to_log_cholesky_vector(L)
        L_rt = log_cholesky_vector_to_lower_triangle(log_vec)
        torch.testing.assert_close(L, L_rt)
        # Then ensure we can recover with full API.
        log_vec = matrix_to_log_cholesky_vector(M)
        M_rt = log_cholesky_vector_to_matrix(log_vec)
        torch.testing.assert_close(M, M_rt)

    @torch.no_grad()
    def test_log_cholesky(self):
        masses, coms, rot_inertias = self.make_inertial_params()
        pseudo_inertias = inertial_param_to_pseudo_inertia(
            masses, coms, rot_inertias
        )
        self.check_log_cholesky(rot_inertias)
        self.check_log_cholesky(pseudo_inertias)

    @torch.no_grad()
    def test_log_cholesky_dair(self):
        masses, coms, rot_inertias = self.make_inertial_params()
        pseudo_inertias = inertial_param_to_pseudo_inertia(
            masses, coms, rot_inertias
        )
        thetas = matrix_to_log_cholesky_vector(pseudo_inertias)
        dair_param = LogCholeskyInertialParameter(masses, coms, rot_inertias)
        # We need to slightly permute the dair_pll ordering. (alpha, d, s, t)
        # are symbols uesd in [3].
        # fmt: off
        inidces_dair_to_other = [
            0,        # alpha
            1, 2, 3,  # d
            4, 6, 5,  # s - note swap of (5, 6)
            7, 8, 9,  # t
        ]
        # fmt: on
        thetas_dair = dair_param.theta[..., inidces_dair_to_other]
        torch.testing.assert_close(thetas, thetas_dair, atol=1e-5, rtol=0.0)

    @torch.no_grad()
    def test_inertial_parameterizations(self):
        """
        Ensure inertial parmaterizations are invertible at least at initial
        guess. Checks via round-trip ("rt").
        """
        masses, coms, rot_inertias = self.make_inertial_params()

        def check_inertial_param(inertial_params):
            masses_rt, coms_rt, rot_inertias_rt = inertial_params()
            torch.testing.assert_close(masses, masses_rt)
            torch.testing.assert_close(coms, coms_rt)
            torch.testing.assert_close(rot_inertias, rot_inertias_rt)

        cls_list = [
            RawInertialParameter,
            VectorInertialParameter,
            LogCholeskyInertialParameter,
            LogCholeskyLinAlgInertialParameter,
            PseudoInertiaCholeskyInertialParameter,
            LogCholeskyComInertialParameter,
            MassAndComInertialParameter,
            UnitRotationalInertialParameter,
            UnitLogCholeskyInertialParameter,
        ]
        for cls in cls_list:
            with self.subTest(cls=cls):
                inertial_params = cls(masses, coms, rot_inertias)
                check_inertial_param(inertial_params)

        # Ensure that changing representation does not change values.
        inertial_params_1 = LogCholeskyInertialParameter(
            masses, coms, rot_inertias
        )
        check_inertial_param(inertial_params_1)
        inertial_params_2 = MassAndComInertialParameter.from_other(
            inertial_params_1
        )
        check_inertial_param(inertial_params_2)

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

        inverse_dynamics = DrakeInverseDynamics(
            plant, bodies, inertial_param_cls=inertial_param_cls
        )
        num_q = plant.num_positions()
        # For now, simplify expectations.
        assert plant.num_velocities() == num_q
        dim_q = batch_shape + (num_q,)
        q0 = torch.zeros(dim_q)
        v0 = torch.zeros(dim_q)
        vd0 = torch.ones(dim_q)

        # Ensure we can backprop more than once.
        for i in range(2):
            ensure_zero_grad(inverse_dynamics.parameters())
            tau = inverse_dynamics(q0, v0, vd0)
            fake_loss = tau.sum()
            fake_loss.backward()

    def check_model(self, add_model):
        # Unbatched.
        self.check_forward_and_backward(add_model, RawInertialParameter)
        self.check_forward_and_backward(add_model, VectorInertialParameter)
        self.check_forward_and_backward(
            add_model, LogCholeskyInertialParameter
        )
        # Batched.
        self.check_forward_and_backward(
            add_model,
            RawInertialParameter,
            batch_shape=(3,),
        )

    def test_1_panda_j7(self):
        self.check_model(add_model_panda_j7)

    def test_2_acrobot(self):
        self.check_model(add_model_acrobot)

    def test_3_panda(self):
        self.check_model(add_model_panda)

    def test_numpy_hash(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(numpy_hash(a), numpy_hash(a.copy()))
        eps = np.finfo(np.float64).eps
        b = a.copy()
        b[0, 0] += eps
        self.assertNotEqual(numpy_hash(a), numpy_hash(b))

    @torch.no_grad()
    def test_inertial_entropic_divergence(self):
        plant = MultibodyPlant(time_step=0.0)
        _, bodies = add_model_panda(plant)
        plant.Finalize()
        context = plant.CreateDefaultContext()

        masses, coms, rot_inertias = map(
            to_torch, get_plant_inertial_params(plant, context, bodies)
        )
        # Show that we have (relatively) unique masses and inertias.
        self.assertEqual(num_unique_tensors(masses), 4)
        self.assertEqual(num_unique_tensors(coms), 7)
        self.assertEqual(num_unique_tensors(rot_inertias), 7)
        # Show that we have unique psuedo-inertias.
        pseudo_inertias = inertial_param_to_pseudo_inertia(
            masses, coms, rot_inertias
        )
        self.assertEqual(num_unique_tensors(pseudo_inertias), 7)

        entropic_divergence = InertialEntropicDivergence(
            masses, coms, rot_inertias
        )
        # Ensure we're at zero for same inertial.
        dist = entropic_divergence(masses, coms, rot_inertias)
        torch.testing.assert_close(dist, torch.zeros_like(dist))

        # Ensure that shifting pseudo inertia to CoM has nonzero distance from
        # non-CoM psuedo inertia.
        # N.B. This assumes all CoMs are nonzero.
        rot_inertias_cm = parallel_axis_theorem(rot_inertias, masses, coms)
        zeros = torch.zeros_like(coms)
        dist_cm = entropic_divergence(masses, zeros, rot_inertias_cm)
        # Threshold is specific to this test.
        self.assertTrue((dist_cm > 5.0).all())

        # Perturb by scaling mass and inertia.
        scale_1 = 0.1
        masses_1 = scale_1 * masses
        rot_inertias_1 = scale_1 * rot_inertias
        # Confirm that we still have unique pseudo inertias.
        pseudo_inertias_1 = inertial_param_to_pseudo_inertia(
            masses_1, coms, rot_inertias_1
        )
        self.assertEqual(num_unique_tensors(pseudo_inertias_1), 7)

        dist_scale_1 = entropic_divergence(masses_1, coms, rot_inertias_1)
        dist_scale_1 = torch.round(dist_scale_1, decimals=3)
        # N.B. This is specific to this test point.
        # TODO(eric.cousineau): I assume this scale invariance is a desired
        # property (same distance from scaling different pseudo inertias), but
        # should check.
        dist_1_expected = 5.6
        self.assertEqual(num_unique_tensors(dist_scale_1), 1)
        np.testing.assert_allclose(
            dist_scale_1.numpy(),
            dist_1_expected,
            atol=1e-1,
            rtol=0.0,
        )

        # Now decrease scale more and show that the distance is greater.
        scale_2 = scale_1**2
        masses_2 = scale_2 * masses
        rot_inertias_2 = scale_2 * rot_inertias
        dist_scale_2 = entropic_divergence(masses_2, coms, rot_inertias_2)
        dist_scale_2 = torch.round(dist_scale_2, decimals=4)
        self.assertEqual(num_unique_tensors(dist_scale_2), 1)
        self.assertGreater(dist_scale_2[0], dist_scale_1[0])


if __name__ == "__main__":
    unittest.main()
