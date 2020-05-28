from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Box,
    ConnectDrakeVisualizer,
    CoulombFriction,
    DiagramBuilder,
    RigidTransform,
    Simulator,
    SpatialInertia,
    UnitInertia,
)


def add_body(plant, name):
    # Returns a random body, with an incrementing name.
    inertia = SpatialInertia(
        mass=1.,
        p_PScm_E=[0, 0, 0],
        G_SP_E=UnitInertia(
            Ixx=1.,
            Iyy=1.,
            Izz=1.,
        ),
    )
    return plant.AddRigidBody(
        name=name,
        M_BBo_B=inertia,
    )


def add_geometry(plant, body):
    box = Box(
        width=0.2,
        depth=0.2,
        height=0.2,
    )
    plant.RegisterVisualGeometry(
        body=body,
        X_BG=RigidTransform(),
        shape=box,
        name=f"visual",
        diffuse_color=[1, 1, 1, 1],
    )
    static_friction = 1.
    plant.RegisterCollisionGeometry(
        body=body,
        X_BG=RigidTransform(),
        shape=box,
        name="collision",
        coulomb_friction=CoulombFriction(
            static_friction=1,
            dynamic_friction=1,
        )
    )


def main():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 1e-3)

    body = add_body(plant, "welded_body")
    add_geometry(plant, body)
    plant.WeldFrames(plant.world_frame(), body.body_frame())

    body = add_body(plant, "floating_body")
    add_geometry(plant, body)
    plant.SetDefaultFreeBodyPose(body, RigidTransform([1., 0, 1.]))

    plant.Finalize()
    ConnectDrakeVisualizer(builder, scene_graph)
    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.)
    context = simulator.get_context()
    dt = 0.1
    while context.get_time() < 10.:
        simulator.AdvanceTo(context.get_time() + dt)


assert __name__ == "__main__"
main()
