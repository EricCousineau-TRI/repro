"""
Rewrite of Anzu C++ code in Python.

    std::vector<GeometryId>
    GetGeometries(
        const MultibodyPlant<double>& plant,
        const SceneGraph<double>& scene_graph,
        std::vector<const Body<double>*> bodies) {
      std::vector<GeometryId> geometry_ids;
      auto& inspector = scene_graph.model_inspector();
      for (auto* body : bodies) {
        auto frame_id = plant.GetBodyFrameIdOrThrow(body->index());
        auto body_geometry_ids = inspector.GetGeometries(frame_id);
        geometry_ids.insert(
            geometry_ids.end(), body_geometry_ids.begin(),
            body_geometry_ids.end());
      }
      // N.B. `inspector.GetGeometries` returns the ids in a consistent (sorted)
      // order, but we re-sort here just in case the geometries have been mutated.
      std::sort(geometry_ids.begin(), geometry_ids.end());
      return geometry_ids;
    }

    void RemoveRoleFromGeometries(
        const MultibodyPlant<double>& plant,
        SceneGraph<double>* scene_graph,
        const Role role,
        std::vector<const Body<double>*> bodies) {
      for (auto geometry_id : GetGeometries(plant, *scene_graph, bodies)) {
        scene_graph->RemoveRole(*plant.get_source_id(), geometry_id, role);
      }
    }
"""

def GetGeometries(plant, scene_graph, bodies):
    geometry_ids = []
    inspector = scene_graph.model_inspector()
    for geometry_id in inspector.GetAllGeometryIds():
        body = plant.GetBodyFromFrameId(inspector.GetFrameId(geometry_id))
        if body in bodies:
            geometry_ids.append(geometry_id)
    geometry_ids.sort()
    return geometry_ids


def RemoveRoleFromGeometries(plant, scene_graph, role, bodies):
    for geometry_id in GetGeometries(plant, scene_graph, bodies):
        scene_graph.RemoveRole(plant.get_source_id(), geometry_id, role)
