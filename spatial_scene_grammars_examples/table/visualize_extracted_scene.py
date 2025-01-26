import argparse
import os
import pickle

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    MeshcatVisualizer,
    Parser,
    RigidTransform,
    Simulator,
    StartMeshcat,
)


def load_scene(dataset_path: str, scene_idx: int):
    """Load a specific scene from the dataset."""
    scenes = []
    with open(dataset_path, "rb") as f:
        while True:
            try:
                scene = pickle.load(f)
                scenes.append(scene)
            except EOFError:
                break

    if scene_idx >= len(scenes):
        raise ValueError(
            f"Scene index {scene_idx} is out of range. Dataset contains {len(scenes)} scenes."
        )

    return scenes[scene_idx]


def visualize_scene(scene):
    """Visualize the scene using Meshcat."""
    # Create meshcat instance
    meshcat = StartMeshcat()

    # Setup plant
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    parser.SetAutoRenaming(True)

    # Add Greg Table package
    package_file_abs_path = os.path.abspath(
        os.path.expanduser("greg_table/package.xml")
    )
    parser.package_map().Add("greg_table", os.path.dirname(package_file_abs_path))

    # Add static models (table)
    parser.AddModelsFromUrl("package://greg_table/models/misc/cafe_table/model.sdf")
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetBodyByName("cafe_table_body").body_frame(),
        RigidTransform(p=[0.0, 0.0, 0.0]),
    )

    # Add scene models
    for obj in scene:
        model_path = obj["model_path"]
        transform = obj["transform"]

        model = parser.AddModelsFromUrl(model_path)
        assert len(model) == 1
        model = model[0]

        # Set scene model transforms
        body_indices = plant.GetBodyIndices(model)
        for body_index in body_indices:
            body = plant.get_body(body_index)
            plant.WeldFrames(
                plant.world_frame(),
                body.body_frame(),
                RigidTransform(transform),
            )

    plant.Finalize()

    # Add visualizer
    _ = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    diagram = builder.Build()

    # Simulate
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(20.0)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a scene from the generated dataset"
    )
    parser.add_argument(
        "dataset_path", type=str, help="Path to the pickle dataset file"
    )
    parser.add_argument("scene_idx", type=int, help="Index of the scene to visualize")
    args = parser.parse_args()

    # Load and visualize the scene
    scene = load_scene(args.dataset_path, args.scene_idx)
    visualize_scene(scene)


if __name__ == "__main__":
    main()
