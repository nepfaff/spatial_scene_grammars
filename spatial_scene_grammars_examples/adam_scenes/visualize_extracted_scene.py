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


def load_scenes(dataset_path: str, scene_idx: int):
    """Load all scenes from the dataset and validate the starting scene index."""
    scenes = []
    with open(dataset_path, "rb") as f:
        while True:
            try:
                scene = pickle.load(f)
                scenes.append(scene)
            except EOFError:
                break

    print(f"Loaded {len(scenes)} scenes.")

    if scene_idx >= len(scenes) or scene_idx < 0:
        raise ValueError(
            f"Scene index {scene_idx} is out of range. Dataset contains "
            f"{len(scenes)} scenes (valid indices: 0-{len(scenes)-1})."
        )

    return scenes


def visualize_scene(scene, meshcat_instance):
    """Visualize the scene using Meshcat."""
    # Clear previous scene
    meshcat_instance.Delete()

    # Setup plant
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    parser.SetAutoRenaming(True)

    # Add Anzu package
    package_file_abs_path = os.path.abspath(
        os.path.expanduser("models/tri/package.xml")
    )
    parser.package_map().Add("tri", os.path.dirname(package_file_abs_path))
    # Add Gazebo package
    package_file_abs_path = os.path.abspath(
        os.path.expanduser("models/gazebo/package.xml")
    )
    parser.package_map().Add("gazebo", os.path.dirname(package_file_abs_path))
    # Add Greg package
    package_file_abs_path = os.path.abspath(
        os.path.expanduser("models/greg/package.xml")
    )
    parser.package_map().Add("greg", os.path.dirname(package_file_abs_path))

    # Add scene models
    for obj in scene:
        model_path = obj["model_path"]
        transform = obj["transform"]

        model = parser.AddModelsFromUrl(model_path)
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
    _ = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat_instance)

    diagram = builder.Build()

    # Simulate
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(0.1)  # Just enough to visualize

    return diagram, simulator


def main():
    parser = argparse.ArgumentParser(
        description="Visualize scenes from the generated dataset with keyboard navigation"
    )
    parser.add_argument(
        "dataset_path", type=str, help="Path to the pickle dataset file"
    )
    parser.add_argument(
        "scene_idx", type=int, help="Index of the starting scene to visualize"
    )
    args = parser.parse_args()

    # Create meshcat instance
    meshcat_instance = StartMeshcat()

    # Load all scenes from dataset
    scenes = load_scenes(args.dataset_path, args.scene_idx)
    total_scenes = len(scenes)

    # Start at the specified scene index
    current_scene = args.scene_idx

    # Visualize the starting scene
    print(f"\nViewing scene {current_scene+1}/{total_scenes}.")
    print("Instructions: Enter 'n' for next scene, 'p' for previous scene, 'q' to quit")
    diagram, simulator = visualize_scene(scenes[current_scene], meshcat_instance)

    # Interactive loop for viewing scenes
    try:
        while True:
            cmd = input(f"Scene {current_scene+1}/{total_scenes} > ").strip().lower()
            if cmd == "n" or cmd == "next":
                current_scene = (current_scene + 1) % total_scenes
                diagram, simulator = visualize_scene(
                    scenes[current_scene], meshcat_instance
                )
                print(f"Viewing scene {current_scene+1}/{total_scenes}")
            elif cmd == "p" or cmd == "prev":
                current_scene = (current_scene - 1) % total_scenes
                diagram, simulator = visualize_scene(
                    scenes[current_scene], meshcat_instance
                )
                print(f"Viewing scene {current_scene+1}/{total_scenes}")
            elif cmd == "q" or cmd == "quit":
                break
            else:
                print(
                    "Unknown command. Use 'n' for next, 'p' for previous, 'q' to quit"
                )
    except KeyboardInterrupt:
        print("\nExiting scene viewer.")


if __name__ == "__main__":
    main()
