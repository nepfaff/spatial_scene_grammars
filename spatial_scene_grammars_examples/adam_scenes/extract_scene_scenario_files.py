import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    Parser,
)
from pydrake.multibody.parsing import (
    LoadModelDirectives,
    ProcessModelDirectives,
)


def extract_body_name_from_model(model_path: str) -> str:
    """Extract the body name from a model file.

    Args:
        model_path: URI to the model file (e.g., package://tri/model.sdf)

    Returns:
        Name of the first body in the model
    """
    # Create a temporary plant to load the model
    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    parser.SetAutoRenaming(True)

    # Add package paths (same as visualize_extracted_scene.py)
    package_file_abs_path = os.path.abspath(
        os.path.expanduser("models/tri/package.xml")
    )
    if os.path.exists(package_file_abs_path):
        parser.package_map().Add("tri", os.path.dirname(package_file_abs_path))

    package_file_abs_path = os.path.abspath(
        os.path.expanduser("models/gazebo/package.xml")
    )
    if os.path.exists(package_file_abs_path):
        parser.package_map().Add("gazebo", os.path.dirname(package_file_abs_path))

    package_file_abs_path = os.path.abspath(
        os.path.expanduser("models/greg/package.xml")
    )
    if os.path.exists(package_file_abs_path):
        parser.package_map().Add("greg", os.path.dirname(package_file_abs_path))

    # Load model and extract body name
    try:
        # Check if this is a model directives file (.dmd.yaml)
        if model_path.endswith('.dmd.yaml'):
            # Load and process model directives
            directives = LoadModelDirectives(model_path)
            ProcessModelDirectives(directives, parser)
            plant.Finalize()

            # Get all non-world bodies
            world_instance = plant.GetModelInstanceByName("world")
            body_indices = [
                idx
                for idx in plant.GetBodyIndices(world_instance)
                if plant.get_body(idx).name() != "world"
            ]

            # If no bodies found in world, check all model instances
            if len(body_indices) == 0:
                for model_idx in range(plant.num_model_instances()):
                    model_instance = plant.get_model_instance(model_idx)
                    body_indices = plant.GetBodyIndices(model_instance)
                    if len(body_indices) > 0:
                        break
        else:
            # Single model file (URDF/SDF)
            model_instance = parser.AddModelsFromUrl(model_path)[0]
            body_indices = plant.GetBodyIndices(model_instance)

        if len(body_indices) == 0:
            raise ValueError(f"No bodies found in model {model_path}")

        # Get the first body name (sufficient per user requirements)
        first_body = plant.get_body(body_indices[0])
        return first_body.name()
    except Exception as e:
        raise RuntimeError(
            f"Failed to extract body name from {model_path}: {e}"
        )


def convert_transform_to_pose(transform: np.ndarray) -> tuple[list[float], list[float]]:
    """Convert a 4x4 homogeneous transform to translation and RPY in degrees.

    Args:
        transform: 4x4 numpy array representing homogeneous transformation

    Returns:
        Tuple of (translation [x, y, z], rpy_degrees [roll, pitch, yaw])

    Note:
        Drake model directives require rotations in degrees. Full precision is
        maintained by using 17 significant digits, which is just as accurate
        as using radians.
    """
    # Extract translation (meters) with full precision
    translation = transform[:3, 3].tolist()

    # Extract rotation matrix and convert to RPY in degrees (required by Drake)
    rotation_matrix = transform[:3, :3]
    scipy_rotation = R.from_matrix(rotation_matrix)
    rpy_deg = scipy_rotation.as_euler('xyz', degrees=True).tolist()

    return translation, rpy_deg


def generate_scene_yaml(scene: list[dict], scene_idx: int) -> str:
    """Generate Drake scenario YAML content for a single scene.

    Args:
        scene: List of objects, each with 'model_path' and 'transform' keys
        scene_idx: Index of the scene (for error messages)

    Returns:
        YAML content as a string
    """
    yaml_lines = ["directives:"]

    for obj_idx, obj in enumerate(scene):
        model_path = obj["model_path"]
        transform = obj["transform"]

        # Handle .dmd.yaml files - use add_directives instead of add_model
        if model_path.endswith('.dmd.yaml'):
            # Check if transform is identity
            translation = transform[:3, 3]
            rotation_matrix = transform[:3, :3]
            is_identity_translation = np.allclose(translation, 0.0, atol=1e-6)
            is_identity_rotation = np.allclose(
                rotation_matrix, np.eye(3), atol=1e-6
            )

            if is_identity_translation and is_identity_rotation:
                # Use add_directives for identity transform
                yaml_lines.append(f"- add_directives:")
                yaml_lines.append(f"    file: {model_path}")
            else:
                # Can't handle non-identity transforms for .dmd.yaml files
                print(
                    f"Warning: Scene {scene_idx}, object {obj_idx}: "
                    f"Skipping .dmd.yaml with non-identity transform "
                    f"(not supported): {model_path}"
                )
            continue

        # Extract body name from model
        try:
            body_name = extract_body_name_from_model(model_path)
        except Exception as e:
            print(f"Warning: Scene {scene_idx}, object {obj_idx}: {e}")
            print(f"  Skipping object with model_path: {model_path}")
            continue

        # Convert transform to translation and RPY degrees
        translation, rpy_deg = convert_transform_to_pose(transform)

        # Generate unique model name
        model_name = f"object_{obj_idx}"

        # Format values with full precision (17 significant digits)
        trans_str = (
            f"[{translation[0]:.17g}, "
            f"{translation[1]:.17g}, "
            f"{translation[2]:.17g}]"
        )
        rpy_str = (
            f"[{rpy_deg[0]:.17g}, " f"{rpy_deg[1]:.17g}, " f"{rpy_deg[2]:.17g}]"
        )

        # Add model directive (without pose - pose goes in weld)
        yaml_lines.append(f"- add_model:")
        yaml_lines.append(f"    name: {model_name}")
        yaml_lines.append(f"    file: {model_path}")

        # Add weld directive with X_PC transform to fix object at pose
        yaml_lines.append(f"- add_weld:")
        yaml_lines.append(f"    parent: world")
        yaml_lines.append(f"    child: {model_name}::{body_name}")
        yaml_lines.append(f"    X_PC:")
        yaml_lines.append(f"      translation: {trans_str}")
        yaml_lines.append(f"      rotation: !Rpy {{ deg: {rpy_str} }}")

    return "\n".join(yaml_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Extract Drake scenario files from scene pickle dataset"
    )
    parser.add_argument(
        "input_pickle",
        type=str,
        help="Path to the input pickle file containing scenes",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to write the .dmd.yaml scenario files",
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load all scenes from pickle file
    print(f"Loading scenes from {args.input_pickle}...")
    scenes = []
    with open(args.input_pickle, "rb") as f:
        while True:
            try:
                scene = pickle.load(f)
                scenes.append(scene)
            except EOFError:
                break

    print(f"Loaded {len(scenes)} scenes.")

    # Process each scene
    for scene_idx, scene in enumerate(scenes):
        print(f"Processing scene {scene_idx + 1}/{len(scenes)}...")

        # Generate YAML content
        yaml_content = generate_scene_yaml(scene, scene_idx)

        # Write to file with zero-padded naming
        output_file = output_path / f"scene_{scene_idx:03d}.dmd.yaml"
        with open(output_file, "w") as f:
            f.write(yaml_content)

        print(f"  Wrote {output_file}")

    print(f"\nSuccessfully converted {len(scenes)} scenes to {output_path}")


if __name__ == "__main__":
    main()
