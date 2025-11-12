import argparse
import os
from pathlib import Path

from pydrake.geometry import StartMeshcat
from pydrake.multibody.parsing import (
    LoadModelDirectives,
    ProcessModelDirectives,
)
from pydrake.planning import RobotDiagramBuilder
from pydrake.visualization import (
    ApplyVisualizationConfig,
    VisualizationConfig,
)


def discover_dmd_files(dmd_dir: str, scene_idx: int) -> list[Path]:
    """Discover all .dmd.yaml files in the directory and validate scene index.

    Args:
        dmd_dir: Directory containing .dmd.yaml files
        scene_idx: Starting scene index to validate

    Returns:
        List of Path objects for .dmd.yaml files, sorted alphabetically
    """
    dmd_path = Path(dmd_dir)
    if not dmd_path.exists():
        raise ValueError(f"Directory does not exist: {dmd_dir}")
    if not dmd_path.is_dir():
        raise ValueError(f"Path is not a directory: {dmd_dir}")

    # Find all .dmd.yaml files
    dmd_files = sorted(dmd_path.glob("*.dmd.yaml"))

    if len(dmd_files) == 0:
        raise ValueError(f"No .dmd.yaml files found in {dmd_dir}")

    print(f"Found {len(dmd_files)} .dmd.yaml files.")

    if scene_idx >= len(dmd_files) or scene_idx < 0:
        raise ValueError(
            f"Scene index {scene_idx} is out of range. "
            f"Directory contains {len(dmd_files)} files "
            f"(valid indices: 0-{len(dmd_files)-1})."
        )

    return dmd_files


def setup_package_paths(
    tri_package: str = None,
    gazebo_package: str = None,
    greg_package: str = None,
) -> dict[str, str]:
    """Setup package paths with auto-detection if not provided.

    Args:
        tri_package: Path to tri package directory (optional)
        gazebo_package: Path to gazebo package directory (optional)
        greg_package: Path to greg package directory (optional)

    Returns:
        Dictionary mapping package names to package paths
    """
    packages = {}

    # Try to auto-detect or use provided paths
    package_configs = [
        ("tri", tri_package, "models/tri/package.xml"),
        ("gazebo", gazebo_package, "models/gazebo/package.xml"),
        ("greg", greg_package, "models/greg/package.xml"),
    ]

    for package_name, provided_path, default_xml_path in package_configs:
        if provided_path:
            # Use provided path
            package_path = Path(provided_path)
            if package_path.exists():
                packages[package_name] = str(package_path)
                print(f"Using {package_name} package from: {package_path}")
            else:
                print(
                    f"Warning: Provided {package_name} package path "
                    f"does not exist: {provided_path}"
                )
        else:
            # Try auto-detection
            package_file_abs_path = os.path.abspath(
                os.path.expanduser(default_xml_path)
            )
            if os.path.exists(package_file_abs_path):
                package_dir = os.path.dirname(package_file_abs_path)
                packages[package_name] = package_dir
                print(f"Auto-detected {package_name} package at: {package_dir}")

    if not packages:
        print(
            "Warning: No packages registered. "
            "Make sure your .dmd.yaml files don't require package:// URIs."
        )

    return packages


def visualize_dmd_file(dmd_file: Path, meshcat_instance, packages: dict[str, str]):
    """Visualize a single .dmd.yaml file using Drake.

    Args:
        dmd_file: Path to the .dmd.yaml file
        meshcat_instance: Meshcat instance to use for visualization
        packages: Dictionary of package names to paths

    Returns:
        The built diagram (must be kept in scope for visualization to persist)
    """
    # Clear previous scene
    meshcat_instance.Delete()

    # Create robot diagram builder (includes plant, scene_graph, and parser)
    builder = RobotDiagramBuilder()

    # Register package paths
    for package_name, package_path in packages.items():
        builder.parser().package_map().Add(package_name, package_path)

    # Load model directives
    directives = LoadModelDirectives(str(dmd_file))

    # Process directives into the plant
    ProcessModelDirectives(directives, builder.parser())

    # Finalize the plant
    builder.plant().Finalize()

    # Configure and apply visualization
    ApplyVisualizationConfig(
        config=VisualizationConfig(),
        plant=builder.plant(),
        scene_graph=builder.scene_graph(),
        builder=builder.builder(),
        meshcat=meshcat_instance,
    )

    # Build the diagram
    diagram = builder.Build()

    # Create context and publish to visualizer
    context = diagram.CreateDefaultContext()
    diagram.ForcedPublish(context)

    return diagram


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize Drake model directive (.dmd.yaml) files "
            "with keyboard navigation"
        )
    )
    parser.add_argument(
        "dmd_dir", type=str, help="Directory containing .dmd.yaml files"
    )
    parser.add_argument(
        "--scene-idx",
        type=int,
        help="Index of the starting scene to visualize",
        default=0,
    )
    parser.add_argument(
        "--tri-package",
        type=str,
        default=None,
        help="Path to tri package directory (auto-detected if not provided)",
    )
    parser.add_argument(
        "--gazebo-package",
        type=str,
        default=None,
        help=(
            "Path to gazebo package directory "
            "(auto-detected if not provided)"
        ),
    )
    parser.add_argument(
        "--greg-package",
        type=str,
        default=None,
        help=(
            "Path to greg package directory " "(auto-detected if not provided)"
        ),
    )
    args = parser.parse_args()

    # Create Meshcat instance
    meshcat_instance = StartMeshcat()

    # Discover all .dmd.yaml files
    dmd_files = discover_dmd_files(args.dmd_dir, args.scene_idx)
    total_scenes = len(dmd_files)

    # Setup package paths
    packages = setup_package_paths(
        tri_package=args.tri_package,
        gazebo_package=args.gazebo_package,
        greg_package=args.greg_package,
    )

    # Start at the specified scene index
    current_scene = args.scene_idx

    # Visualize the starting scene
    scene_name = dmd_files[current_scene].name
    print(f"\nViewing scene {current_scene+1}/{total_scenes}: {scene_name}")
    print(
        "Instructions: Enter 'n' for next scene, "
        "'p' for previous scene, 'q' to quit"
    )
    _ = visualize_dmd_file(
        dmd_files[current_scene], meshcat_instance, packages
    )

    # Interactive loop for viewing scenes
    try:
        while True:
            prompt = f"Scene {current_scene+1}/{total_scenes} > "
            cmd = input(prompt).strip().lower()
            if cmd == "n" or cmd == "next":
                current_scene = (current_scene + 1) % total_scenes
                scene_name = dmd_files[current_scene].name
                print(
                    f"Viewing scene {current_scene+1}/{total_scenes}: "
                    f"{scene_name}"
                )
                _ = visualize_dmd_file(
                    dmd_files[current_scene], meshcat_instance, packages
                )
            elif cmd == "p" or cmd == "prev":
                current_scene = (current_scene - 1) % total_scenes
                scene_name = dmd_files[current_scene].name
                print(
                    f"Viewing scene {current_scene+1}/{total_scenes}: "
                    f"{scene_name}"
                )
                _ = visualize_dmd_file(
                    dmd_files[current_scene], meshcat_instance, packages
                )
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
