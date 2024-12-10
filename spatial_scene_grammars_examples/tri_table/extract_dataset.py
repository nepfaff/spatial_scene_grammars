"""
Script for extracting a dataset to make it independent of the `spatial_scene_grammars`
library.
The dataset is saved in dictionary form, where each object is represented by a
dictionary with the following keys:
- "transform": The 4x4 transformation matrix of the object.
- "model_path": The path to the object's model.


Optionally also filtes the dataset for failure cases:

Considered failure cases are:
- Shared objects with non-zero rotations about the roll and pitch axes
- Shared objects with too high z-translation

Only the affected object is removed. The scene is removed if removing the object leads
to fewer than 3 objects remaining.

Shared objects are:
- SharedPlate
- SharedBowl
- CerealBox
- Jug
"""

import argparse
import pickle
from typing import List

import numpy as np
from tqdm import tqdm

from spatial_scene_grammars.drake_interop import PhysicsGeometryInfo
from spatial_scene_grammars.nodes import Node
from spatial_scene_grammars.scene_grammar import SceneTree
from scipy.spatial.transform import Rotation as rot
import spatial_scene_grammars_examples


def main(dataset_pickle_path: str, filter: bool, verbose: bool):
    assert dataset_pickle_path.endswith(".pickle") or dataset_pickle_path.endswith(
        ".pkl"
    )

    target_dataset_trees: List[SceneTree] = []

    pbar = tqdm(desc="Loading scenes")
    with open(dataset_pickle_path, "rb") as f:
        while 1:
            try:
                tree = pickle.load(f)
                if isinstance(tree, SceneTree):
                    target_dataset_trees.append(tree)
                elif isinstance(tree, tuple):
                    target_dataset_trees.append(tree[0])
                else:
                    raise ValueError(f"Unexpected type {type(tree)}")
                pbar.update(1)
            except EOFError:
                break
    pbar.close()

    observed_nodes: List[List[Node]] = [
        tree.get_observed_nodes() for tree in target_dataset_trees
    ]

    if filter:
        raise NotImplementedError(
            "Filtering is outdated. See generate_target_dataset_parallel.py for the new "
            "filtering."
        )

        filtered_observed_nodes = []
        num_objs_removed = 0
        for i, nodes in enumerate(tqdm(observed_nodes, desc="Filtering scenes")):
            filtered_nodes = []
            for node in nodes:
                translation = node.translation

                # Extract the local z-axis of the object's rotation matrix.
                local_z_axis = node.rotation @ np.array([0, 0, 1])

                # Not sure why need full path here for this to work.
                if (
                    isinstance(
                        node,
                        spatial_scene_grammars_examples.tri_table.grammar.SharedPlate,
                    )
                    or isinstance(
                        node,
                        spatial_scene_grammars_examples.tri_table.grammar.SharedBowl,
                    )
                    or isinstance(
                        node,
                        spatial_scene_grammars_examples.tri_table.grammar.CerealBox,
                    )
                ):
                    # Should have close to zero translation.
                    if not np.allclose(translation[2], 0.0, atol=3e-3, rtol=0.0):
                        if verbose:
                            print(
                                "Warning: Expected zero z-translation for scene "
                                f"{i}, {node}, got {translation[2]}."
                            )
                        num_objs_removed += 1
                        continue

                    # Should have close to zero roll and pitch.
                    if not np.allclose(local_z_axis, [0, 0, 1], atol=1e-2):
                        if verbose:
                            print(
                                "Warning: Expected zero roll and pitch for scene "
                                f"{i}, {node}, got z-axis = {local_z_axis}."
                            )
                        num_objs_removed += 1
                        continue

                if isinstance(
                    node, spatial_scene_grammars_examples.tri_table.grammar.Jug
                ):
                    # Should have close to 0.091m translation in z (frame at center).
                    if not np.allclose(translation[2], 0.091, atol=3e-3, rtol=0.0):
                        if verbose:
                            print(
                                "Warning: Expected 0.091m translation in z for scene "
                                f"{i}, {node}, got {translation[2]}."
                            )
                        num_objs_removed += 1
                        continue

                    # Should have close to zero roll and pitch.
                    if not np.allclose(local_z_axis, [0, 0, 1], atol=1e-2):
                        if verbose:
                            print(
                                "Warning: Expected zero roll and pitch for scene "
                                f"{i}, {node}, got z-axis = {local_z_axis}."
                            )
                        num_objs_removed += 1
                        continue

                filtered_nodes.append(node)

            # Keep all scenes with more than 3 objects.
            if len(filtered_nodes) >= 3:
                filtered_observed_nodes.append(filtered_nodes)

        # Print statistics.
        num_scenes_removed = len(observed_nodes) - len(filtered_observed_nodes)
        print(f"Removed {num_objs_removed} objects and {num_scenes_removed} scenes.")
    else:
        filtered_observed_nodes = observed_nodes

    observed_node_data: List[List[dict]] = []
    for nodes in tqdm(filtered_observed_nodes, desc="Processing scenes"):
        data = []
        for node in nodes:
            translation = node.translation
            rotation = node.rotation
            geometry_info: PhysicsGeometryInfo = node.physics_geometry_info

            # We expect all geometries to be specified with model paths.
            assert len(geometry_info.model_paths) == 1
            assert not geometry_info.visual_geometry
            assert not geometry_info.collision_geometry

            transform, model_path, _, q0_dict = geometry_info.model_paths[0]

            # `transform` is the transform from the object frame to the geometry frame.
            transform = transform.numpy()
            assert np.allclose(
                transform[:3, :3], np.eye(3)
            ), f"Expected identity rotation, got\n{transform[:3,:3]}"
            if not np.allclose(transform[:3, 3], 0) and verbose:
                print(
                    f"Warning: Got non-zero translation of {transform[:3, 3]} for "
                    + f"{model_path}!"
                )

            # We expect no joints and thus no joint angles.
            assert not q0_dict

            combined_transform = np.eye(4)
            combined_transform[:3, 3] = translation + transform[:3, 3]
            combined_transform[:3, :3] = rotation
            data.append(
                {
                    "transform": combined_transform,
                    "model_path": model_path,
                }
            )

        observed_node_data.append(data)

    print(f"Dataset has {len(observed_node_data)} examples.")

    postfix = f"{'_filtered' if filter else ''}_dict_form"
    save_path = dataset_pickle_path.replace(".pickle", f"{postfix}.pickle")
    save_path = save_path.replace(".pkl", f"{postfix}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(observed_node_data, f)
    print(f"Saved dataset in dictionary form to {save_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_pickle_path",
        type=str,
        help="Path to the dataset pickle file to extract the dataset from.",
    )
    parser.add_argument(
        "--not_filter",
        action="store_true",
        help="Filter out failure cases.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print more information."
    )
    args = parser.parse_args()
    main(args.dataset_pickle_path, not args.not_filter, args.verbose)
