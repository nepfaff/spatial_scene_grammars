import argparse
import pickle
from typing import List

import numpy as np
from tqdm import tqdm

from spatial_scene_grammars.drake_interop import PhysicsGeometryInfo
from spatial_scene_grammars.nodes import Node
from spatial_scene_grammars.scene_grammar import SceneTree


def main(dataset_pickle_path: str, verbose: bool):
    assert dataset_pickle_path.endswith(".pickle") or dataset_pickle_path.endswith(".pkl")

    target_dataset_trees: List[SceneTree] = []

    pbar = tqdm(desc="Loading scenes")
    with open(dataset_pickle_path, "rb") as f:
        while 1:
            try:
                tree =  pickle.load(f)
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

    observed_node_data: List[List[dict]] = []
    for nodes in tqdm(observed_nodes, desc="Processing scenes"):
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

    save_path = dataset_pickle_path.replace(".pickle", "_dict_form.pickle")
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
        "--verbose", action="store_true", help="Print more information."
    )
    args = parser.parse_args()
    main(args.dataset_pickle_path, args.verbose)
