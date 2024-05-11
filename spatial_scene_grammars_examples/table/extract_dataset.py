import pickle
from typing import List

import numpy as np

from spatial_scene_grammars.drake_interop import PhysicsGeometryInfo
from spatial_scene_grammars.nodes import Node
from spatial_scene_grammars.scene_grammar import SceneTree


def main():
    target_dataset_trees: List[SceneTree] = []
    with open("target_dataset_examples.pickle", "rb") as f:
        while 1:
            try:
                target_dataset_trees.append(pickle.load(f))
            except EOFError:
                break

    observed_nodes: List[List[Node]] = [
        tree.get_observed_nodes() for tree in target_dataset_trees
    ]

    observed_node_data: List[List[dict]] = []
    for nodes in observed_nodes:
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
            if not np.allclose(transform[:3, 3], 0):
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

    with open("dimsum_dataset_dict_form.pickle", "wb") as f:
        pickle.dump(observed_node_data, f)


if __name__ == "__main__":
    main()
