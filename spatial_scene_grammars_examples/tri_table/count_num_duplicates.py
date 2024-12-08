import argparse
import hashlib
import pickle

from typing import Dict, List

from tqdm import tqdm
import numpy as np

def load_trees(dataset_save_file):
    list_of_trees = []
    with open(dataset_save_file, "rb") as f:
        while True:
            try:
                tree_list = pickle.load(f)
                list_of_trees.append(tree_list)
            except EOFError:
                break  # End of file reached
    return list_of_trees

def compare_scenes(scene1: List[Dict], scene2: List[Dict], atol=1e-8) -> bool:
    """
    Compare two scenes to check if they are identical within a specified absolute tolerance.

    Parameters:
    - scene1: List[Dict], first scene to compare.
    - scene2: List[Dict], second scene to compare.
    - atol: float, absolute tolerance for comparing transformation matrices.

    Returns:
    - bool: True if scenes are identical within the tolerance, False otherwise.
    """
    if len(scene1) != len(scene2):
        return False
    for obj1, obj2 in zip(scene1, scene2):
        # Compare model paths.
        if obj1.get("model_path") != obj2.get("model_path"):
            return False
        # Compare transformation matrices.
        if not np.allclose(obj1.get("transform"), obj2.get("transform"), atol=atol):
            return False
    return True


def hash_scene(scene: List[Dict], atol=1e-8) -> str:
    """
    Generate a hash for a scene based on its contents.

    Parameters:
    - scene: The scene to hash.
    - atol: Absolute tolerance used for rounding numerical values.

    Returns:
    - str: A hexadecimal hash string representing the scene.
    """
    scene_data = []
    for obj in scene:
        model_path = obj.get("model_path")
        transform = obj.get("transform")
        # Round the transform matrix to account for tolerance
        transform_rounded = np.round(transform / atol) * atol
        scene_data.append((model_path, transform_rounded.tolist()))
    # Serialize the scene data
    scene_bytes = pickle.dumps(scene_data)
    # Generate a hash
    scene_hash = hashlib.sha256(scene_bytes).hexdigest()
    return scene_hash


def remove_all_duplicates(scenes: List[List[Dict]], atol=1e-8) -> List[List[Dict]]:
    """
    Remove all duplicate scenes using hashing for improved performance.

    Parameters:
    - scenes: List of scenes to process.
    - atol: float, absolute tolerance for comparing scenes.

    Returns:
    - List of scenes with duplicates removed.
    """
    unique_scenes = []
    seen_hashes = set()
    for current_scene in tqdm(scenes):
        scene_hash = hash_scene(current_scene, atol=atol)
        if scene_hash not in seen_hashes:
            seen_hashes.add(scene_hash)
            unique_scenes.append(current_scene)
    return unique_scenes


def main():
    parser = argparse.ArgumentParser(
        description="Remove duplicate scenes from a Greg scene grammar pickle file."
    )
    parser.add_argument(
        "scenes_pickle_path",
        type=str,
        help="Path to the pickle file containing scenes.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance for comparing scenes. Default is 1e-3.",
    )
    parser.add_argument(
        "--is_combined",
        type=bool,
        default=False,
        help="Whether the dataset has already been combined. Default is False."
    )
    args = parser.parse_args()
    
    scenes_pickle_path = args.scenes_pickle_path
    assert scenes_pickle_path.endswith(".pkl") or scenes_pickle_path.endswith(".pickle")
    with open(scenes_pickle_path, "rb") as f:
        dataset = pickle.load(f)
    if not args.is_combined:
        # Check that format is correct.
        if not all(
            isinstance(data, list) for data in dataset
        ):
            dataset = load_trees(scenes_pickle_path)
        assert all(
            isinstance(data, list) for data in dataset
        ), "Wrong dataset format!"
    
    # Validate data format.
    scene_sample = dataset[0]
    obj_sample = scene_sample[0]
    assert (
        isinstance(obj_sample, dict)
        and "transform" in obj_sample
        and "model_path" in obj_sample
    )

    # Remove all duplicates across the entire list.
    deduplicated_scenes = remove_all_duplicates(dataset, atol=args.atol)

    print("Num scenes:", len(dataset))
    print("Num duplicates:", len(deduplicated_scenes) - len(deduplicated_scenes))
    print("Num unique scenes:", len(deduplicated_scenes))


if __name__ == "__main__":
    main()
