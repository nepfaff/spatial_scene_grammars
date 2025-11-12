import logging

logging.disable(level=logging.ERROR)
logger = logging.getLogger("root").setLevel(logging.ERROR)
import argparse
import multiprocessing as mp
import os
import pickle
import time
import tempfile
from multiprocessing import Pool
import fcntl
import warnings
from copy import deepcopy
from datetime import timedelta
from typing import List

import numpy as np
import torch
from tqdm import tqdm

torch.set_default_dtype(torch.double)
from spatial_scene_grammars.constraints import *
from spatial_scene_grammars.dataset import *
from spatial_scene_grammars.drake_interop import PhysicsGeometryInfo
from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.nodes import Node
from spatial_scene_grammars.parameter_estimation import *
from spatial_scene_grammars.parsing import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.sampling import *
from spatial_scene_grammars.scene_grammar import *
from spatial_scene_grammars.visualization import *
from spatial_scene_grammars_examples.adam_scenes.grammar import (
    AdamScene,
    ObjectsOutsideIiwa,
    MinNumShelvesAndBinsConstraint,
    ShelvesNotInCollisionWithBinsConstraint,
    ObjectsWithinArcConstraint,
    SharedStuffNotInCollisionWithShelvesAndBins,
)
from spatial_scene_grammars_examples.adam_scenes.multi_stage_sampling import (
    sample_hierarchical_scene,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Prevent numpy, torch multiprocessing to interfere with the outer multiprocessing loop.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)


def extract_tree(tree: SceneTree) -> List[dict]:
    """
    Function for extracting a dataset to make it independent of the
    `spatial_scene_grammars` library.
    The dataset is saved in dictionary form, where each object is represented by a
    dictionary with the following keys:
    - "transform": The 4x4 transformation matrix of the object.
    - "model_path": The path to the object's model.

    Note: Filtering for failure cases is now handled by the hierarchical sampling
    pipeline, so this function only does the extraction.
    """
    observed_nodes: List[Node] = tree.get_observed_nodes()

    data: List[dict] = []
    for node in observed_nodes:
        translation = node.translation
        rotation = node.rotation
        geometry_info: PhysicsGeometryInfo = node.physics_geometry_info

        # We expect all geometries to be specified with model paths.
        assert len(geometry_info.model_paths) == 1
        assert not geometry_info.visual_geometry
        assert not geometry_info.collision_geometry

        transform, model_path, _, q0_dict = geometry_info.model_paths[0]

        transform = transform.numpy()
        assert np.allclose(
            transform[:3, :3], np.eye(3)
        ), f"Expected identity rotation, got\n{transform[:3,:3]}"

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

    return data


def sample_hierarchical_realistic_scene(
    grammar, constraints, seed=None, skip_physics_constraints=False
):
    """Sample a scene using hierarchical multi-stage approach.

    This function uses the multi-stage sampling pipeline instead of
    sampling the full scene with rejection constraints.

    Args:
        grammar: Grammar object (not used, kept for API consistency)
        constraints: List of constraints to check (pose constraints only, structure handled in stages)
        seed: Random seed
        skip_physics_constraints: If True, skip physics projection

    Returns:
        (feasible_tree, good_tree) tuple, or (None, None) if failed
    """
    if seed is not None:
        torch.random.manual_seed(seed)

    # Extract only pose constraints (structure already handled by stage grammars)
    _, pose_constraints = split_constraints(constraints)

    # Use hierarchical sampling with pose constraints applied at each stage
    # Note: grammar parameter is not used here since stage grammars are created internally
    tree = sample_hierarchical_scene(pose_constraints=pose_constraints, seed=seed)
    if tree is None:
        logging.error("Hierarchical sampling failed.")
        return None, None

    # HMC is now applied per-stage within sample_hierarchical_scene()
    # No need to apply HMC again on the combined scene
    good_tree = tree

    if skip_physics_constraints:
        return None, good_tree

    feasible_tree = project_tree_to_feasibility(
        deepcopy(good_tree),
        do_forward_sim=True,
        timestep=0.001,
        T=1.0,  # Reduced from 2.5s - objects already mostly stable from per-stage projection
    )
    return feasible_tree, good_tree


def sample_and_save_direct(extract, output_file, task_id):
    """
    Sample a scene and save it directly to the output file using file locking.
    This avoids the shared memory manager bottleneck while maintaining a single output file.
    """
    # Set a unique seed for each process
    seed = (int(time.time() * 1000000) + os.getpid() + task_id) % (2**32)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create grammar and constraints inside the worker
    grammar = SpatialSceneGrammar(
        root_node_type=AdamScene,
        root_node_tf=drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
    )
    constraints = [
        # Pose constraints (applied per-stage during hierarchical sampling)
        ObjectsOutsideIiwa(radius=0.35),
        ObjectsWithinArcConstraint(angle_min=-115.0, angle_max=115.0),
        # Structure constraints (applied during stage 1 container layout)
        MinNumShelvesAndBinsConstraint(min_count=2),
        ShelvesNotInCollisionWithBinsConstraint(),
        # Collision constraints (applied during stage 2c floor sampling)
        SharedStuffNotInCollisionWithShelvesAndBins(),
        # Note: Shelf-specific structure constraints (BoardGameStackHeightConstraint, etc.)
        # are now enforced naturally by the shelf grammar during stage 2a sampling
    ]

    max_tries = 1
    counter = 0
    while counter < max_tries:
        try:
            tree, _ = sample_hierarchical_realistic_scene(grammar, constraints)
            if tree is not None:
                result = extract_tree(tree) if extract else tree
                # Create a temporary file with the result
                with tempfile.NamedTemporaryFile(
                    delete=False, mode="wb"
                ) as temp_file:
                    pickle.dump(result, temp_file)
                    temp_path = temp_file.name

                # Append the temporary file to the output file with file locking
                with open(output_file, "ab") as f:
                    # Acquire an exclusive lock
                    fcntl.flock(f, fcntl.LOCK_EX)
                    try:
                        # Read the temporary file and append its contents
                        with open(temp_path, "rb") as temp:
                            f.write(temp.read())
                        # Ensure data is written to disk
                        f.flush()
                        os.fsync(f.fileno())
                    finally:
                        # Release the lock
                        fcntl.flock(f, fcntl.LOCK_UN)

                # Remove the temporary file
                os.unlink(temp_path)

                return True
        except Exception as e:
            logging.error(f"Exception during sampling in worker (task {task_id}): {e}")
        counter += 1
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_save_file", type=str)
    parser.add_argument("--points", type=int)
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    parser.add_argument("--extract", type=bool, default=True)
    args = parser.parse_args()
    dataset_save_file: str = args.dataset_save_file
    assert dataset_save_file.endswith(".pkl")
    extract: bool = args.extract
    N: int = args.points
    processes: int = min(args.workers, mp.cpu_count())

    start = time.time()

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(dataset_save_file)), exist_ok=True)

    # Create an empty output file if it doesn't exist
    if not os.path.exists(dataset_save_file):
        with open(dataset_save_file, "wb") as f:
            pass

    # Set sharing strategy to file_system
    torch.multiprocessing.set_sharing_strategy("file_system")

    pool = Pool(processes=processes)

    # Create task arguments - each worker writes directly to the output file
    task_args = [(extract, dataset_save_file, i) for i in range(N)]

    # Launch all tasks
    print(f"Launching {N} tasks across {processes} workers...")
    
    # Use a list to collect results and a tqdm progress bar
    results = []
    pbar = tqdm(total=N, desc="Generating scenes")
    
    # Define a callback function to update the progress bar
    def update_pbar(result):
        pbar.update(1)
        results.append(result)
    
    # Use apply_async with callback to update progress bar
    jobs = []
    for args in task_args:
        job = pool.apply_async(sample_and_save_direct, args, callback=update_pbar)
        jobs.append(job)
    
    # Wait for all jobs to complete
    for job in jobs:
        job.wait()
    
    # Close the progress bar
    pbar.close()
    
    # Close the pool
    pool.close()
    pool.join()

    # Count successful scenes
    successful_count = sum(1 for r in results if r)
    print(f"Successfully generated {successful_count} scenes out of {N} attempts")

    print(
        f"Generating dataset of {successful_count} samples took {timedelta(seconds=time.time()-start)}"
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
