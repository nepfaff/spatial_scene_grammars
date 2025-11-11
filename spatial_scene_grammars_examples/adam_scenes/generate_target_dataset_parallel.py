import logging

logging.disable(level=logging.ERROR)
logger = logging.getLogger("root").setLevel(logging.ERROR)
import argparse
import multiprocessing as mp
import os
import pickle
import time
import uuid
import tempfile
from multiprocessing import Pool, Lock
import fcntl
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

torch.set_default_dtype(torch.double)
import argparse
import os
import pickle
import warnings
from copy import deepcopy
from datetime import timedelta
from functools import partial
from typing import List

import numpy as np
from scipy.spatial.transform import Rotation as rot
from tqdm import tqdm

import spatial_scene_grammars_examples
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
from spatial_scene_grammars_examples.dimsum_restaurant.grammar import (
    ObjectOnTableSpacingConstraint,
    ObjectsOnTableConstraint,
    Restaurant,
    TallStackConstraint,
    TablesChairsAndShelvesNotInCollisionConstraint,
)
from spatial_scene_grammars_examples.tri_living_room_shelf.grammar import (
    BoardGameStackHeightConstraint,
    LargeBoardGameStackHeightConstraint,
    MinNumObjectsConstraint,
    ObjectsNotInCollisionWithStacksConstraintStructure,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Prevent numpy, torch multiprocessing to interfere with the outer multiprocessing loop.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)


def extract_tree(tree: SceneTree, filter: bool) -> List[dict] | None:
    """
    Function for extracting a dataset to make it independent of the
    `spatial_scene_grammars` library.
    The dataset is saved in dictionary form, where each object is represented by a
    dictionary with the following keys:
    - "transform": The 4x4 transformation matrix of the object.
    - "model_path": The path to the object's model.

    Optionally also filters the dataset for failure cases:

    Considered failure cases are:
    - Shared objects with non-zero rotations about the roll and pitch axes
    - Shared objects with too high z-translation

    Only the affected object is removed. The scene is removed if removing the object
    leads to fewer than 3 objects remaining.

    Note that the entire scene is removed if a main plate/ bowl does't have close to
    zero translation and roll/pitch.

    Shared objects are:
    - SharedPlate
    - SharedBowl
    - CerealBox
    - Jug
    """
    observed_nodes: List[Node] = tree.get_observed_nodes()

    filtered_observed_nodes = None
    if filter:
        filtered_nodes = []
        for node in observed_nodes:
            translation = np.array(node.translation)

            # Remove nodes with translation above 8m in any direction.
            if np.any(np.abs(translation) > 8):
                continue

            objects = (
                spatial_scene_grammars_examples.tri_living_room_shelf.grammar.Lamp,
                spatial_scene_grammars_examples.tri_living_room_shelf.grammar.BigBowl,
                spatial_scene_grammars_examples.tri_living_room_shelf.grammar.StandingEatToLiveBook,
                spatial_scene_grammars_examples.tri_living_room_shelf.grammar.StackingRing,
                spatial_scene_grammars_examples.tri_living_room_shelf.grammar.ToyTrain,
                spatial_scene_grammars_examples.tri_living_room_shelf.grammar.CokeCan,
                spatial_scene_grammars_examples.tri_living_room_shelf.grammar.TeaBottle,
                spatial_scene_grammars_examples.tri_living_room_shelf.grammar.JBLSpeaker,
            )
            if isinstance(node, objects):
                # Extract the local z-axis of the object's rotation matrix.
                local_z_axis = np.array(node.rotation) @ np.array([0, 0, 1])

                # Should have close to zero roll and pitch.
                if not np.allclose(local_z_axis, [0, 0, 1], atol=1e-2):
                    continue

            filtered_nodes.append(node)

        # Keep all scenes with more than 5 objects.
        if len(filtered_nodes) >= 5:
            filtered_observed_nodes = filtered_nodes
        else:
            return None
    else:
        filtered_observed_nodes = observed_nodes

    data: List[dict] = []
    for node in filtered_observed_nodes:
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


def sample_realistic_scene(
    grammar, constraints, seed=None, skip_physics_constraints=False
):
    """
    Sample a realistic scene from the given grammar and constraints.
    Optionally skip physics constraints.

    Attempts to first sample a tree structure that meets structure constraints.
    Then it samples poses (HMC-based) that meet pose constraints.
    Finally, it tries to project the solution to feasibility using Drake's physical sim.
    """
    if seed is not None:
        torch.random.manual_seed(seed)
    structure_constraints, pose_constraints = split_constraints(constraints)
    if len(structure_constraints) > 0:
        tree, success = rejection_sample_under_constraints(
            grammar, structure_constraints, 5000, detach=True, verbose=-1
        )
        if not success:
            return None, None
    else:
        tree = grammar.sample_tree(detach=True)

    if len(pose_constraints) > 0:
        samples = do_fixed_structure_hmc_with_constraint_penalties(
            grammar,
            tree,
            num_samples=25,
            subsample_step=1,
            with_nonpenetration=False,
            zmq_url="",
            constraints=pose_constraints,
            kernel_type="NUTS",
            max_tree_depth=6,
            target_accept_prob=0.8,
            adapt_step_size=True,
            verbose=-1,
        )

        # Check samples for constraint satisfaction
        good_tree = None
        best_bad_tree = None
        best_violation = None
        for candidate_tree in samples[::-1]:
            total_violation = eval_total_constraint_set_violation(
                candidate_tree, constraints
            )
            if total_violation <= 0.0:
                good_tree = candidate_tree
                break
            else:
                if best_bad_tree is None or total_violation <= best_violation:
                    best_bad_tree = candidate_tree
                    best_violation = total_violation.detach()
    else:
        good_tree = tree

    if good_tree is None:
        # No tree in samples satisfied constraints.
        return None, None

    if skip_physics_constraints:
        return None, good_tree

    feasible_tree = project_tree_to_feasibility(
        deepcopy(good_tree),
        do_forward_sim=True,
        timestep=0.001,
        T=2.5,
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
        root_node_type=Restaurant,
        root_node_tf=drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
    )
    constraints = [
        # Restaurant and table constraints.
        TallStackConstraint(),
        ObjectOnTableSpacingConstraint(),
        ObjectsOnTableConstraint(),
        TablesChairsAndShelvesNotInCollisionConstraint(),
        # Shelf constraints.
        BoardGameStackHeightConstraint(max_height=5),
        LargeBoardGameStackHeightConstraint(max_height=3),
        MinNumObjectsConstraint(min_num_objects=3),
        ObjectsNotInCollisionWithStacksConstraintStructure(),
    ]

    max_tries = 1
    counter = 0
    while counter < max_tries:
        try:
            tree, _ = sample_realistic_scene(grammar, constraints)
            if tree is not None:
                result = extract_tree(tree, filter=True) if extract else tree
                if result is not None:
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
            print(f"Exception during sampling in worker (task {task_id}): {e}")
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
