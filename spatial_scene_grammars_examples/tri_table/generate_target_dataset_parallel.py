import logging

logging.disable(level=logging.ERROR)
logger = logging.getLogger("root").setLevel(logging.ERROR)
import argparse
import multiprocessing as mp
import os
import pickle
import time
from multiprocessing import Pool

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
from spatial_scene_grammars_examples.tri_table.grammar import (
    MinNumObjectsConstraint,
    ObjectsOnTableConstraint,
    SharedObjectsNotInCollisionWithPlateSettingsConstraint,
    Table,
)
from spatial_scene_grammars_examples.tri_table.grammar_high_clutter import (
    Table as HighClutterTable,
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

            # Extract the local z-axis of the object's rotation matrix.
            local_z_axis = np.array(node.rotation) @ np.array([0, 0, 1])

            main_objects = (
                spatial_scene_grammars_examples.tri_table.grammar.MainPlate,
                spatial_scene_grammars_examples.tri_table.grammar.MainBowl,
                spatial_scene_grammars_examples.tri_table.grammar_high_clutter.MainPlate,
                spatial_scene_grammars_examples.tri_table.grammar_high_clutter.MainBowl,
            )
            if isinstance(node, main_objects):
                # Ideally we remove the entire plate setting but we just drop the scene
                # for simplicity.

                # Should have close to zero translation.
                if not np.allclose(translation[2], 0.0, atol=3e-3, rtol=0.0):
                    return None

                # Should have close to zero roll and pitch.
                if not np.allclose(local_z_axis, [0, 0, 1], atol=1e-2):
                    return None

            # Remove nodes with translation above 5m in any direction.
            if np.any(np.abs(translation) > 5.0):
                continue

            # Not sure why need full path here for this to work.
            shared_objects = (
                spatial_scene_grammars_examples.tri_table.grammar.SharedPlate,
                spatial_scene_grammars_examples.tri_table.grammar.SharedBowl,
                spatial_scene_grammars_examples.tri_table.grammar.CerealBox,
                spatial_scene_grammars_examples.tri_table.grammar_high_clutter.SharedPlate,
                spatial_scene_grammars_examples.tri_table.grammar_high_clutter.SharedBowl,
                spatial_scene_grammars_examples.tri_table.grammar_high_clutter.CerealBox,
            )
            if isinstance(node, shared_objects):
                # Should have close to zero translation.
                if not np.allclose(translation[2], 0.0, atol=3e-3, rtol=0.0):
                    continue

                # Should have close to zero roll and pitch.
                if not np.allclose(local_z_axis, [0, 0, 1], atol=1e-2):
                    continue

            jugs = (
                spatial_scene_grammars_examples.tri_table.grammar.Jug,
                spatial_scene_grammars_examples.tri_table.grammar_high_clutter.Jug,
            )
            if isinstance(node, jugs):
                # Should have close to 0.091m translation in z (frame at center).
                if not np.allclose(translation[2], 0.091, atol=3e-3, rtol=0.0):
                    continue

                # Should have close to zero roll and pitch.
                if not np.allclose(local_z_axis, [0, 0, 1], atol=1e-2):
                    continue

            if isinstance(
                node, spatial_scene_grammars_examples.tri_table.grammar.UtensilCrock
            ):
                # Should have close to zero roll and pitch.
                if not np.allclose(local_z_axis, [0, 0, 1], atol=1e-2):
                    # Can't just remove as it contains utensils.
                    return None

            filtered_nodes.append(node)

        # Keep all scenes with more than 3 objects.
        if len(filtered_nodes) >= 3:
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
            grammar, structure_constraints, 1000, detach=True, verbose=-1
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
        T=5.0,
        static_models="package://anzu/models/visuomotor/add_riverway_without_arms.dmd.yaml",
    )
    return feasible_tree, good_tree


def sample_and_save(grammar, constraints, extract, max_tries: int = 1):
    """
    Attempt to sample a realistic scene that meets constraints.
    Returns either the extracted data or the tree.

    If it fails up to max_tries times, returns None.
    """
    # Set a unique seed for each process
    seed = (int(time.time() * 1000000) + os.getpid()) % (2**32)
    np.random.seed(seed)
    torch.manual_seed(seed)

    counter = 0
    while counter < max_tries:
        try:
            tree, _ = sample_realistic_scene(grammar, constraints)
            if tree is not None:
                if extract:
                    return extract_tree(tree, filter=True)
                return tree
        except Exception as e:
            # If something goes wrong, try again until max_tries is reached.
            print("Exception during sampling in worker:", e)
        counter += 1
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_save_file", type=str)
    parser.add_argument("--points", type=int)
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    parser.add_argument("--extract", type=bool, default=True)
    parser.add_argument("--high_clutter", type=bool, default=False)
    args = parser.parse_args()
    dataset_save_file: str = args.dataset_save_file
    assert dataset_save_file.endswith(".pkl")
    extract: bool = args.extract
    high_clutter: bool = args.high_clutter
    N: int = args.points
    processes: int = min(args.workers, mp.cpu_count())

    start = time.time()

    grammar = SpatialSceneGrammar(
        root_node_type=HighClutterTable if high_clutter else Table,
        root_node_tf=drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
    )
    constraints = [
        ObjectsOnTableConstraint(),
        MinNumObjectsConstraint(
            min_num_objects=3,
            table_node_type=HighClutterTable if high_clutter else Table,
        ),
        SharedObjectsNotInCollisionWithPlateSettingsConstraint(
            table_node_type=HighClutterTable if high_clutter else Table
        ),
    ]

    # Set sharing strategy to file_system instead of file_descriptor
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    pool = Pool(processes=processes)

    chunk_size = 1000 if N > 1000 else N
    num_chunks = N // chunk_size
    remainder = N % chunk_size

    task_func = sample_and_save
    task_args = (grammar, constraints, extract)
    task_timeout = 800

    # Process full chunks, write after each chunk
    for _ in tqdm(range(num_chunks), desc="Generating dataset"):
        # Launch all tasks concurrently
        async_results = [
            pool.apply_async(task_func, args=task_args) 
            for _ in range(chunk_size)
        ]
        
        # Collect results incrementally to avoid memory buildup
        with open(dataset_save_file, "ab") as f:
            for i, async_res in enumerate(async_results):
                try:
                    res = async_res.get(timeout=task_timeout)
                    if res is not None:
                        pickle.dump(res, f)
                        f.flush()  # Ensure data is written to disk
                except mp.TimeoutError:
                    print(f"Worker {i} timeout")
                except Exception as e:
                    print(f"Worker {i} exception: {e}")
                
                # Clear reference to help garbage collection
                async_results[i] = None
                
                # Periodically force garbage collection
                if i % 100 == 99:
                    import gc
                    gc.collect()

    # Process remainder, if any
    if remainder > 0:
        async_results = [
            pool.apply_async(task_func, args=task_args)
            for _ in range(remainder)
        ]
        
        with open(dataset_save_file, "ab") as f:
            for i, async_res in enumerate(async_results):
                try:
                    res = async_res.get(timeout=task_timeout)
                    if res is not None:
                        pickle.dump(res, f)
                        f.flush()
                except mp.TimeoutError:
                    print(f"Remainder worker {i} timeout")
                except Exception as e:
                    print(f"Remainder worker {i} exception: {e}")
                
                # Clear reference
                async_results[i] = None

    pool.close()
    pool.join()

    print(
        f"Generating dataset of {N} samples took {timedelta(seconds=time.time()-start)}"
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
