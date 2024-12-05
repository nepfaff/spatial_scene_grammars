import logging

logging.disable(level=logging.ERROR)
logger = logging.getLogger("root").setLevel(logging.ERROR)
import argparse
import os
import pickle
import time
from multiprocessing import Pool
import multiprocessing as mp
import numpy as np
import torch
from tqdm import tqdm

torch.set_default_dtype(torch.double)
from datetime import timedelta
from functools import partial

from spatial_scene_grammars.constraints import *
from spatial_scene_grammars.dataset import *
from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.parameter_estimation import *
from spatial_scene_grammars.parsing import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.sampling import *
from spatial_scene_grammars.scene_grammar import *
from spatial_scene_grammars.visualization import *
from spatial_scene_grammars_examples.tri_table.grammar import (
    Table,
    ObjectsOnTableConstraint,
    MinNumObjectsConstraint,
    SharedObjectsNotInCollisionWithPlateSettingsConstraint,
)
from spatial_scene_grammars_examples.tri_table.grammar_high_clutter import (
    Table as HighClutterTable,
)
import argparse
import pickle
from typing import List

import numpy as np
from tqdm import tqdm

from spatial_scene_grammars.drake_interop import PhysicsGeometryInfo
from spatial_scene_grammars.nodes import Node
from scipy.spatial.transform import Rotation as rot
import spatial_scene_grammars_examples

import os

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Prevent numpy, torch multiprocessing to interfer with the outer multiprocessing loop.
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


    Optionally also filtes the dataset for failure cases:

    Considered failure cases are:
    - Shared objects with non-zero rotations about the roll and pitch axes
    - Shared objects with too high z-translation

    Only the affected object is removed. The scene is removed if removing the object
    leads to fewer than 3 objects remaining.

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
                    continue

                # Should have close to zero roll and pitch.
                if not np.allclose(local_z_axis, [0, 0, 1], atol=1e-2):
                    continue

            if isinstance(node, spatial_scene_grammars_examples.tri_table.grammar.Jug):
                # Should have close to 0.091m translation in z (frame at center).
                if not np.allclose(translation[2], 0.091, atol=3e-3, rtol=0.0):
                    continue

                # Should have close to zero roll and pitch.
                if not np.allclose(local_z_axis, [0, 0, 1], atol=1e-2):
                    continue

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

        # `transform` is the transform from the object frame to the geometry frame.
        transform = transform.numpy()
        assert np.allclose(
            transform[:3, :3], np.eye(3)
        ), f"Expected identity rotation, got\n{transform[:3,:3]}"

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

    return data


def sample_realistic_scene(
    grammar, constraints, seed=None, skip_physics_constraints=False
):
    if seed is not None:
        torch.random.manual_seed(seed)
    structure_constraints, pose_constraints = split_constraints(constraints)
    if len(structure_constraints) > 0:
        tree, success = rejection_sample_under_constraints(
            grammar, structure_constraints, 1000, detach=True, verbose=-1
        )
        if not success:
            # logging.error("Couldn't rejection sample a feasible tree config.")
            return None, None
    else:
        tree = grammar.sample_tree(detach=True)

    samples = do_fixed_structure_hmc_with_constraint_penalties(
        grammar,
        tree,
        num_samples=25,
        subsample_step=1,
        with_nonpenetration=False,  # Too difficult
        zmq_url="",
        constraints=pose_constraints,
        kernel_type="NUTS",
        max_tree_depth=6,
        target_accept_prob=0.8,
        adapt_step_size=True,
        verbose=-1,
        # kernel_type="HMC", num_steps=1, step_size=1E-1, adapt_step_size=False, # Langevin-ish
        structure_vis_kwargs={
            "with_triad": False,
            "linewidth": 30,
            "node_sphere_size": 0.02,
            "alpha": 0.5,
        },
    )

    # Step through samples backwards in HMC process and pick out a tree that satisfies
    # the constraints.
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

    if good_tree == None:
        # logging.error("No tree in samples satisfied constraints.")
        # print("Best total violation: %f" % best_violation)
        # print("Violations of best bad tree:")
        # for constraint in constraints:
        #     print("constraint ", constraint, ": ", constraint.eval(best_bad_tree))
        return None, None

    if skip_physics_constraints:
        return None, good_tree

    feasible_tree = project_tree_to_feasibility(
        deepcopy(good_tree), do_forward_sim=True, timestep=0.001, T=2.5
    )
    return feasible_tree, good_tree


def sample_and_save(
    grammar, constraints, extract, discard_arg=None, max_tries: int = 30
):
    # Set a unique seed for each process to prevent each process producing the same
    # scene.
    seed = (int(time.time() * 1000) + os.getpid()) % (2**32)
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
        except BaseException as e:
            print("Exception during sampling!", e)
            pass

        counter += 1

    print("Failed to find tree within budget!")


def save_tree(tree, dataset_save_file):
    if tree is None:
        # print("Tree is None, skipping save.")
        return
    with open(dataset_save_file, "a+b") as f:
        pickle.dump(tree, f)


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

    # Ensure regular saving.
    num_chunks = N // 1000

    # Check if file already exists
    # assert not os.path.exists(dataset_save_file), "Dataset file already exists!"

    start = time.time()

    # Set up grammar and constraint set.
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

    # Produce dataset by sampling a bunch of environments.
    # Try to collect a target number of examples, and save them out
    if processes == 1:
        for _ in tqdm(range(N), desc="Generating dataset"):
            tree = sample_and_save(grammar, constraints, extract)
            save_tree(tree, dataset_save_file)
    else:
        chunks = np.split(np.array(list(range(N))), num_chunks)
        for chunk in tqdm(chunks, desc="Generating dataset", position=0):
            with Pool(processes=processes) as pool:
                trees = pool.map(
                    partial(sample_and_save, grammar, constraints, extract), chunk
                )

                # Remove None trees.
                trees = [tree for tree in trees if tree is not None]
                # print(f"Collected {len(trees)} trees out of {len(chunk)} samples.")
                for tree in tqdm(trees, desc="  Saving trees", leave=False, position=1):
                    save_tree(tree, dataset_save_file)

    print(
        f"Generating dataset of {N} samples took {timedelta(seconds=time.time()-start)}"
    )


if __name__ == "__main__":
    main()
