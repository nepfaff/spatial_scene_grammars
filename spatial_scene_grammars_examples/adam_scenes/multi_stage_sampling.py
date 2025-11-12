"""Multi-stage hierarchical sampling for Adam scenes.

This module implements a 3-stage sampling approach to reduce rejection sampling overhead:
- Stage 1: Sample empty bins and shelves layout
- Stage 2: Populate each container independently + sample floor objects
- Stage 3: Combine all trees into final scene

This approach is more efficient for dataset generation than sampling the full scene
with rejection constraints.
"""

import logging
import torch
import numpy as np
from copy import deepcopy
from pydrake.all import RigidTransform

from spatial_scene_grammars.drake_interop import drake_tf_to_torch_tf, project_tree_to_feasibility
from spatial_scene_grammars.sampling import (
    rejection_sample_under_constraints,
    do_fixed_structure_hmc_with_constraint_penalties,
    eval_total_constraint_set_violation,
)
from spatial_scene_grammars.scene_grammar import SpatialSceneGrammar, SceneTree

from .grammar import (
    Stage1AdamScene,
    EmptyClutteredBin,
    EmptyShelf,
    ClutteredBin,
    Shelf,
    SharedStuff,
    FloorObjectsRoot,
    ShelvesNotInCollisionWithBinsConstraint,
    SharedStuffNotInCollisionWithShelvesAndBins,
    ObjectsOutsideIiwa,
    ObjectsWithinArcConstraint,
    MinNumShelvesAndBinsConstraint,
    Teacup,
    Teapot,
)
from spatial_scene_grammars_examples.tri_living_room_shelf.grammar import (
    BoardGameStackHeightConstraint,
    LargeBoardGameStackHeightConstraint,
    MinNumObjectsConstraint,
    ObjectsNotInCollisionWithStacksConstraintStructure,
    Lamp,
    BigBowl,
    StandingEatToLiveBook,
    StackingRing,
    ToyTrain,
    CokeCan,
    TeaBottle,
    JBLSpeaker,
)


def sample_stage1_layout(seed=None, max_attempts=5000):
    """Sample stage 1: Layout of empty containers (bins and shelves).

    Args:
        seed: Random seed for reproducibility
        max_attempts: Maximum rejection sampling attempts

    Returns:
        SceneTree with empty bins and shelves at final poses, or None if failed
    """
    if seed is not None:
        torch.random.manual_seed(seed)

    # Create stage 1 grammar (empty containers only)
    stage1_grammar = SpatialSceneGrammar(
        root_node_type=Stage1AdamScene,
        root_node_tf=drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
    )

    # Define structure constraints for container layout
    # Note: Only use constraints that work with empty containers (EmptyShelf, EmptyClutteredBin)
    structure_constraints = [
        MinNumShelvesAndBinsConstraint(min_count=2),
        ShelvesNotInCollisionWithBinsConstraint(),
        # Note: ObjectsWithinArcConstraint checks tabletop objects, which don't exist in stage 1
        # Arc constraint is already enforced by CircularOffsetRule in EmptyShelves/EmptyBins
    ]

    # Rejection sample container layout
    tree, success = rejection_sample_under_constraints(
        stage1_grammar, structure_constraints, max_attempts, detach=True, verbose=-1
    )

    if not success:
        logging.info(
            f"[Stage 1] Failed to sample container layout after {max_attempts} attempts"
        )
        return None

    logging.info(f"[Stage 1] Successfully sampled container layout")
    return tree


def sample_shelf_contents(shelf_pose, seed=None, max_projection_attempts=50):
    """Sample contents for a single shelf.

    Args:
        shelf_pose: 4x4 transformation matrix for shelf pose
        seed: Random seed for reproducibility
        max_projection_attempts: Maximum attempts to sample + project before giving up

    Returns:
        SceneTree with shelf and its contents
    """
    if seed is not None:
        torch.random.manual_seed(seed)

    # Create grammar rooted at Shelf with the given pose
    shelf_grammar = SpatialSceneGrammar(root_node_type=Shelf, root_node_tf=shelf_pose)

    # Define shelf structure constraints
    shelf_structure_constraints = [
        BoardGameStackHeightConstraint(max_height=5),
        LargeBoardGameStackHeightConstraint(max_height=3),
        MinNumObjectsConstraint(min_num_objects=3, table_node_type=Shelf),
        ObjectsNotInCollisionWithStacksConstraintStructure(),
    ]

    # Retry sampling + projection if constraints not satisfied or projection fails
    for attempt in range(max_projection_attempts):
        # Sample shelf contents using rejection sampling with structure constraints
        tree, constraints_satisfied = rejection_sample_under_constraints(
            shelf_grammar,
            shelf_structure_constraints,
            max_num_attempts=1000,
            detach=True,
            verbose=0,
        )

        if not constraints_satisfied:
            logging.info(f"[Stage 2a] Warning: Could not satisfy all shelf constraints (attempt {attempt+1}/{max_projection_attempts}), re-sampling...")
            continue

        # Check if shelf has any movable objects (not just Null nodes)
        if not _has_movable_objects(tree):
            logging.info(f"[Stage 2a] Shelf contains only Null nodes, skipping projection")
            return tree

        # Project shelf contents to physical feasibility
        logging.info(f"[Stage 2a] Projecting shelf contents to feasibility (attempt {attempt+1}/{max_projection_attempts})...")
        projected_tree = project_tree_to_feasibility(
            deepcopy(tree),
            do_forward_sim=True,
            timestep=0.001,
            T=2.5,  # Shorter sim time for smaller scenes
            fix_orientation=True,  # Keep books upright
        )

        if projected_tree is not None:
            logging.info(f"[Stage 2a] Shelf projection successful, filtering objects...")

            # Filter objects with bad poses (flew away, tipped over, etc.)
            filtered_tree = _filter_shelf_objects(projected_tree, min_objects=3)

            if filtered_tree is not None:
                logging.info(f"[Stage 2a] Shelf filtering successful")
                return filtered_tree
            else:
                logging.info(f"[Stage 2a] Shelf filtering failed (too few valid objects), re-sampling...")
                continue
        else:
            logging.info(f"[Stage 2a] Shelf projection failed, re-sampling...")

    # If all attempts failed, return last unprojected tree
    logging.info(f"[Stage 2a] WARNING: All {max_projection_attempts} projection attempts failed, using unprojected tree")
    return tree


def _has_movable_objects(tree):
    """Check if a scene tree has any movable (non-fixed) objects with geometry.

    Returns False if tree only contains Null nodes or fixed geometry.
    """
    for node in tree.nodes:
        # Skip Null nodes (no geometry)
        if node.__class__.__name__ == 'Null':
            continue

        # Check if node has movable geometry
        if hasattr(node, 'physics_geometry_info') and node.physics_geometry_info is not None:
            if not node.physics_geometry_info.fixed:
                return True

    return False


def _filter_shelf_objects(tree, min_objects=3):
    """Filter shelf objects with bad poses after projection.

    Removes objects that:
    - Flew away during simulation (|translation| > 5m)
    - Tipped over (for upright objects like Lamp, BigBowl, etc.)

    Args:
        tree: SceneTree to filter
        min_objects: Minimum number of valid objects required

    Returns:
        Filtered SceneTree or None if too few valid objects remain
    """
    # Object types that should remain upright (local z-axis aligned with world z-axis)
    upright_objects = (Lamp, BigBowl, StandingEatToLiveBook, StackingRing,
                      ToyTrain, CokeCan, TeaBottle, JBLSpeaker)

    filtered_tree = SceneTree()
    filtered_nodes = []

    for node in tree.nodes:
        # Skip non-observed nodes (containers, root, etc.)
        if not node.observed:
            continue

        # Check translation threshold (objects shouldn't fly away)
        translation = np.array(node.translation)
        if np.any(np.abs(translation) > 5.0):
            logging.info(f"[Filter] Removing {node.__class__.__name__} due to excessive translation: {translation}")
            continue

        # Check orientation for upright objects
        if isinstance(node, upright_objects):
            # Extract the local z-axis of the object's rotation matrix
            local_z_axis = np.array(node.rotation) @ np.array([0, 0, 1])

            # Should have close to zero roll and pitch
            if not np.allclose(local_z_axis, [0, 0, 1], atol=1e-2):
                logging.info(f"[Filter] Removing {node.__class__.__name__} due to bad orientation: {local_z_axis}")
                continue

        filtered_nodes.append(node)

    # Check minimum object count
    if len(filtered_nodes) < min_objects:
        logging.info(f"[Filter] Only {len(filtered_nodes)} valid shelf objects, need at least {min_objects}")
        return None

    # Reconstruct filtered tree
    # TODO: This is a simplified version - may need to preserve tree structure properly
    for node in tree.nodes:
        if not node.observed or node in filtered_nodes:
            filtered_tree.add_node(node)

    for parent, child in tree.edges:
        if parent in filtered_tree.nodes and child in filtered_tree.nodes:
            filtered_tree.add_edge(parent, child)

    logging.info(f"[Filter] Shelf filtering: {len(filtered_nodes)} valid objects (removed {len([n for n in tree.nodes if n.observed]) - len(filtered_nodes)})")
    return filtered_tree


def _filter_floor_objects(tree, min_objects=2):
    """Filter floor objects with bad poses after projection.

    Removes objects that:
    - Flew away during simulation (|translation| > 8m)
    - Fell through or floated above floor (z not in [-0.1, 0.5])
    - Tipped over (for Teacup and Teapot only - steamers can tumble)

    Args:
        tree: SceneTree to filter
        min_objects: Minimum number of valid objects required

    Returns:
        Filtered SceneTree or None if too few valid objects remain
    """
    # Object types that should remain upright
    upright_floor_objects = (Teacup, Teapot)  # NOT steamers - they can tumble

    filtered_tree = SceneTree()
    filtered_nodes = []

    for node in tree.nodes:
        # Skip non-observed nodes
        if not node.observed:
            continue

        # Check translation threshold (objects shouldn't fly away)
        translation = np.array(node.translation)
        if np.any(np.abs(translation) > 8.0):
            logging.info(f"[Filter] Removing {node.__class__.__name__} due to excessive translation: {translation}")
            continue

        # Check z-height (objects should be on floor, not fallen through or floating)
        z = translation[2]
        if z < -0.1 or z > 0.5:
            logging.info(f"[Filter] Removing {node.__class__.__name__} due to bad z-height: {z}")
            continue

        # Check orientation for upright floor objects (teacups and teapots)
        if isinstance(node, upright_floor_objects):
            # Extract the local z-axis of the object's rotation matrix
            local_z_axis = np.array(node.rotation) @ np.array([0, 0, 1])

            # Should have close to zero roll and pitch
            if not np.allclose(local_z_axis, [0, 0, 1], atol=1e-2):
                logging.info(f"[Filter] Removing {node.__class__.__name__} due to bad orientation: {local_z_axis}")
                continue

        filtered_nodes.append(node)

    # Check minimum object count
    if len(filtered_nodes) < min_objects:
        logging.info(f"[Filter] Only {len(filtered_nodes)} valid floor objects, need at least {min_objects}")
        return None

    # Reconstruct filtered tree
    for node in tree.nodes:
        if not node.observed or node in filtered_nodes:
            filtered_tree.add_node(node)

    for parent, child in tree.edges:
        if parent in filtered_tree.nodes and child in filtered_tree.nodes:
            filtered_tree.add_edge(parent, child)

    logging.info(f"[Filter] Floor filtering: {len(filtered_nodes)} valid objects (removed {len([n for n in tree.nodes if n.observed]) - len(filtered_nodes)})")
    return filtered_tree


def _filter_bin_objects(tree, min_objects=2):
    """Filter bin objects with bad poses after projection.

    Removes objects that:
    - Flew away during simulation (|translation| > 5m)
    - Fell through bin floor (z < -0.1)

    NO orientation checks - bin objects can tumble freely.

    Args:
        tree: SceneTree to filter
        min_objects: Minimum number of valid objects required (including bin)

    Returns:
        Filtered SceneTree or None if too few valid objects remain
    """
    filtered_tree = SceneTree()
    filtered_nodes = []

    for node in tree.nodes:
        if not node.observed:
            continue

        translation = np.array(node.translation)

        # Check XY translation threshold (objects shouldn't fly away)
        if np.any(np.abs(translation) > 5.0):
            logging.info(f"[Filter] Removing {node.__class__.__name__} due to excessive translation: {translation}")
            continue

        # Check z-height (objects shouldn't fall through bin floor)
        z = translation[2]
        if z < -0.1:
            logging.info(f"[Filter] Removing {node.__class__.__name__} due to falling through bin floor: z={z}")
            continue

        # NO orientation checks - bins allow arbitrary rotations

        filtered_nodes.append(node)

    # Check minimum object count (bin + at least 1 object)
    if len(filtered_nodes) < min_objects:
        logging.info(f"[Filter] Only {len(filtered_nodes)} valid bin objects, need at least {min_objects}")
        return None

    # Reconstruct filtered tree
    for node in tree.nodes:
        if not node.observed or node in filtered_nodes:
            filtered_tree.add_node(node)

    for parent, child in tree.edges:
        if parent in filtered_tree.nodes and child in filtered_tree.nodes:
            filtered_tree.add_edge(parent, child)

    logging.info(f"[Filter] Bin filtering: {len(filtered_nodes)} valid objects (removed {len([n for n in tree.nodes if n.observed]) - len(filtered_nodes)})")
    return filtered_tree


def sample_bin_contents(bin_pose, seed=None, max_projection_attempts=50):
    """Sample contents for a single bin.

    Args:
        bin_pose: 4x4 transformation matrix for bin pose
        seed: Random seed for reproducibility
        max_projection_attempts: Maximum attempts to sample + project before giving up

    Returns:
        SceneTree with bin and its contents
    """
    if seed is not None:
        torch.random.manual_seed(seed)

    # Create grammar rooted at ClutteredBin with the given pose
    bin_grammar = SpatialSceneGrammar(
        root_node_type=ClutteredBin, root_node_tf=bin_pose
    )

    # Retry sampling + projection if projection fails
    for attempt in range(max_projection_attempts):
        # Sample bin contents
        tree = bin_grammar.sample_tree(detach=True)

        # Check if bin has any movable objects (not just Null nodes)
        if not _has_movable_objects(tree):
            logging.info(f"[Stage 2b] Bin contains only Null nodes, skipping projection")
            return tree

        # Project bin contents to physical feasibility
        logging.info(f"[Stage 2b] Projecting bin contents to feasibility (attempt {attempt+1}/{max_projection_attempts})...")
        projected_tree = project_tree_to_feasibility(
            deepcopy(tree),
            do_forward_sim=True,
            timestep=0.001,
            T=5.0,  # Longer sim time for dense bin packing
            fix_orientation=False,  # Allow bin objects to rotate freely
        )

        if projected_tree is not None:
            logging.info(f"[Stage 2b] Bin projection successful, filtering objects...")

            # Filter objects with bad poses (flew away, fell through floor, etc.)
            filtered_tree = _filter_bin_objects(projected_tree, min_objects=2)

            if filtered_tree is not None:
                logging.info(f"[Stage 2b] Bin filtering successful")
                return filtered_tree
            else:
                logging.info(f"[Stage 2b] Bin filtering failed (too few valid objects), re-sampling...")
                continue
        else:
            logging.info(f"[Stage 2b] Bin projection failed, re-sampling...")

    # If all attempts failed, return last unprojected tree
    logging.info(f"[Stage 2b] WARNING: All {max_projection_attempts} projection attempts failed, using unprojected tree")
    return tree


def sample_floor_objects(stage1_tree, bin_poses, shelf_poses, pose_constraints=None, seed=None, max_attempts_collision=1000, max_attempts_hmc=100, max_projection_attempts=50):
    """Sample floor objects (SharedStuff) with bins/shelves as fixed obstacles.

    Args:
        stage1_tree: Stage 1 scene tree (contains scene root with floor geometry)
        bin_poses: List of 4x4 transformation matrices for bin poses
        shelf_poses: List of 4x4 transformation matrices for shelf poses
        pose_constraints: List of PoseConstraints to apply via HMC (e.g., ObjectsOutsideIiwa, ObjectsWithinArcConstraint)
        seed: Random seed for reproducibility
        max_attempts_collision: Maximum rejection sampling attempts for collision checking
        max_attempts_hmc: Maximum attempts to find HMC-satisfying tree
        max_projection_attempts: Maximum attempts to sample + project before giving up

    Returns:
        SceneTree with SharedStuff, or None if failed
    """
    if seed is not None:
        torch.random.manual_seed(seed)

    if pose_constraints is None:
        pose_constraints = []

    # Create FloorObjectsRoot grammar at origin
    # FloorObjectsRoot will place SharedStuff via CircularOffsetRule in pie region [-110, 110]
    floor_grammar = SpatialSceneGrammar(
        root_node_type=FloorObjectsRoot,
        root_node_tf=torch.eye(4),
    )

    # Retry entire sampling + projection process if projection fails
    for projection_attempt in range(max_projection_attempts):
        logging.info(f"[Stage 2c] Floor sampling attempt {projection_attempt+1}/{max_projection_attempts}")

        # Create temporary scene tree with fixed obstacles for constraint checking
        # We'll sample floor objects with collision checking, then return just the floor objects
        def sample_with_obstacle_checking():
            # Sample floor objects
            floor_tree = floor_grammar.sample_tree(detach=True)

            # Create a temporary combined tree for constraint checking
            temp_tree = SceneTree()

            # Add floor objects to temp tree
            for node in floor_tree.nodes:
                temp_tree.add_node(node)
            for node in floor_tree.nodes:
                parent = floor_tree.get_parent(node)
                if parent is not None:
                    temp_tree.add_edge(parent, node)

            # Add fixed empty bins as obstacles in temp tree
            for bin_pose in bin_poses:
                obstacle_bin = EmptyClutteredBin(tf=bin_pose)
                temp_tree.add_node(obstacle_bin)
                # Note: obstacle bins are not connected to tree, just present for collision checking

            # Add fixed empty shelves as obstacles in temp tree
            for shelf_pose in shelf_poses:
                obstacle_shelf = EmptyShelf(tf=shelf_pose)
                temp_tree.add_node(obstacle_shelf)
                # Note: obstacle shelves are not connected to tree, just present for collision checking

            # Check collision constraint
            collision_constraint = SharedStuffNotInCollisionWithShelvesAndBins()
            violation = collision_constraint.eval(temp_tree)

            # Return floor_tree only if no collision
            if torch.all(violation >= 0.0):
                return floor_tree
            else:
                return None

        # Rejection sample until we get a collision-free configuration
        floor_tree = None
        for attempt in range(max_attempts_collision):
            result = sample_with_obstacle_checking()
            if result is not None:
                floor_tree = result
                logging.info(f"[Stage 2c] Sampled collision-free floor objects (attempt {attempt+1})")
                break

        if floor_tree is None:
            logging.info(f"[Stage 2c] Failed to sample collision-free floor objects after {max_attempts_collision} attempts")
            floor_tree = floor_grammar.sample_tree(detach=True)  # Fallback without collision checking

        # Apply HMC with pose constraints if any
        if len(pose_constraints) > 0:
            logging.info(f"[Stage 2c] Applying HMC with {len(pose_constraints)} pose constraints...")

            # Try HMC multiple times since it might not always converge
            for hmc_attempt in range(max_attempts_hmc):
                samples = do_fixed_structure_hmc_with_constraint_penalties(
                    None,  # No grammar needed for fixed structure HMC
                    floor_tree,
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
                    structure_vis_kwargs={
                        "with_triad": False,
                        "linewidth": 30,
                        "node_sphere_size": 0.02,
                        "alpha": 0.5,
                    },
                )

                # Find a sample that satisfies constraints
                good_tree = None
                for candidate_tree in samples[::-1]:
                    total_violation = eval_total_constraint_set_violation(candidate_tree, pose_constraints)
                    if total_violation <= 0.0:
                        good_tree = candidate_tree
                        break

                if good_tree is not None:
                    logging.info(f"[Stage 2c] HMC succeeded (attempt {hmc_attempt+1})")
                    floor_tree = good_tree
                    break
                else:
                    # Retry HMC from a new collision-free sample
                    new_sample = sample_with_obstacle_checking()
                    if new_sample is not None:
                        floor_tree = new_sample
                    # Otherwise keep the current floor_tree and retry
            else:
                logging.info(f"[Stage 2c] HMC failed after {max_attempts_hmc} attempts, proceeding to projection")

        # Project floor objects to physical feasibility with bins/shelves as obstacles
        logging.info(f"[Stage 2c] Projecting floor objects to feasibility with obstacles...")

        # Create temporary tree with scene root (includes floor geometry) + floor objects + empty containers
        temp_tree_for_projection = SceneTree()

        # Add scene root with floor geometry (from stage1_tree)
        scene_root = stage1_tree.get_root()
        temp_tree_for_projection.add_node(scene_root)

        # Add floor objects as children of scene root
        floor_root = floor_tree.get_root()  # FloorObjectsRoot
        for child in floor_tree.get_children(floor_root):  # SharedStuff and its children
            _add_subtree_recursive(temp_tree_for_projection, scene_root, child, floor_tree)

        # Add empty bins/shelves as fixed obstacles (also children of scene root)
        for bin_pose in bin_poses:
            obstacle_bin = EmptyClutteredBin(tf=bin_pose)
            temp_tree_for_projection.add_node(obstacle_bin)
            temp_tree_for_projection.add_edge(scene_root, obstacle_bin)

        for shelf_pose in shelf_poses:
            obstacle_shelf = EmptyShelf(tf=shelf_pose)
            temp_tree_for_projection.add_node(obstacle_shelf)
            temp_tree_for_projection.add_edge(scene_root, obstacle_shelf)

        # Project with obstacles
        projected_tree = project_tree_to_feasibility(
            deepcopy(temp_tree_for_projection),
            do_forward_sim=True,
            timestep=0.001,
            T=3.0,
            fix_orientation=True,
        )

        if projected_tree is not None:
            # Extract just the floor objects back (remove obstacle nodes and scene root)
            # Floor objects are in SharedStuff subtree (FloorObjectsRoot was not added to projection tree)
            shared_stuff_node = None
            for node in projected_tree.nodes:
                if node.__class__.__name__ == 'SharedStuff':
                    shared_stuff_node = node
                    break

            if shared_stuff_node is not None:
                # Create new tree with SharedStuff and all its descendants
                final_floor_tree = SceneTree()
                final_floor_tree.add_node(shared_stuff_node)
                for child in projected_tree.get_children(shared_stuff_node):
                    _add_subtree_recursive(final_floor_tree, shared_stuff_node, child, projected_tree)

                logging.info(f"[Stage 2c] Floor projection successful, filtering objects...")

                # Filter objects with bad poses (flew away, fell through floor, tipped over, etc.)
                filtered_floor_tree = _filter_floor_objects(final_floor_tree, min_objects=2)

                if filtered_floor_tree is not None:
                    logging.info(f"[Stage 2c] Floor filtering successful")
                    return filtered_floor_tree
                else:
                    logging.info(f"[Stage 2c] Floor filtering failed (too few valid objects), re-sampling...")
                    continue
            else:
                logging.info(f"[Stage 2c] WARNING: Could not find SharedStuff in projected tree, re-sampling...")
                # Continue to next projection attempt
        else:
            logging.info(f"[Stage 2c] Floor projection failed, re-sampling...")
            # Continue to next projection attempt

    # If all projection attempts failed, return last unprojected tree
    logging.info(f"[Stage 2c] WARNING: All {max_projection_attempts} floor projection attempts failed, using unprojected tree")
    return floor_tree


def _add_subtree_recursive(target_tree, new_parent, node, source_tree):
    """Recursively add node and its descendants from source_tree to target_tree.

    Args:
        target_tree: SceneTree to add nodes to
        new_parent: Parent node in target_tree for the subtree
        node: Root node of subtree to add
        source_tree: Source SceneTree containing the subtree
    """
    target_tree.add_node(node)
    target_tree.add_edge(new_parent, node)

    # Recursively add children
    for child in source_tree.get_children(node):
        _add_subtree_recursive(target_tree, node, child, source_tree)


def combine_trees(stage1_tree, shelf_contents_dict, bin_contents_dict, floor_tree):
    """Combine all stage results into final complete scene tree.

    Args:
        stage1_tree: Stage 1 tree with empty containers
        shelf_contents_dict: {empty_shelf_node: populated_shelf_tree}
        bin_contents_dict: {empty_bin_node: populated_bin_tree}
        floor_tree: Tree with floor objects (SharedStuff)

    Returns:
        Complete combined SceneTree
    """
    combined_tree = SceneTree()

    # Add all nodes and edges from stage 1 (environment + empty containers)
    for node in stage1_tree.nodes:
        combined_tree.add_node(node)

    for node in stage1_tree.nodes:
        parent = stage1_tree.get_parent(node)
        if parent is not None:
            combined_tree.add_edge(parent, node)

    # Replace empty shelves with populated shelves
    for empty_shelf, shelf_tree in shelf_contents_dict.items():
        # Find parent of empty shelf in combined tree
        parent = combined_tree.get_parent(empty_shelf)

        # Get the root of the populated shelf tree (the Shelf node)
        shelf_root = shelf_tree.get_root()

        # Remove empty shelf from combined tree (NetworkX method)
        combined_tree.remove_node(empty_shelf)

        # Add populated shelf and all its descendants
        _add_subtree_recursive(combined_tree, parent, shelf_root, shelf_tree)

    # Replace empty bins with populated bins
    for empty_bin, bin_tree in bin_contents_dict.items():
        # Find parent of empty bin in combined tree
        parent = combined_tree.get_parent(empty_bin)

        # Get the root of the populated bin tree (the ClutteredBin node)
        bin_root = bin_tree.get_root()

        # Remove empty bin from combined tree (NetworkX method)
        combined_tree.remove_node(empty_bin)

        # Add populated bin and all its descendants
        _add_subtree_recursive(combined_tree, parent, bin_root, bin_tree)

    # Add floor objects to scene root
    if floor_tree is None:
        logging.info("[Stage 3] WARNING: floor_tree is None, skipping floor objects")
    else:
        floor_root = floor_tree.get_root()
        scene_root = combined_tree.get_root()
        for child in floor_tree.get_children(floor_root):
            _add_subtree_recursive(combined_tree, scene_root, child, floor_tree)

    logging.info(f"[Stage 3] Combined all trees: {len(combined_tree.nodes)} total nodes")
    return combined_tree


def sample_hierarchical_scene(pose_constraints=None, seed=None):
    """Main entry point for hierarchical multi-stage scene sampling.

    This function orchestrates all three stages sequentially:
    1. Sample empty container layout
    2. Populate each container + sample floor objects
    3. Combine into final scene

    Args:
        pose_constraints: List of PoseConstraints to apply at appropriate stages
        seed: Random seed for reproducibility

    Returns:
        Complete SceneTree, or None if sampling failed
    """
    if pose_constraints is None:
        pose_constraints = []

    # Extract floor-specific pose constraints
    # These are constraints that apply to tabletop objects on the floor (SharedStuff)
    floor_pose_constraints = [
        c for c in pose_constraints
        if isinstance(c, (ObjectsOutsideIiwa, ObjectsWithinArcConstraint))
    ]

    # Stage 1: Sample container layout
    stage1_tree = sample_stage1_layout(seed=seed)
    if stage1_tree is None:
        return None

    # Extract empty containers from stage 1
    empty_bins = stage1_tree.find_nodes_by_type(EmptyClutteredBin)
    empty_shelves = stage1_tree.find_nodes_by_type(EmptyShelf)

    logging.info(f"[Stage 1] Found {len(empty_bins)} bins and {len(empty_shelves)} shelves")

    # Stage 2a: Sample contents for each shelf SEQUENTIALLY (no parallelism)
    shelf_contents = {}
    for i, empty_shelf in enumerate(empty_shelves):
        logging.info(f"[Stage 2a] Sampling shelf {i+1}/{len(empty_shelves)} contents...")
        shelf_tree = sample_shelf_contents(empty_shelf.tf)
        shelf_contents[empty_shelf] = shelf_tree

    # Stage 2b: Sample contents for each bin SEQUENTIALLY
    bin_contents = {}
    for i, empty_bin in enumerate(empty_bins):
        logging.info(f"[Stage 2b] Sampling bin {i+1}/{len(empty_bins)} contents...")
        bin_tree = sample_bin_contents(empty_bin.tf)
        bin_contents[empty_bin] = bin_tree

    # Stage 2c: Sample floor objects with bins/shelves as obstacles
    bin_poses = [b.tf for b in empty_bins]
    shelf_poses = [s.tf for s in empty_shelves]
    floor_tree = sample_floor_objects(
        stage1_tree,  # Pass stage1_tree to include floor geometry in projection
        bin_poses, shelf_poses,
        pose_constraints=floor_pose_constraints,
        seed=seed
    )

    # Stage 3: Combine all trees
    final_tree = combine_trees(stage1_tree, shelf_contents, bin_contents, floor_tree)

    logging.info(f"[Hierarchical Sampling] Complete!")
    return final_tree
