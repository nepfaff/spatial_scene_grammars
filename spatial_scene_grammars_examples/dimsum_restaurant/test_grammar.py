import logging
import os
import argparse
import multiprocessing as mp
import time
from multiprocessing import Pool
from copy import deepcopy

import numpy as np
import torch

torch.set_default_dtype(torch.double)

from pydrake.all import (
    StartMeshcat,
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Parser,
    RigidTransform,
    Simulator,
    MeshcatVisualizer,
)

from spatial_scene_grammars.constraints import *
from spatial_scene_grammars.dataset import *
from spatial_scene_grammars.nodes import *
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


def sample_realistic_scene(
    grammar, constraints, seed=None, skip_physics_constraints=False
):
    if seed is not None:
        torch.random.manual_seed(seed)
    structure_constraints, pose_constraints = split_constraints(constraints)
    if len(structure_constraints) > 0:
        tree, success = rejection_sample_under_constraints(
            grammar, structure_constraints, 5000, detach=True, verbose=-1
        )  # Need a high-rejection sample budged due to the hard structure constraints
        if not success:
            logging.error("Couldn't rejection sample a feasible tree config.")
            return None, None
        print("Successfully rejection sampled a tree config.")
    else:
        tree = grammar.sample_tree(detach=True)

    if len(pose_constraints) > 0:
        print("HMC sampling with constraints.")
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
            logging.error("No tree in samples satisfied constraints.")
            print("Best total violation: %f" % best_violation)
            print("Violations of best bad tree:")
            for constraint in constraints:
                print("constraint ", constraint, ": ", constraint.eval(best_bad_tree))
            return None, None
    else:
        # Don't need HMC if have no pose constraints.
        good_tree = tree

    if skip_physics_constraints:
        return None, good_tree

    print("Projecting tree to feasibility.")
    feasible_tree = project_tree_to_feasibility(
        deepcopy(good_tree),
        do_forward_sim=True,
        timestep=0.001,
        T=2.5,
    )
    # feasible_tree = good_tree  # TODO: remove
    return feasible_tree, good_tree


def extract_scene_data(tree):
    """Extract scene data from a tree for visualization"""
    scene = []
    for node in tree.get_observed_nodes():
        node: Node
        translation = node.translation.numpy()
        rotation = node.rotation.numpy()
        geometry_info: PhysicsGeometryInfo = node.physics_geometry_info

        # We expect all geometries to be specified with model paths.
        assert len(geometry_info.model_paths) == 1
        assert not geometry_info.visual_geometry
        assert not geometry_info.collision_geometry

        transform, model_path, _, q0_dict = geometry_info.model_paths[0]

        # `transform` is the transform from the object frame to the geometry frame.
        # Don't currently support non-identity rotations.
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
        scene.append(
            {
                "transform": combined_transform,
                "model_path": model_path,
            }
        )
    return scene


def sample_and_extract(grammar, constraints, max_tries=3):
    """Worker function for parallel sampling"""
    # Set a unique seed for each process
    seed = (int(time.time() * 1000000) + os.getpid()) % (2**32)
    np.random.seed(seed)
    torch.manual_seed(seed)

    counter = 0
    while counter < max_tries:
        try:
            tree, _ = sample_realistic_scene(grammar, constraints)
            if tree is not None:
                return extract_scene_data(tree)
        except Exception as e:
            print(f"Exception during sampling in worker {os.getpid()}: {e}")
        counter += 1
    return None


def visualize_scene(scene, meshcat_instance):
    """Visualize a scene in meshcat"""
    # Clear previous scene
    meshcat_instance.Delete()

    # Setup plant.
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    parser.SetAutoRenaming(True)
    # Add Anzu package.
    package_file_abs_path = os.path.abspath(os.path.expanduser("anzu/package.xml"))
    if os.path.exists(package_file_abs_path):
        parser.package_map().Add("anzu", os.path.dirname(package_file_abs_path))
    # Add Gazebo package.
    package_file_abs_path = os.path.abspath(os.path.expanduser("gazebo/package.xml"))
    if os.path.exists(package_file_abs_path):
        parser.package_map().Add("gazebo", os.path.dirname(package_file_abs_path))
    # Add Greg table package.
    package_file_abs_path = os.path.abspath(
        os.path.expanduser("greg_table/package.xml")
    )
    if os.path.exists(package_file_abs_path):
        parser.package_map().Add("greg_table", os.path.dirname(package_file_abs_path))

    # Add scene models.
    for obj in scene:
        model_path = obj["model_path"]
        transform = obj["transform"]

        model = parser.AddModelsFromUrl(model_path)
        assert len(model) == 1
        model = model[0]

        # Set scene model transforms.
        body_indices = plant.GetBodyIndices(model)
        for body_index in body_indices:
            body = plant.get_body(body_index)
            plant.WeldFrames(
                plant.world_frame(),
                body.body_frame(),
                RigidTransform(transform),
            )

    plant.Finalize()

    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat_instance)

    diagram = builder.Build()

    # Simulate.
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(0.1)  # Just enough to visualize

    return diagram, simulator


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate restaurant scenes")
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes (default: 1)"
    )
    args = parser.parse_args()

    num_workers = min(args.workers, mp.cpu_count())

    # Create meshcat instance
    meshcat_instance = StartMeshcat()

    # Create grammar and constraints
    grammar = SpatialSceneGrammar(
        root_node_type=Restaurant,
        root_node_tf=drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
    )
    constraint_list = [
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

    # If single worker, use original behavior
    if num_workers == 1:
        tree, _ = sample_realistic_scene(grammar, constraint_list)
        if tree is None:
            print("Sampling failed!")
            exit()

        # Extract data
        scene = extract_scene_data(tree)
        print(scene)

        # Visualize
        diagram, simulator = visualize_scene(scene, meshcat_instance)

        # Simulate for a longer time
        simulator.set_target_realtime_rate(1.0)
        simulator.AdvanceTo(20.0)
    else:
        # Set up multiprocessing
        print(f"Generating {num_workers} scenes in parallel...")

        # Prevent numpy/torch from interfering with multiprocessing
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        torch.set_num_threads(1)

        # Use spawn method for better compatibility
        mp.set_start_method("spawn", force=True)

        # Create a pool of workers
        pool = Pool(processes=num_workers)

        # Generate scenes in parallel - one per worker
        async_results = [
            pool.apply_async(sample_and_extract, args=(grammar, constraint_list))
            for _ in range(num_workers)
        ]

        # Collect successful scenes
        scenes = []
        current_scene = 0
        diagram = None
        simulator = None
        viewing_started = False

        try:
            # Process results as they become available
            remaining = list(range(len(async_results)))

            while remaining:
                for i in remaining[:]:
                    async_res = async_results[i]
                    if async_res.ready():
                        try:
                            scene = async_res.get(timeout=1)
                            remaining.remove(i)

                            if scene is not None:
                                scenes.append(scene)
                                print(
                                    f"Successfully generated scene {len(scenes)}/{num_workers}"
                                )

                                # Start viewing as soon as the first scene is available
                                if not viewing_started and scenes:
                                    viewing_started = True
                                    print(
                                        f"\nViewing scene 1/{len(scenes)}. More scenes are being generated..."
                                    )
                                    print(
                                        "Instructions: Enter 'n' for next scene, 'p' for previous scene, 'q' to quit"
                                    )
                                    diagram, simulator = visualize_scene(
                                        scenes[0], meshcat_instance
                                    )
                        except Exception as e:
                            print(f"Error processing scene {i+1}: {e}")
                            remaining.remove(i)

                # Check if user wants to interact while waiting for more scenes
                if viewing_started:
                    import select
                    import sys

                    # Check if there's input available (non-blocking)
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        cmd = (
                            input(f"Scene {current_scene+1}/{len(scenes)} > ")
                            .strip()
                            .lower()
                        )
                        if cmd == "n" or cmd == "next":
                            if (
                                len(scenes) > 1
                            ):  # Only allow navigation if we have multiple scenes
                                current_scene = (current_scene + 1) % len(scenes)
                                diagram, simulator = visualize_scene(
                                    scenes[current_scene], meshcat_instance
                                )
                                print(f"Viewing scene {current_scene+1}/{len(scenes)}")
                        elif cmd == "p" or cmd == "prev":
                            if (
                                len(scenes) > 1
                            ):  # Only allow navigation if we have multiple scenes
                                current_scene = (current_scene - 1) % len(scenes)
                                diagram, simulator = visualize_scene(
                                    scenes[current_scene], meshcat_instance
                                )
                                print(f"Viewing scene {current_scene+1}/{len(scenes)}")
                        elif cmd == "q" or cmd == "quit":
                            break
                        else:
                            print(
                                "Unknown command. Use 'n' for next, 'p' for previous, 'q' to quit"
                            )
                else:
                    # If we haven't started viewing yet, just wait a bit
                    time.sleep(0.5)

            # All workers have finished
            pool.close()
            pool.join()
            print(f"Successfully generated {len(scenes)}/{num_workers} scenes")

        except KeyboardInterrupt:
            print("\nInterrupted by user. Stopping workers...")
            pool.terminate()
            pool.join()
            print(f"Generated {len(scenes)}/{num_workers} scenes before interruption")

        # If we haven't started viewing yet but have scenes, start now
        if not viewing_started and scenes:
            viewing_started = True
            current_scene = 0
            diagram, simulator = visualize_scene(
                scenes[current_scene], meshcat_instance
            )

        # Interactive visualization of successful scenes
        if not scenes:
            print("No successful scenes were generated.")
            exit()

        total_scenes = len(scenes)

        if viewing_started:
            print(
                f"\nAll scenes generated. Viewing scene {current_scene+1}/{total_scenes}."
            )
            print(
                "Instructions: Enter 'n' for next scene, 'p' for previous scene, 'q' to quit"
            )

        # Interactive loop for viewing scenes (after all generation is complete)
        try:
            while viewing_started:
                cmd = (
                    input(f"Scene {current_scene+1}/{total_scenes} > ").strip().lower()
                )
                if cmd == "n" or cmd == "next":
                    current_scene = (current_scene + 1) % total_scenes
                    diagram, simulator = visualize_scene(
                        scenes[current_scene], meshcat_instance
                    )
                    print(f"Viewing scene {current_scene+1}/{total_scenes}")
                elif cmd == "p" or cmd == "prev":
                    current_scene = (current_scene - 1) % total_scenes
                    diagram, simulator = visualize_scene(
                        scenes[current_scene], meshcat_instance
                    )
                    print(f"Viewing scene {current_scene+1}/{total_scenes}")
                elif cmd == "q" or cmd == "quit":
                    break
                else:
                    print(
                        "Unknown command. Use 'n' for next, 'p' for previous, 'q' to quit"
                    )
        except KeyboardInterrupt:
            print("\nExiting scene viewer.")
