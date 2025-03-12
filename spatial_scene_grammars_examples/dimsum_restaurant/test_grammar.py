import logging
import os

import numpy as np
import torch

torch.set_default_dtype(torch.double)

from pydrake.all import StartMeshcat

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
    NumStacksConstraint,
    ObjectOnTableSpacingConstraint,
    ObjectsOnTableConstraint,
    Restaurant,
    TallStackConstraint,
)


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
            logging.error("Couldn't rejection sample a feasible tree config.")
            return None, None
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
    return feasible_tree, good_tree


# Create meshcat.
meshcat_instance = StartMeshcat()

grammar = SpatialSceneGrammar(
    root_node_type=Restaurant,
    root_node_tf=drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.0])),
)
constraint_list = [
    NumStacksConstraint(),
    TallStackConstraint(),
    ObjectOnTableSpacingConstraint(),
    ObjectsOnTableConstraint(),
]

tree, _ = sample_realistic_scene(grammar, constraint_list)
if tree is None:
    print("Sampling failed!")
    exit()

# Extract data.
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
print(scene)

## Visualize.

# Setup plant.
builder = DiagramBuilder()
plant: MultibodyPlant
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
parser = Parser(plant)
parser.SetAutoRenaming(True)
# Add Anzu package.
package_file_abs_path = os.path.abspath(os.path.expanduser("anzu/package.xml"))
parser.package_map().Add("anzu", os.path.dirname(package_file_abs_path))
# Add Gazebo package.
package_file_abs_path = os.path.abspath(os.path.expanduser("gazebo/package.xml"))
parser.package_map().Add("gazebo", os.path.dirname(package_file_abs_path))
# Add Greg table package.
package_file_abs_path = os.path.abspath(os.path.expanduser("greg_table/package.xml"))
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
simulator.AdvanceTo(20.0)
