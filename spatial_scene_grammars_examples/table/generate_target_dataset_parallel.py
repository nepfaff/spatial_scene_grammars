import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pickle
import time
from tqdm.notebook import tqdm
from multiprocessing import Pool
import torch

torch.set_default_tensor_type(torch.DoubleTensor)
from datetime import timedelta
from spatial_scene_grammars.constraints import *
from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.scene_grammar import *
from spatial_scene_grammars.visualization import *
from spatial_scene_grammars_examples.table.grammar import *
from spatial_scene_grammars.parsing import *
from spatial_scene_grammars.sampling import *
from spatial_scene_grammars.parameter_estimation import *
from spatial_scene_grammars.dataset import *
from functools import partial


def sample_realistic_scene(
    grammar, constraints, seed=None, skip_physics_constraints=False
):
    if seed is not None:
        torch.random.manual_seed(seed)
    structure_constraints, pose_constraints = split_constraints(constraints)
    if len(structure_constraints) > 0:
        tree, success = rejection_sample_under_constraints(
            grammar, structure_constraints, 1000, detach=True
        )
        if not success:
            logging.error("Couldn't rejection sample a feasible tree config.")
            return None, None
    else:
        tree = grammar.sample_tree(detach=True)

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

    if skip_physics_constraints:
        return None, good_tree

    feasible_tree = project_tree_to_feasibility(
        deepcopy(good_tree), do_forward_sim=True, timestep=0.001, T=1.0
    )
    return feasible_tree, good_tree


def sample_and_save(grammar, constraints, discard_arg=None):
    while True:
        tree, _ = sample_realistic_scene(grammar, constraints)
        if tree is not None:
            return tree


def save_tree(tree, dataset_save_file):
    with open(dataset_save_file, "a+b") as f:
        pickle.dump(tree, f)


def main():
    dataset_save_file = "structure_constraint_examples_1000.pickle"
    N = 1000
    processes = 20
    
    # Check if file already exists
    assert not os.path.exists(dataset_save_file), "Dataset file already exists!"

    start = time.time()

    # Set up grammar and constraint set.
    grammar = SpatialSceneGrammar(
        root_node_type=Table,
        root_node_tf=drake_tf_to_torch_tf(RigidTransform(p=[0.0, 0.0, 0.8])),
    )
    constraints = [
        ObjectsOnTableConstraint(),
        ObjectSpacingConstraint(),
        TallStackConstraint(),
        NumStacksConstraint(),
    ]

    # Produce dataset by sampling a bunch of environments.
    # Try to collect a target number of examples, and save them out
    num_chunks = N // 100
    chunks = np.split(np.array(list(range(N))), num_chunks)
    trees = []
    for chunk in chunks:
        with Pool(processes=processes) as pool:
            trees += pool.map(partial(sample_and_save, grammar, constraints), chunk)

    print("Finished sampling, saving trees...")
    for tree in trees:
        save_tree(tree, dataset_save_file)

    print(
        f"Generating dataset of {N} samples took {timedelta(seconds=time.time()-start)}"
    )


if __name__ == "__main__":
    main()
