import pytest

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import time

import torch
torch.set_default_dtype(torch.double)

from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.scene_grammar import *
from spatial_scene_grammars.visualization import *
from spatial_scene_grammars_examples.oriented_clusters.grammar import *
from spatial_scene_grammars.parsing import *

import meshcat
import meshcat.geometry as meshcat_geom

from pydrake.all import SnoptSolver

def test_sampling():
    vis = meshcat.Visualizer()

    # Draw a random sample from the grammar and visualize it.
    grammar = SpatialSceneGrammar(
        root_node_type = OrientedCluster,
        root_node_tf = torch.eye(4)
    )
    torch.random.manual_seed(42)
    tree = grammar.sample_tree()
    
    assert len(tree.find_nodes_by_type(OrientedCluster)) > 0, "Didn't sample any clusters."
    assert torch.isfinite(tree.score(verbose=True)), "Sampled tree was infeasible."

    draw_scene_tree_contents_meshcat(tree, zmq_url=vis.window.zmq_url)
    draw_scene_tree_structure_meshcat(tree, zmq_url=vis.window.zmq_url)

@pytest.mark.skipif(os.environ.get('GUROBI_PATH') is None or not SnoptSolver().available(),
                    reason='This test relies on Gurobi and SNOPT.')
def test_parsing():
    # Try to parse an example of this grammar.
    grammar = SpatialSceneGrammar(
        root_node_type = OrientedCluster,
        root_node_tf = torch.eye(4)
    )
    torch.random.manual_seed(42)
    observed_tree = grammar.sample_tree(detach=True)
    observed_nodes = observed_tree.get_observed_nodes()

    inference_results = infer_mle_tree_with_mip(
        grammar, observed_nodes, verbose=True,
    )
    mip_optimized_tree = get_optimized_tree_from_mip_results(inference_results)
    refinement_results = optimize_scene_tree_with_nlp(grammar, mip_optimized_tree, verbose=True)
    refined_tree = refinement_results.refined_tree
    score = refined_tree.score(verbose=True)
    assert torch.isfinite(score), "Refined tree was infeasible."

if __name__ == "__main__":
    pytest.main()
