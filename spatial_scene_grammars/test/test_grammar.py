import pytest

import torch
import pyro
import pyro.poutine

from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.scene_grammar import *

from .grammar import *

from torch.distributions import constraints

from pytorch3d.transforms.rotation_conversions import (
    axis_angle_to_matrix
)

torch.set_default_dtype(torch.double)

@pytest.fixture(params=range(10))
def set_seed(request):
    pyro.set_rng_seed(request.param)


## Basic grammar and tree functionality
def test_grammar(set_seed):
    grammar = SpatialSceneGrammar(
        root_node_type = NodeA,
        root_node_tf = torch.eye(4)
    )

    assert grammar.all_types == set([NodeA, NodeB, NodeC, NodeD, NodeE, NodeF, NodeG])

    tree = grammar.sample_tree()

    assert(torch.isfinite(tree.score()))

    assert isinstance(tree, SceneTree)

    root = tree.get_root()
    assert isinstance(root, NodeA)
    assert len(list(tree.predecessors(root))) == 0
    assert len(list(tree.successors(root))) == 2 # AND rule with 2 children

    obs = tree.get_observed_nodes()
    if len(obs) > 0:
        assert all([isinstance(c, (NodeD, NodeE, NodeF, NodeG)) for c in obs])

    assert len(tree.make_from_observed_nodes(obs).nodes) == len(obs)

    assert len(tree.find_nodes_by_type(NodeA)) == 1

    A = tree.find_nodes_by_type(NodeA)[0]
    B = tree.find_nodes_by_type(NodeB)[0]
    assert tree.get_parent(A) is None
    assert tree.get_parent(B) is A
    assert tree.get_rule_for_child(A, B) is A.rules[0]

def test_tree_score(set_seed):
    grammar = SpatialSceneGrammar(
        root_node_type = NodeA,
        root_node_tf = torch.eye(4)
    )

    trace = pyro.poutine.trace(grammar.sample_tree).get_trace()
    tree = trace.nodes["_RETURN"]["value"]
    expected_score = trace.log_prob_sum()
    score = tree.score()
    assert torch.isclose(expected_score, score), "%f vs %f" % (score, expected_score)

def test_supertree():
    grammar = SpatialSceneGrammar(
        root_node_type = NodeA,
        root_node_tf = torch.eye(4)
    )
    # excessive recursion depth for this grammar
    super_tree = grammar.make_super_tree(max_recursion_depth=20)
    # Some basic, necessary-but-not-sufficient checks that the
    # tree isn't obviously wrong.
    assert isinstance(super_tree.get_root(), NodeA)
    C = super_tree.find_nodes_by_type(NodeC)[0]
    assert len(super_tree.find_nodes_by_type(NodeF)) == C.max_children
    Bs = super_tree.find_nodes_by_type(NodeB)
    assert len(Bs) == 1
    B = Bs[0]
    assert len(list(super_tree.successors(B))) == 2 # Takes both options

    # Test that recursion depth takes effect by making it too small
    super_tree = grammar.make_super_tree(max_recursion_depth=1)
    assert len(super_tree.find_nodes_by_type(NodeA)) == 1
    assert len(super_tree.find_nodes_by_type(NodeC)) == 1
    assert len(super_tree.find_nodes_by_type(NodeF)) == 0
    
# No depends, but this test is depended on.
@pytest.mark.dependency(name="test_param_prior")
def test_param_prior(set_seed):
    # Generate a grammar with random parameters from
    # their priors.
    grammar = SpatialSceneGrammar(
        root_node_type = NodeA,
        root_node_tf = torch.eye(4),
        sample_params_from_prior=True
    )
    # Scoring should still work identically.
    trace = pyro.poutine.trace(grammar.sample_tree).get_trace()
    tree = trace.nodes["_RETURN"]["value"]
    expected_score = trace.log_prob_sum()
    score = tree.score()
    assert torch.isclose(expected_score, score), "%f vs %f" % (score, expected_score)
    # But the underlying param values should be different from the
    # default values.
    default_grammar = SpatialSceneGrammar(
        root_node_type = NodeA,
        root_node_tf = torch.eye(4),
        sample_params_from_prior=False
    )

    all_identical = True
    for p1, p2 in zip(grammar.parameters(), default_grammar.parameters()):
        if not torch.allclose(p1, p2):
            all_identical = False
            break
    assert not all_identical, "Grammar draw from prior was exactly the same as default grammar."

@pytest.mark.dependency(depends=["test_param_prior"])
def test_save_load_grammar():
    # Generate a grammar with random parameters from
    # their priors. From `test_param_prior`, we know this'll
    # be unique from the default grammar.
    grammar = SpatialSceneGrammar(
        root_node_type = NodeA,
        root_node_tf = torch.eye(4),
        sample_params_from_prior=True
    )
    # Default grammar, for comparison.
    default_grammar = SpatialSceneGrammar(
        root_node_type = NodeA,
        root_node_tf = torch.eye(4),
        sample_params_from_prior=False
    )
    torch.save(grammar, "/tmp/test_saved_grammar.torch")

    loaded_grammar = torch.load("/tmp/test_saved_grammar.torch")
    
    all_identical = True
    for p1, p2 in zip(loaded_grammar.parameters(), default_grammar.parameters()):
        if not torch.allclose(p1, p2):
            all_identical = False
            break
    assert not all_identical, "Grammar draw from prior and saved/loaded was exactly the same as default grammar; didn't load right."

    all_identical = True
    for p1, p2 in zip(loaded_grammar.parameters(), grammar.parameters()):
        if not torch.allclose(p1, p2):
            all_identical = False
            break
    assert all_identical, "Grammar loaded from file had different params than one saved."

    # Make sure loaded grammar still works
    trace = pyro.poutine.trace(loaded_grammar.sample_tree).get_trace()
    tree = trace.nodes["_RETURN"]["value"]
    expected_score = trace.log_prob_sum()
    score = tree.score()
    assert torch.isclose(expected_score, score), "%f vs %f" % (score, expected_score)

def test_attach_reattach():
    grammar = SpatialSceneGrammar(
        root_node_type = NodeA,
        root_node_tf = torch.eye(4),
        sample_params_from_prior=True
    )
    tree = grammar.sample_tree(detach=True)
    before_score = tree.score()
    assert before_score.grad_fn is None
    grammar.update_tree_grammar_parameters(tree)
    after_score = tree.score()
    assert after_score.grad_fn is not None
    after_score.backward()
    assert torch.isclose(before_score, after_score)