import logging
import typing as tp
from collections import OrderedDict, namedtuple
from copy import deepcopy
from dataclasses import dataclass
from functools import partial

import networkx as nx
import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.distributions.constraints as constraints
from pyro.contrib.autoname import name_count, scope

from .nodes import (
    AndNode,
    IndependentSetNode,
    Node,
    OrNode,
    RepeatingSetNode,
    TerminalNode,
)
from .rules import ProductionRule
from .torch_utils import ConstrainedParameter


def get_tree_root(tree):
    # Warning: will infinite loop if this isn't a tree.
    # I don't check...
    root_node = list(tree.nodes)[0]
    while len(list(tree.predecessors(root_node))) > 0:
        root_node = tree.predecessors(root_node)[0]
    return root_node


class SceneTree(nx.DiGraph):
    # Minor additional bookkeeping on top of a digraph:
    # - Keeps track of node - parent rule correspondence
    # - Accessors for observed nodes, nodes by type, + tree root
    # - Score calculation
    # - Optionally can store the Pyro trace of generation of this
    #   tree, for internal use in sampling routines.

    def __init__(self, **kwargs):
        self._trace = None
        super().__init__(self, **kwargs)

    @staticmethod
    def make_from_observed_nodes(observed_nodes):
        """Constructs a SceneTree that contains the observed nodes.
        A scene tree constructed like this will have very limited
        functionality, but can be used for visualization."""
        tree = SceneTree()
        for node in observed_nodes:
            tree.add_node(node)
        return tree

    @property
    def trace(self):
        assert self._trace is not None, "Trace not recorded for this tree."
        return self._trace

    @trace.setter
    def trace(self, trace):
        self._trace = trace

    def get_parent(self, child):
        parents = list(self.predecessors(child))
        if len(parents) == 0:
            return None
        else:
            assert len(parents) == 1
            return parents[0]

    def get_children(self, parent):
        assert parent in self.nodes
        return sorted(list(self.successors(parent)), key=lambda x: x.rule_k)

    def get_children_recursive(self, parent, type_filter: tp.Optional[tp.Type] = None):
        unexpanded = [parent]
        all_children = []
        while len(unexpanded) > 0:
            new_children = self.get_children(unexpanded.pop(0))
            if type_filter is not None:
                new_children = [c for c in new_children if isinstance(c, type_filter)]
            all_children += new_children
            unexpanded += new_children
        return all_children

    def get_rule_for_child(self, parent, child):
        if isinstance(parent, RepeatingSetNode):
            return parent.rule
        elif isinstance(parent, (AndNode, OrNode, IndependentSetNode)):
            return parent.rules[child.rule_k]
        else:
            raise ValueError(
                "Parent %s of child %s is of bad type for getting rules."
                % (parent, child)
            )

    def get_children_and_rules(self, parent):
        """Return the child nodes and their corresponding
        rules (selected from the parent node rule list according
        to the parent type and child rule_k index) as matching lists."""
        children = self.get_children(parent)
        # Get child rule list.
        if isinstance(parent, RepeatingSetNode):
            rules = [parent.rule for k in range(len(children))]
        elif isinstance(parent, AndNode):
            rules = parent.rules
        elif isinstance(parent, OrNode):
            assert len(children) == 1
            rules = [parent.rules[children[0].rule_k]]
        elif isinstance(parent, IndependentSetNode):
            rules = [parent.rules[child.rule_k] for child in children]
        elif isinstance(parent, TerminalNode):
            rules = []
        else:
            raise ValueError("Unexpected node type: ", type(parent))
        return children, rules

    def get_observed_nodes(self):
        # Pulls out only nodes in the tree that are
        # "observable."
        return [n for n in self if n.observed is True]

    def get_root(self):
        # Assumes tree structure; might infinite loop otherwise.
        k = 0
        root = list(self.nodes)[0]
        while len(list(self.predecessors(root))) > 0:
            root = list(self.predecessors(root))[0]
            k += 1
            if k > 10000:
                raise ValueError(">10k iters when finding root. Not a tree?")
        return root

    def find_nodes_by_type(self, node_type):
        return [n for n in self.nodes if isinstance(n, node_type)]

    def score(self, include_discrete=True, include_continuous=True, verbose=False):
        # Compute total score of parents and children.
        total_score = torch.tensor(0.0)
        root = self.get_root()
        for node in self.nodes:
            if node is not root:
                if self.get_parent(node) is None:
                    logging.warning("Orphan non-root node %s detected", node)
                    total_score -= np.inf
                    break
            children = list(self.successors(node))
            if include_discrete:
                contrib = node.score_child_set(children)
                total_score = total_score + contrib
                if verbose:
                    print(node, ": ", contrib.item())
            if include_continuous:
                for child in children:
                    assert child.rule_k is not None
                    assert child.rule_k >= 0
                    if isinstance(node, (AndNode, OrNode, IndependentSetNode)):
                        rule = node.rules[child.rule_k]
                    elif isinstance(node, RepeatingSetNode):
                        rule = node.rule
                    else:
                        raise ValueError(
                            "Unknown node type %s has children." % type(node)
                        )
                    contrib = rule.score_child(node, child, verbose=verbose)
                    total_score = total_score + contrib
                    if verbose:
                        print(node, " -> ", child, ": ", contrib.item())
        return total_score


class SpatialSceneGrammar(torch.nn.Module):
    """
    Manages a scene grammar that produces scene trees by composition
    of subclasses of the node types in this repo.

    The (torch) parameters of the grammar are the set of parameters
    for the node types and their rules:
        - For each node type in the grammar, the parameters of the
            underlying node type. (For an OR node, the rule_probs;
            for a GEOMETRIC_SET_NODE, the stopping probability p.)
        - For each node type in the grammar, the parameters of each of
            that node type's rules. E.g. if a node type uses
            a BoundingBox XYZ offset rule, the lower and upper bound
            of that BoundingBox.
    """

    def __init__(
        self,
        root_node_type,
        root_node_tf,
        sample_params_from_prior=False,
        do_sanity_checks=True,
    ):
        """Given a root node type and its tf, prepares this grammar for use."""
        super().__init__()
        self.root_node_type = root_node_type
        self.root_node_tf = root_node_tf
        self.do_sanity_checks = do_sanity_checks

        # Build the list of all types in the grammar, and use it
        # to set up the grammar parameters initialized from the
        # node default.
        self.all_types = self._collect_all_types_in_grammar()
        self.params_by_node_type = torch.nn.ModuleDict()
        self.rule_params_by_node_type = torch.nn.ModuleDict()
        for node_type in self.all_types:
            # Set up node parameters.
            param_prior = node_type.get_parameter_prior()
            if param_prior is None:
                self.params_by_node_type[node_type.__name__] = None
            else:
                if sample_params_from_prior:
                    init_value = param_prior.sample()
                else:
                    # Grab the default parameter value from an instance
                    # of this node.
                    init_value = node_type(tf=torch.eye(4)).parameters
                    if len(init_value) > 0:
                        assert torch.all(
                            torch.isfinite(param_prior.log_prob(init_value))
                        ), (
                            "Bad initial value / prior match in node type %s"
                            % node_type
                        )
                self.params_by_node_type[node_type.__name__] = ConstrainedParameter(
                    init_value=init_value, constraint=param_prior.support
                )
            # Set up rule parameters.
            rules = node_type.generate_rules()
            rule_params = torch.nn.ModuleList()
            for rule in rules:
                # TODO(gizatt) This process is pretty ugly; moving stuff
                # more into a direction of not having to regenerate rules a bunch
                # of times would be cleaner.
                xyz_param_prior_dict, rot_param_prior_dict = rule.get_parameter_prior()

                if sample_params_from_prior:
                    xyz_param_value_dict = {
                        k: v.sample() for k, v in xyz_param_prior_dict.items()
                    }
                    rot_param_value_dict = {
                        k: v.sample() for k, v in rot_param_prior_dict.items()
                    }
                else:
                    # Grab the default parameter value from the instantiated rule.
                    xyz_param_value_dict, rot_param_value_dict = rule.parameters
                xyz_param_dict = torch.nn.ModuleDict(
                    {
                        k: ConstrainedParameter(
                            init_value=xyz_param_value_dict[k],
                            constraint=xyz_param_prior_dict[k].support,
                        )
                        for k in xyz_param_prior_dict.keys()
                    }
                )
                rot_param_dict = torch.nn.ModuleDict(
                    {
                        k: ConstrainedParameter(
                            init_value=rot_param_value_dict[k],
                            constraint=rot_param_prior_dict[k].support,
                        )
                        for k in rot_param_prior_dict.keys()
                    }
                )
                rule_params.append(
                    torch.nn.ModuleList([xyz_param_dict, rot_param_dict])
                )
            self.rule_params_by_node_type[node_type.__name__] = rule_params

    def print_params(self, node_names=None):
        for node_type in self.all_types:
            if node_names is not None:
                if node_type.__name__ not in node_names:
                    continue
            print("\t%s:" % node_type.__name__)
            constrained_params = self.params_by_node_type[node_type.__name__]
            if constrained_params is not None:
                print(
                    "\t\t%s: %s"
                    % ("child weights", constrained_params().detach().cpu().numpy())
                )
            for k, (xyz_param_dict, rot_param_dict) in enumerate(
                self.rule_params_by_node_type[node_type.__name__]
            ):
                print("\t\tRule %d:" % (k))
                for k, v in xyz_param_dict.items():
                    print("\t\t\tXYZ %s: %s" % (k, v()))
                for k, v in rot_param_dict.items():
                    print("\t\t\tRot %s: %s" % (k, v()))

    def _collect_all_types_in_grammar(self):
        # Similar to supertree logic, but doesn't track supertree.
        # Needs to instantiate nodes to get their rule lists.
        root = self.root_node_type(tf=torch.eye(4))
        all_types = set()
        input_queue = [root]
        while len(input_queue) > 0:
            node = input_queue.pop(0)
            if type(node) in all_types:
                continue
            all_types.add(type(node))
            for rule in node.rules:
                # Having the root type be produced by anything makes it
                # much more awkward to rebuild scene trees in a
                # bottom-up way, as it's not obvious when you're done
                # (since if you hit a node of root type, you *could*
                # be done, or you could keep going...)
                # I don't think this is fundamental, but making sure grammars
                # satisfy this assumption makes my parsing procedures a
                # little easier to write.
                assert (
                    rule.child_type is not self.root_node_type
                ), "The root type shouldn't be produced by any node."
                if rule.child_type not in all_types:
                    input_queue.append(rule.child_type(tf=torch.eye(4)))
        return all_types

    def _set_node_parameters(self, node, detach):
        # Node params
        unconstrained_params = self.params_by_node_type[type(node).__name__]
        if unconstrained_params is not None:
            # Resolve to constrained value, and use that to set up node
            # parameters. Gradients should flow back to this module's
            # torch params.
            constrained_params = unconstrained_params()
            if detach:
                constrained_params = constrained_params.detach()
            node.parameters = constrained_params
        # Rule params
        unconstrained_params_by_rule = self.rule_params_by_node_type[
            type(node).__name__
        ]
        for xyz_and_rot_params, rule in zip(unconstrained_params_by_rule, node.rules):
            # This rule has XYZ and Rotation params. Resolve them to constrained
            # values; then detach if requested, before feeding them to the
            # rule.
            constrained_xyz_params, constrained_rot_params = xyz_and_rot_params
            if detach:
                unconstrained_xyz_params = {
                    k: v().detach() for k, v in constrained_xyz_params.items()
                }
                unconstrained_rot_params = {
                    k: v().detach() for k, v in constrained_rot_params.items()
                }
            else:
                unconstrained_xyz_params = {
                    k: v() for k, v in constrained_xyz_params.items()
                }
                unconstrained_rot_params = {
                    k: v() for k, v in constrained_rot_params.items()
                }
            rule.parameters = (unconstrained_xyz_params, unconstrained_rot_params)

    def sample_tree(self, detach=False):
        # If detach is true, this tree will be detached from having
        # gradients from this grammar's parameters. This is useful in
        # cases where the sampled tree is going to be deepcopied / pickled.
        tree = SceneTree()

        def do_sampling():
            root = self.root_node_type(tf=self.root_node_tf)
            self._set_node_parameters(root, detach=detach)
            tree.add_node(root)
            node_queue = [root]
            k = 0
            while len(node_queue) > 0:
                parent = node_queue.pop(0)
                # Ask node to sample its children.
                with scope(prefix=parent.name):
                    children = parent.sample_children()
                k += 1
                for child in children:
                    self._set_node_parameters(child, detach=detach)
                    tree.add_node(child)
                    tree.add_edge(parent, child)
                    node_queue.append(child)

        tree.trace = pyro.poutine.trace(do_sampling).get_trace()
        return tree

    def update_tree_grammar_parameters(self, tree, detach=False):
        # Given a tree, iterates through its nodes and sets the node
        # parameters to the grammar's parameters for that node type.
        # If a tree has been detached, this will re-attach it.
        for node in tree.nodes:
            self._set_node_parameters(node, detach=detach)

    def make_super_tree_from_root_node_type(
        self,
        root_node_type,
        root_node_tf,
        max_recursion_depth=15,
        detach=False,
        terminate_at_observed=False,
    ):
        # Forms a graph of nodes for which any actual sampled tree would be a subgraph.
        # This form exposes some special cases that can be useful in e.g. parsing:
        #  - Ability to start a supertree from any node type in the grammar.
        #  - Ability to terminate supertree when it hits observed nodes (for use
        #    in top-down node proposal generation.) This'll generate a supertree
        #    whose non-root nodes are all unobserved, and the root node is the
        #    requested root node type.
        tree = SceneTree()

        root_tf = root_node_tf
        if detach:
            root_tf = root_tf.detach()
        root = root_node_type(tf=root_tf)

        self._set_node_parameters(root, detach=detach)
        # Label recursion depth in on nodes of super tree.
        root._recursion_depth = 0
        tree.add_node(root)
        node_queue = [root]
        while len(node_queue) > 0:
            parent = node_queue.pop(0)
            if isinstance(parent, (AndNode, OrNode, IndependentSetNode, TerminalNode)):
                rules = parent.rules
            elif isinstance(parent, RepeatingSetNode):
                rules = [parent.rule for k in range(parent.max_children)]
            else:
                raise ValueError(type(parent))

            for k, rule in enumerate(rules):
                child = rule.sample_child(parent)
                self._set_node_parameters(child, detach=detach)
                child.rule_k = k
                child._recursion_depth = parent._recursion_depth + 1
                if terminate_at_observed is True and child.observed:
                    continue
                tree.add_node(child)
                tree.add_edge(parent, child)
                if child._recursion_depth < max_recursion_depth:
                    node_queue.append(child)
        return tree

    def make_super_tree(self, max_recursion_depth=15, detach=False):
        return self.make_super_tree_from_root_node_type(
            self.root_node_type,
            self.root_node_tf,
            max_recursion_depth=max_recursion_depth,
            detach=detach,
            terminate_at_observed=False,
        )


def apply_decoration_rules_to_tree(tree, decoration_mapping):
    """
    tree: A SceneTree, which will be modified in-place.
    decoration_mapping: A dict mapping node types to
        new node types.

    Randomly applies decoration rules to an existing scene tree.
    For each node in the existing tree, changes its geometry type (if present)
    to fixed to freeze it in place.

    Then looks up each existing node in the tree in the decoration mapping:
    for each node with a mapping, adds a new node to the tree of the mapped
    type at the same TF as the original node.

    Finally, recursively expands the set of new decoration nodes following
    their built-in rules.
    """
    # Freeze existing nodes.
    for node in tree.nodes:
        if node.physics_geometry_info is not None:
            node.physics_geometry_info.fixed = True
    # Build list of new decoration nodes (and their source node).
    # (Delay modification of the original tree.)
    new_pairs = []
    for existing_type, new_type in decoration_mapping.items():
        for existing_node in tree.nodes:
            if isinstance(existing_node, existing_type):
                new_node = new_type(existing_node.tf)
                new_pairs.append((existing_node, new_node))
    # Now add the new nodes to the tree and initialize the expansion queue.
    expand_queue = []
    for existing_node, new_node in new_pairs:
        # This node is *not* connected to its parent, since there is no
        # rule on the parent for explaining this new node.
        tree.add_node(new_node)
        expand_queue.append(new_node)
    # Finish recursive expansion of any nodes we added.
    while len(expand_queue) > 0:
        parent = expand_queue.pop(0)
        # Ask node to sample its children.
        children = parent.sample_children()
        for child in children:
            # Always use default params for the decoration nodes.
            tree.add_node(child)
            tree.add_edge(parent, child)
            expand_queue.append(child)
    return tree
