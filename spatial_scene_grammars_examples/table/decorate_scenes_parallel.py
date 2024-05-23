import os
import pickle
import time
from typing import List

import numpy as np
import torch
from tqdm import tqdm

torch.set_default_tensor_type(torch.DoubleTensor)
from datetime import timedelta

from spatial_scene_grammars.constraints import *
from spatial_scene_grammars.dataset import *
from spatial_scene_grammars.nodes import *
from spatial_scene_grammars.parameter_estimation import *
from spatial_scene_grammars.parsing import *
from spatial_scene_grammars.rules import *
from spatial_scene_grammars.sampling import *
from spatial_scene_grammars.scene_grammar import *
from spatial_scene_grammars.visualization import *
from spatial_scene_grammars_examples.table.grammar import *
from spatial_scene_grammars_examples.table.grammar_decoration import *

torch.set_num_threads(1)


def load_dataset(dataset_pickle_path):
    assert dataset_pickle_path.endswith(".pickle")

    target_dataset_trees: List[SceneTree] = []

    pbar = tqdm(desc="Loading scenes")
    with open(dataset_pickle_path, "rb") as f:
        while 1:
            try:
                tree = pickle.load(f)
                if isinstance(tree, SceneTree):
                    target_dataset_trees.append(tree)
                elif isinstance(tree, tuple):
                    target_dataset_trees.append(tree[0])
                else:
                    raise ValueError(f"Unexpected type {type(tree)}")
                pbar.update(1)
            except EOFError:
                break
    pbar.close()

    return target_dataset_trees


def split_list(data, num_chunks):
    total_length = len(data)

    # Calculate base chunk size and the remainder
    base_size = total_length // num_chunks
    remainder = total_length % num_chunks

    # Calculate sizes for each chunk
    chunk_sizes = [
        base_size + 1 if i < remainder else base_size for i in range(num_chunks)
    ]

    # Split data into chunks
    chunks = []
    start = 0
    for size in chunk_sizes:
        end = start + size
        chunks.append(data[start:end])
        start = end

    return chunks


def replace_lids(tree, p=0.8):
    """
    Replace lids with Null nodes with probability p. This enables us to see the food
    when decorating the scenes.
    """
    to_remove = [n for n in tree if isinstance(n, SteamerTop)]
    for node in to_remove:
        if np.random.random() <= p:
            parent = tree.get_parent(node)
            tree.remove_node(node)
            replacement_child = Null(tf=parent.tf)
            replacement_child.rule_k = 2
            tree.add_edge(parent, replacement_child)
    return tree


def decorate_tree(tree):
    decorated_tree = replace_lids(tree)
    decorated_tree = apply_decoration_rules_to_tree(decorated_tree, decoration_mapping)
    projected_tree = project_tree_to_feasibility(decorated_tree, do_forward_sim=True)
    return projected_tree


def save_tree(tree, dataset_save_file):
    if tree is None:
        print("Tree is None, skipping save.")
        return
    with open(dataset_save_file, "a+b") as f:
        pickle.dump(tree, f)


def main():
    dataset_load_file = "dimsum_noStackConstraints_10k.pickle"
    decorated_dataset_save_file = "dimsum_decorated_noStackConstraints_10k.pickle"

    assert os.path.exists(dataset_load_file), "Dataset file doesn't exists!"
    assert not os.path.exists(
        decorated_dataset_save_file
    ), "Decorate dataset file already exists!"

    start = time.time()

    dataset = load_dataset(dataset_load_file)

    # NOTE: Somehow can't parallelize due to memory issues
    for tree in tqdm(dataset, desc="Decorating dataset"):
        decorated_tree = decorate_tree(tree)
        save_tree(decorated_tree, decorated_dataset_save_file)

    print(f"Decorating dataset took {timedelta(seconds=time.time()-start)}")


if __name__ == "__main__":
    main()
