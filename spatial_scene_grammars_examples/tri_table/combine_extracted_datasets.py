"""
Script for combining multiple datasets that have been produced by `extract_dataset.py`.
"""

import os
import argparse
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_pickle_folder_path",
        type=str,
        help="The path to the folder containing the dataset pickle files to combine.",
    )
    parser.add_argument(
        "output_pickle_path",
        type=str,
        help="The path to save the combined dataset to.",
    )
    args = parser.parse_args()
    path = args.dataset_pickle_folder_path
    out_path = args.output_pickle_path

    assert out_path.endswith(".pkl") or out_path.endswith(".pickle")

    # Read all pickle files in the folder.
    datasets = []
    for file in os.listdir(path):
        if file.endswith(".pkl") or file.endswith(".pickle"):
            with open(os.path.join(path, file), "rb") as f:
                dataset = pickle.load(f)
            # Check that format is correct.
            assert all(
                isinstance(data, list) for data in dataset
            ), "Expected list of lists."
            scene_sample = dataset[0]
            obj_sample = scene_sample[0]
            assert (
                isinstance(obj_sample, dict)
                and "transform" in obj_sample
                and "model_path" in obj_sample
            )
            datasets.append(dataset)
    print(f"Read {len(datasets)} datasets.")

    # Combine datasets.
    combined_dataset = []
    for dataset in datasets:
        combined_dataset.extend(dataset)

    print(f"Combined dataset has {len(combined_dataset)} examples.")

    # Save combined dataset.
    with open(out_path, "wb") as f:
        pickle.dump(combined_dataset, f)


if __name__ == "__main__":
    main()
