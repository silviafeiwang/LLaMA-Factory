#!/usr/bin/env python3
"""
Example script to load and analyze samples from the saved pickle file.
"""

import pickle
import os
from typing import List, Dict, Any


def load_samples(pickle_path: str) -> List[Dict[str, Any]]:
    """Load samples from pickle file."""
    with open(pickle_path, "rb") as f:
        samples = pickle.load(f)
    return samples


def analyze_samples(samples: List[Dict[str, Any]]) -> None:
    """Perform basic analysis on the loaded samples."""
    print(f"Total samples loaded: {len(samples)}")
    print(f"Sample structure: {list(samples[0].keys()) if samples else 'No samples'}")

    # Show first few samples
    print("\nFirst 3 samples:")
    for i, sample in enumerate(samples[:3]):
        print(f"  Sample {i}:")
        print(f"    ID: {sample['sample_id']}")
        print(f"    Original Index: {sample['original_index']}")
        print(f"    Data keys: {list(sample['data'].keys())}")
        print(f"    Data preview: {str(sample['data'])[:100]}...")
        print()


def get_sample_by_id(samples: List[Dict[str, Any]], sample_id: int) -> Dict[str, Any]:
    """Get a specific sample by its ID."""
    for sample in samples:
        if sample["sample_id"] == sample_id:
            return sample
    return None


def main():
    # Example usage
    dataset_dir = "data"  # Replace with your actual dataset directory
    pickle_filename = (
        "your_dataset_selected_samples.pkl"  # Replace with actual filename
    )
    pickle_path = os.path.join(dataset_dir, pickle_filename)

    if not os.path.exists(pickle_path):
        print(f"Pickle file not found: {pickle_path}")
        print(
            "Make sure to run the training/data loading first to generate the pickle file."
        )
        return

    # Load samples
    samples = load_samples(pickle_path)

    # Analyze samples
    analyze_samples(samples)

    # Get specific sample by ID
    sample_5 = get_sample_by_id(samples, 5)
    if sample_5:
        print(f"Sample with ID 5: {sample_5}")

    # You can add more analysis here
    # For example:
    # - Count samples by some criteria
    # - Extract specific fields
    # - Generate statistics
    # - Create visualizations


if __name__ == "__main__":
    main()
