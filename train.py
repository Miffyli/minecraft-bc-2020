# Simple env test.
import logging
import os

import gym
import minerl

import coloredlogs
coloredlogs.install(logging.DEBUG)

# All the evaluations will be evaluated on MineRLObtainDiamond-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLObtainDiamondVectorObf-v0')
# You need to ensure that your submission is trained in under MINERL_TRAINING_MAX_STEPS steps
MINERL_TRAINING_MAX_STEPS = int(os.getenv('MINERL_TRAINING_MAX_STEPS', 8000000))
# You need to ensure that your submission is trained by launching less than MINERL_TRAINING_MAX_INSTANCES instances
MINERL_TRAINING_MAX_INSTANCES = int(os.getenv('MINERL_TRAINING_MAX_INSTANCES', 5))
# You need to ensure that your submission is trained within allowed training time.
# Round 1: Training timeout is 15 minutes
# Round 2: Training timeout is 4 days
MINERL_TRAINING_TIMEOUT = int(os.getenv('MINERL_TRAINING_TIMEOUT_MINUTES', 4 * 24 * 60))
# The dataset is available in data/ directory from repository root.
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')


# My variables
HDF5_DATA_FILE = "train/data.hdf5"
HDF5_DATA_FILE_FRAMESKIPPED = "train/data_frameskipped.hdf5"
ACTION_CENTROIDS_FILE = "train/action_centroids.npy"
TRAINED_MODEL_PATH = "train/trained_model.th"


def main():
    """
    This function will be called for training phase.
    """
    from utils.handle_dataset import store_subset_to_hdf5, remove_frameskipped_samples
    from wrappers.action_wrappers import fit_kmeans, update_hdf5_with_centroids
    from train_bc_lstm import main as main_train_bc
    from train_bc_lstm import parser as train_bc_parser

    os.makedirs("train", exist_ok=True)

    # Turn dataset into HDF5 for clustering actions
    store_subset_to_hdf5_params = [
        MINERL_DATA_ROOT,
        HDF5_DATA_FILE,
        "--subset-names",
        "MineRLTreechopVectorObf-v0",
        "MineRLObtainIronPickaxeVectorObf-v0",
        "MineRLObtainDiamondVectorObf-v0"
    ]
    store_subset_to_hdf5(store_subset_to_hdf5_params)

    # Fit Kmeans on actions from all three datasets
    # Suuuuuper-elegant argument passing, thanks
    # to the big-brain use of argparse
    kmean_params = [
        HDF5_DATA_FILE,
        ACTION_CENTROIDS_FILE,
        "--n-clusters", "150",
        "--n-init", "30"
    ]
    fit_kmeans(kmean_params)

    # Turn dataset into HDF5 for training (no ObtainDiamond)
    store_subset_to_hdf5_params = [
        MINERL_DATA_ROOT,
        HDF5_DATA_FILE,
        "--subset-names", 
        "MineRLTreechopVectorObf-v0",
        "MineRLObtainIronPickaxeVectorObf-v0",
    ]
    store_subset_to_hdf5(store_subset_to_hdf5_params)

    # Update centroid locations in the data
    update_hdf5_params = [
        HDF5_DATA_FILE,
        ACTION_CENTROIDS_FILE
    ]
    update_hdf5_with_centroids(update_hdf5_params)

    # Remove frameskipped frames for LSTM training
    removed_frameskipped_params = [
        HDF5_DATA_FILE,
        HDF5_DATA_FILE_FRAMESKIPPED
    ]
    remove_frameskipped_samples(removed_frameskipped_params)

    # Train model with behavioural cloning
    bc_train_params = [
        HDF5_DATA_FILE_FRAMESKIPPED,
        TRAINED_MODEL_PATH,
        "--num-epochs", "256",
        "--include-frameskip", "16",
        "--discrete-actions",
        "--num-discrete-actions", "150",
        "--frameskip-from-vector",
        "--batch-size", "32",
        "--lr", "0.0000625",
        "--weight-decay", "1e-5",
        "--seq-len", "32",
        "--horizontal-flipping",
        "--entropy-weight", "0.0",
        "--resnet", "ResNetHeadFor64x64DoubleFilters"
    ]
    parsed_args, remaining_args = train_bc_parser.parse_known_args(bc_train_params)
    main_train_bc(parsed_args, remaining_args)

if __name__ == "__main__":
    main()
