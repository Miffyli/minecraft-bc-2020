#
# Tools for modifying/handling the MineRL dataset (offline)
#
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm
import h5py
import gym
import minerl


def store_actions_to_numpy_file(subset_name, data_dir, output_file, num_workers=4):
    """
    Take all actions stored in the MineRL dataset under subset,
    gather them into big array and store in output_file

    Parameters:
        subset_name (str): Name of the dataset to process (e.g. MineRLTreechopVectorObf-v0)
        data_dir (str): Where MineRL dataset is stored.
        output_file (str): Where to store all the actions
        num_workers (int): Number of workers for data loader (default 4)
    """
    assert "Obf" in subset_name, "Must be obfuscated environment"

    data = minerl.data.make(subset_name, data_dir=data_dir, num_workers=num_workers)

    all_actions = []
    for _, actions, _, _, _ in tqdm(data.batch_iter(num_epochs=1, batch_size=32, seq_len=1)):
        all_actions.append(actions["vector"].squeeze())

    all_actions = np.concatenate(all_actions, axis=0)

    np.save(output_file, all_actions)


def store_subset_to_hdf5(remaining_args):
    """
    For "VectorObf" envs only!

    Convert all the samples from a datasets into a single HDF5
    file with different datasets for different variables:
        /observations
          /{any keys in the obs-space dict}
          ...
        /actions
          /vector
          /num_action_repeated (number of times action is repeated, 1 being smallest)
          /start_of_new_action (1 if this index is start of new action)
        /rewards                     (reward from executing action in last step)
        /dones
        /reward_to_go                (aka undiscounted return)
        /discounted_returns_.99      (returns discounted with gamma=0.99)
        /episode_times               (timesteps played in the episode)
        /episodes/episode_starts     (where new episodes start)

    TODO There is no guarantee that the stored data will be sequential, or
         figure out a way to store data per episode.
         NOTE: Apparently this works like this
    """
    parser = ArgumentParser("Convert MineRL dataset into HDF5 file")
    parser.add_argument("--subset-names", type=str, required=True, nargs="+", help="Name of the dataset to convert")
    parser.add_argument("data_dir", type=str, help="Location of MineRL dataset")
    parser.add_argument("output_file", type=str, help="Location where HDF5 file should be stored")
    args = parser.parse_args(remaining_args)
    subset_names = args.subset_names
    data_dir = args.data_dir
    output_file = args.output_file

    assert all(map(lambda x: "Obf" in x, subset_names)), "Environments must be Obf envs"

    datas = [minerl.data.make(subset_name, data_dir=data_dir, num_workers=1) for subset_name in subset_names]

    # First measure how many observations we have
    num_observations = 0
    for data in datas:
        for _, _, rewards, _, _ in tqdm(data.batch_iter(num_epochs=1, batch_size=1, seq_len=64), desc="size"):
            num_observations += rewards.shape[1]

    print("Total count of observations: {}".format(num_observations))

    # Assume that obs/action_spaces are dicts with only one depth
    obs_keys = list(data.observation_space.spaces.keys())
    obs_spaces = list(data.observation_space.spaces.values())
    act_space = data.action_space.spaces["vector"]

    # Create unified list of dataset names and their spaces
    dataset_keys = (
        list(map(lambda x: "observations/" + x, obs_keys)) +
        [
            "actions/vector",
            "actions/num_action_repeated",
            "actions/start_of_new_action",
            "rewards",
            "dones",
            "reward_to_go",
            "discounted_returns_.99",
            "episode_times"
        ]
    )
    dataset_spaces = (
        obs_spaces +
        [
            act_space,
            # Number of times action is repeated and where new actions begin
            gym.spaces.Box(shape=(1,), dtype=np.uint8, low=0, high=255),
            gym.spaces.Box(shape=(1,), dtype=np.uint8, low=0, high=1),
            # Reward and dones
            gym.spaces.Box(shape=(1,), dtype=np.float32, low=-np.inf, high=np.inf),
            gym.spaces.Box(shape=(1,), dtype=np.uint8, low=0, high=1),
            # Returns
            gym.spaces.Box(shape=(1,), dtype=np.float32, low=-np.inf, high=np.inf),
            gym.spaces.Box(shape=(1,), dtype=np.float32, low=-np.inf, high=np.inf),
            # Episode time
            gym.spaces.Box(shape=(1,), dtype=np.uint32, low=0, high=np.inf),
        ]
    )
    datasets = {}

    # Create HDF5 file
    store_file = h5py.File(output_file, "w")
    # Create datasets
    for key, space in zip(dataset_keys, dataset_spaces):
        shape = (num_observations,) + space.shape
        dataset = store_file.create_dataset(key, shape=shape, dtype=space.dtype)
        datasets[key] = dataset

    # Read through dataset again and store items
    idx = 0
    episode_time = 0
    last_action = None
    action_repeat_num = 0
    # Keep track where episodes start
    episode_starts = [0]

    for data in datas:
        for observations, actions, rewards, _, dones in tqdm(data.batch_iter(batch_size=1, num_epochs=1, seq_len=64), desc="store"):
            # Careful with the ordering of things here...
            # Iterate over seq len (second dim)
            for i in range(rewards.shape[1]):
                # Store different observations
                for key in obs_keys:
                    datasets["observations/{}".format(key)][idx] = observations[key][0, i]
                # Store action and measure how often they are repeated
                action = actions["vector"][0, i]
                datasets["actions/vector"][idx] = actions[key][0, i]
                # Check if action changed
                if last_action is None or not np.allclose(last_action, action):
                    if last_action is not None:
                        # Store how often the last action was repeated,
                        # each index telling how many times that action was
                        # going to be executed in future.
                        for j in range(1, action_repeat_num + 1):
                            datasets["actions/num_action_repeated"][idx - j] = j
                    last_action = action
                    action_repeat_num = 1
                else:
                    action_repeat_num += 1
                datasets["actions/start_of_new_action"][idx] = 1 if action_repeat_num == 1 else 0
                datasets["episode_times"][idx] = episode_time
                episode_time += 1
                # Store other stuff
                datasets["rewards"][idx] = rewards[0, i]
                datasets["dones"][idx] = np.uint8(dones[0, i])
                if dones[0, i]:
                    episode_time = 0
                    last_action = None
                    episode_starts.append(idx + 1)

                idx += 1

    # Handle returns
    reward_to_go = 0
    current_return = 0
    dataset_reward_to_go = datasets["reward_to_go"]
    dataset_return = datasets["discounted_returns_.99"]
    rewards = datasets["rewards"]
    dones = datasets["dones"]
    for i in tqdm(range(idx - 1, -1, -1), desc="post-process"):
        if dones[i]:
            reward_to_go = 0
            current_return = 0
        dataset_reward_to_go[i] = reward_to_go
        dataset_return[i] = current_return
        reward_to_go += rewards[i]
        current_return = current_return * 0.99 + rewards[i]

    # Add episode_starts dataset
    episode_starts_np = store_file.create_dataset("episodes/episode_starts", shape=(len(episode_starts),), dtype=np.int64)
    # Manual copy because otherwise did not seem
    # to want to copy stuff
    for i in range(len(episode_starts)):
        episode_starts_np[i] = episode_starts[i]

    print("{} experiences moved into hdf5 file".format(idx))

    store_file.close()


def remove_frameskipped_samples(remaining_args):
    """
    Remove all samples from hdf5 dataset that are not new actions (i.e. they are frameskipped).

    Save a new dataset in output_file
    """
    parser = ArgumentParser("Remove frameskipped samples from MineRL HDF5 dataset")
    parser.add_argument("input_file", type=str, help="Location of HDF5 file that should be loaded")
    parser.add_argument("output_file", type=str, help="Location where HDF5 file should be stored")
    parser.add_argument(
        "--new-action-dataset",
        type=str,
        default="actions/start_of_new_action",
        help="Which boolean dataset will be used to determine which samples to keep"
    )
    args = parser.parse_args(remaining_args)
    input_file = args.input_file
    output_file = args.output_file
    new_action_dataset = args.new_action_dataset

    input_file = h5py.File(input_file, "r")
    output_file = h5py.File(output_file, "w")

    # Get number of new samples
    new_action_array = input_file[new_action_dataset][:, 0]
    num_samples = new_action_array.sum()
    print("Original num of samples:", new_action_array.shape[0])
    print("New num of samples:", num_samples)

    dataset_items = []
    input_file.visititems(lambda k, v: dataset_items.append((k, v)))
    # Only keep dataset items
    dataset_items = [(k, v) for k, v in dataset_items if isinstance(v, h5py.Dataset)]

    # Keep track where episodes start
    episode_starts = [0]
    # Create new datasets
    for dataset_name, old_dataset in tqdm(dataset_items, desc="dataset"):

        if old_dataset.shape[0] != len(new_action_array):
            print("Skipping {}. Has different amount of samples than expected.".format(dataset_name))
            new_dataset = output_file.create_dataset(dataset_name, dtype=old_dataset.dtype, shape=old_dataset.shape)
            new_dataset[:] = old_dataset[:]
        else:
            new_shape = [num_samples] + list(old_dataset.shape[1:])
            new_dataset = output_file.create_dataset(dataset_name, dtype=old_dataset.dtype, shape=new_shape)

            # Do step-by-step because other indexing/masking tricks did not speed
            # up or work as expected
            # Special handling for "dones" array
            if dataset_name == "dones":
                new_index = 0
                for old_index in tqdm(range(old_dataset.shape[0]), desc="item", leave=False):
                    if new_action_array[old_index]:
                        new_dataset[new_index] = old_dataset[old_index]
                        new_index += 1
                    # Also make sure that all done=True are included in the new
                    # dataset. We might skip some done=True with above check.
                    if old_dataset[old_index] == 1:
                        new_dataset[new_index - 1] = 1
                        episode_starts.append(new_index)
            else:
                new_index = 0
                for old_index in tqdm(range(old_dataset.shape[0]), desc="item", leave=False):
                    if new_action_array[old_index]:
                        new_dataset[new_index] = old_dataset[old_index]
                        new_index += 1
            # We should have reached end by now
            assert new_index == num_samples, "Copied {} items but should have copied {}".format(new_index, num_samples)

    # Update episode_starts array
    # Manual copy because otherwise did not seem
    # to want to copy stuff
    assert len(episode_starts) == output_file["episodes/episode_starts"].shape[0], "Number of episodes does not match"
    for i in range(len(episode_starts)):
        output_file["episodes/episode_starts"][i] = episode_starts[i]

    input_file.close()
    output_file.close()


AVAILABLE_OPERATIONS = {
    "minerl-to-hdf5": store_subset_to_hdf5,
    "remove-frameskipped": remove_frameskipped_samples,
}

if __name__ == "__main__":
    parser = ArgumentParser("Create and modify HDF5 datasets")
    parser.add_argument("operation", choices=list(AVAILABLE_OPERATIONS.keys()), help="Operation to run")
    args, unparsed_args = parser.parse_known_args()

    operation_fn = AVAILABLE_OPERATIONS[args.operation]
    operation_fn(unparsed_args)
