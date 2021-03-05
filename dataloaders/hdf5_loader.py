#
# A dataloader object for sampling data
# stored in a HDF5 file.
# No other dependencies assumed.
#
import random
from collections import OrderedDict
from multiprocessing import Queue, Process

import numpy as np
import h5py


class HDF5Loader():
    """
    Class loading data from HDF5 files.
    Assumes we have bunch of datasets in the HDF5 file
    with shape (N, ...), where N is number of samples
    in the dataset.

    dataset_mappings tells which arrays should be accessed
    and read.
    """

    def __init__(self, path_to_file, dataset_mappings):
        """
        Parameters:
            path_to_file (str): Path to the HDF5 file to be read
            dataset_mappings (dict str -> (str, fn)): Keys are dataset names
                to be loaded, and values are tuples of names they will
                be returned as along with a function that will be applied
                on each individual sample or slice.
        """
        self.file = h5py.File(path_to_file, "r")
        # Just in case
        self.dataset_mappings = OrderedDict(dataset_mappings)
        self.datasets = [self.file[dataset] for dataset in dataset_mappings.keys()]
        self.dataset_types = [
            {"shape": self.file[dataset].shape[1:], "dtype": self.file[dataset].dtype} for dataset in dataset_mappings.keys()
        ]
        # Assume all datasets have equal amount of samples
        # Eheheh....
        self.num_samples = self.datasets[0].shape[0]

    def get_types(self):
        """
        Returns a dictionary mapping dataset_mappings to
        shapes and dtypes they will return

        Returns:
            Dictionary mapping dataset names to {"shape": shape, "dtype": dtype)
        """
        return dict(
            (name_and_fn[0], types) for name_and_fn, types in zip(self.dataset_mappings.values(), self.dataset_types)
        )

    def __len__(self):
        return self.num_samples

    def get_indeces(self, idxs):
        """
        Return data for idxs in a dictionary
        mapping from dataset name to data returned

        Parameters:
            idxs (ndarray of ints): Indexes to read
        Returns:
            Dictionary mapping dataset names -> data requested
        """
        return_dict = {}
        # Idxs must be in increasing order
        idxs = np.sort(idxs)
        for name_and_fn, dataset in zip(self.dataset_mappings.values(), self.datasets):
            # Avoid lazy-loading by forcing copy
            return_values = dataset[idxs].copy()
            fn = name_and_fn[1]
            # If we have a function, map all items through it
            if fn is not None:
                new_values = []
                for return_value in return_values:
                    new_values.append(fn(return_value))
                return_values = np.array(new_values)
            return_dict[name_and_fn[0]] = return_values
        return return_dict

    def get_slices(self, slices, pad_to):
        """
        Like get_items but for slices. Return data for slices
        in a dictionary mapping from dataset to data returned.
        The returned arrays are of shape [pad_to, len(slices), ...],
        where arrays are padded with NaNs where necessary (from the end).

        Parameters:
            slices (tuple of tuples): List of (start_ind, end_ind)
                slices to return
            pad_to (int): Pad arrays to this length
        """
        return_dict = {}

        for name_and_fn, dataset in zip(self.dataset_mappings.values(), self.datasets):
            return_array = []
            for start_ind, end_ind in slices:
                # Avoid lazy-loading by forcing copy
                return_values = dataset[start_ind:end_ind].copy()
                fn = name_and_fn[1]
                # If we have a function, map all items through it
                if fn is not None:
                    # For slices, apply once per slice
                    return_values = fn(return_values)
                # Checking padding
                if return_values.shape[0] < pad_to:
                    # For first axis
                    padding_needed = pad_to - return_values.shape[0]
                    # For the rest
                    zero_padding = [(0, 0)] * (return_values.ndim - 1)
                    return_values = np.pad(
                        return_values,
                        [(0, padding_needed), ] + zero_padding,
                        constant_values=np.nan
                    )
                return_array.append(return_values)
            # Put batch-dim on second axis and timesteps on first.
            return_array = np.stack(return_array, axis=1)
            return_dict[name_and_fn[0]] = return_array
        return return_dict

    def close(self):
        self.file.close()

    def __del__(self):
        self.close()


def hdf5_loader_worker(hdf5_file_path, dataset_mappings, task_queue, result_queue, num_tasks):
    """
    A worker intended to be used a Process, which reads tasks
    from task_queue and puts results in the result_queue, until
    num_tasks tasks have been done.

    Tasks can be of following types:
        np.ndarray: Assume it is plain indeces which to load,
            and returns them, returning [B, ...] arrays.
        tuple: Assumes it is a tuple (traj_length, slices, arr),
            which returns slices of trajectories of length
            traj_length, padded with zeros in the end if
            not long enough. Returns ([T, B, ...], arr), where
            arr is given in the task (e.g. episode indeces).
    """
    hdf5_loader = HDF5Loader(hdf5_file_path, dataset_mappings)
    num_tasks_done = 0
    while num_tasks_done < num_tasks:
        task = task_queue.get(timeout=60)
        if isinstance(task, np.ndarray):
            # Assume plain indeces
            return_items = hdf5_loader.get_indeces(task)
            result_queue.put(return_items, timeout=60)
        elif isinstance(task, tuple):
            # Assume [traj_length, slices, arr]
            traj_length, slices, copy_arr = task
            return_array = hdf5_loader.get_slices(slices, pad_to=traj_length)
            result_queue.put((return_array, copy_arr), timeout=60)
        num_tasks_done += 1
    hdf5_loader.close()


class HDF5RandomSampler:
    """
    An iterator over HDF5Loader that will return
    random items.
    """

    def __init__(
        self,
        hdf5_file_path,
        dataset_mappings,
        batch_size,
        num_epochs,
        queue_size=200,
        num_workers=1,
        valid_indeces_mask=None
    ):
        """
        Parameters:
            hdf5_file_path (str): Path to the HDF5 file to be read
            dataset_mappings (List of str): Datasets to be loaded
                (see HDF5Loader)
            batch_size (int): Number of items to return per batch
            num_epochs (int): Number of times to iterate over the
                dataset
            valid_indeces_mask (None or str): If str, use this
                dataset from hdf5 file to load a boolean mask,
                where only actions with 1/True are included
                in training.
            queue_size (int): Number of items that can be queued at once
        """
        self.hdf5_file_path = hdf5_file_path
        self.dataset_mappings = dataset_mappings
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.queue_size = queue_size
        self.num_workers = num_workers

        # np array of valid indeces that can be read/returned.
        # This will be shuffled upon new epoch to return new batches
        self.valid_indeces = None
        hdf5_file = h5py.File(self.hdf5_file_path, "r")
        if valid_indeces_mask is not None:
            # Need to find indeces of new actions
            valid_indeces_mask = hdf5_file[valid_indeces_mask][:].copy().ravel()
            self.valid_indeces = np.where(valid_indeces_mask)[0]
        else:
            # All indeces are valid
            self.valid_indeces = np.arange(hdf5_file["actions/vector"].shape[0])
        hdf5_file.close()

        self.num_samples = len(self.valid_indeces)
        self.num_batches = (self.num_samples * num_epochs) // batch_size

        np.random.shuffle(self.valid_indeces)

        self.current_idx = 0
        self.epoch_num = 0

        self.task_queue = Queue(self.queue_size)
        self.result_queue = Queue(self.queue_size)

        # Prefill tasks queue
        for i in range(self.queue_size):
            self._enqueue_next_batch()

        self.loader_processes = []
        for i in range(self.num_workers):
            worker = Process(
                target=hdf5_loader_worker,
                args=(
                    self.hdf5_file_path, self.dataset_mappings,
                    self.task_queue, self.result_queue,
                    self.num_batches
                ),
                daemon=True
            )
            worker.start()
            self.loader_processes.append(worker)

    def _enqueue_next_batch(self):
        """
        Put next random indeces up for loading. Return
        True if iteration continues (there are still tasks remaining),
        otherwise return False
        """
        if (self.current_idx + self.batch_size) >= self.num_samples:
            # Increment epoch
            self.epoch_num += 1
            if self.epoch_num == self.num_epochs:
                return False
            self.current_idx = 0
            np.random.shuffle(self.valid_indeces)
        indeces = self.valid_indeces[self.current_idx:self.current_idx + self.batch_size]
        self.current_idx += self.batch_size
        self.task_queue.put(indeces, timeout=60)
        return True

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_batches

    def __next__(self):
        iteration_continues = self._enqueue_next_batch()
        if not iteration_continues:
            # TODO this will skip out some of the final
            #      batches
            raise StopIteration
        results = self.result_queue.get(timeout=60)
        return results


class HDF5SequenceSampler:
    """
    An iterator that goes over the episodes in order,
    returning fixed length parts of the trajectories.
    Used with RNN networks.

    Returns items like HDF5Loader, i.e. dictionaries
    with name mapping provided in dataset_mappings.

    Returns two items:
        - Dictionary of the data as requested by dataset_mappings,
          each value being array of shape (T, B, ...). If arrays
          need padding to traj_length, they are padded from the end
          with NaNs.
        - ndarray of shape (B,), which tells from which episode sample
          was taken.
    """

    def __init__(
        self,
        hdf5_file_path,
        dataset_mappings,
        batch_size,
        num_epochs,
        traj_length=100,
        queue_size=50,
        randomize_sampling=False
    ):
        """
        Parameters:
            hdf5_file_path (str): Path to the HDF5 file to be read
            dataset_mappings (List of str): Datasets to be loaded
                (see HDF5Loader)
            batch_size (int): Number of items to return per batch
            num_epochs (int): Number of times to iterate over the
                dataset
            traj_length (int): Number of times to iterate over the
                dataset
            queue_size (int): Number of items that can be queued at once
            randomize_sampling (bool): If True, do not read trajectories
                in a sequence, rather pick a random spot each time.
        """
        self.hdf5_file_path = hdf5_file_path
        self.dataset_mappings = dataset_mappings
        self.batch_size = batch_size
        self.traj_length = traj_length
        self.num_epochs = num_epochs
        self.queue_size = queue_size
        self.randomize_sampling = randomize_sampling

        # Get numer of samples and where episodes start
        hdf5_file = h5py.File(self.hdf5_file_path, "r")
        self.num_samples = hdf5_file["observations/pov"].shape[0]
        self.episode_start_indeces = hdf5_file["episodes/episode_starts"][:].copy().ravel()
        self.episode_end_indeces = np.append(self.episode_start_indeces[1:], self.num_samples)
        hdf5_file.close()

        self.num_episodes = len(self.episode_start_indeces)
        assert self.batch_size <= self.num_episodes, "Batch size {} can not be larger than number of episodes {}".format(
            self.batch_size,
            self.num_episodes
        )
        self.num_tasks = (self.num_samples * num_epochs) // (batch_size * traj_length)
        self.tasks_returned = 0

        # Keep track of where we are moving for each episode.
        # This array is used for picking random episodes
        self.episode_ids = np.arange(self.num_episodes)
        self.current_episode_indeces = self.episode_start_indeces.copy()

        self.task_queue = Queue(self.queue_size)
        self.result_queue = Queue(self.queue_size)

        # Prefill tasks queue
        for i in range(self.queue_size):
            self._enqueue_next_batch()

        # Just one loader process to keep ordering
        self.loader_process = Process(
            target=hdf5_loader_worker,
            args=(
                self.hdf5_file_path, self.dataset_mappings,
                self.task_queue, self.result_queue,
                self.num_tasks
            ),
            daemon=True
        )
        self.loader_process.start()

    def _enqueue_next_batch(self):
        """
        Put next indeces for loading data, increment the
        current location of episodes we are at
        and randomly sample new episodes to sample.
        """
        # Pick random episodes to return
        np.random.shuffle(self.episode_ids)
        random_episode_indeces = self.episode_ids[:self.batch_size]
        # Gather tuples (start_ind, end_ind)
        slices = []
        for i in random_episode_indeces:
            # Randomize start index if we want randomized sampling
            start_ind = None
            if self.randomize_sampling:
                start_ind = random.randint(self.episode_start_indeces[i], self.episode_end_indeces[i] - 1)
            else:
                start_ind = self.current_episode_indeces[i]
            end_ind = start_ind + self.traj_length

            # Check if we are crossing to other episode
            if end_ind >= self.episode_end_indeces[i]:
                # Adjust end slice (note that this index is not included in slice)
                end_ind = self.episode_end_indeces[i]
                # Reset this episode counter back to beginning,
                # plus some random offset to avoid doing always same samples
                random_offset = np.random.randint(low=0, high=self.traj_length)
                self.current_episode_indeces[i] = self.episode_start_indeces[i] + random_offset
            else:
                # Move current position to end index
                self.current_episode_indeces[i] = end_ind
            slices.append((start_ind, end_ind))
        # Add how many samples each trajectory should have
        task = (self.traj_length, slices, random_episode_indeces)
        self.task_queue.put(task, timeout=60)

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_tasks

    def __next__(self):
        if self.tasks_returned == self.num_tasks:
            raise StopIteration
        self._enqueue_next_batch()
        results = self.result_queue.get(timeout=60)
        self.tasks_returned += 1
        return results

    def close(self):
        self.task_queue.close()
        self.result_queue.close()
        self.loader_process.join(timeout=5)

    def __del__(self):
        self.close()
