from argparse import ArgumentParser

import numpy as np
from gym import spaces, Wrapper


class CentroidActions:
    """
    Convert the continuous action vectors into a Discrete
    space by using centroids to represent specific actions,
    and transforming human actions to the one with closest match.

    "Vector Quantization" would probably be a better name for this...
    """

    REQUIRED_ARGUMENTS = ["centroids"]

    def __init__(self, centroids="None"):
        """
        centroids: A (N, D) array of N different centroid vectors,
            or str path pointing at numpy file which contains
            the actions
        """
        if isinstance(centroids, str):
            centroids = np.load(centroids)
        self.centroids = centroids
        self.num_actions = centroids.shape[0]

        # Define action space here
        # as well for training (gym.spaces just happens
        # to be convenient for this purpose)
        self.action_space = spaces.Discrete(n=self.num_actions)

    def to_minerl(self, action):
        """
        Turn the action (int) into a vector for
        MineRL Obf action

        Parameters:
            action (int): Action chosen
        Returns
            ndarray of (D,), representing action to be taken
        """
        return self.centroids[action]

    def from_minerl(self, action_vector):
        """
        Turn MineRL action vector into discrete
        by finding the closest centroid

        Parameters:
            action_vector (ndarray): Array of shape (D,)
        Returns
            action (int) of nearest action
        """
        # TODO fixed (Euclidian distance).
        #      Consider others (e.g. mahanalobis)
        distances = np.sum((self.centroids - action_vector) ** 2, axis=1)
        action = np.argmin(distances)
        return action


class CentroidActionsWrapper(Wrapper):
    """
    CentroidActions, but as a wrapper.
    Hardcoded for MineRL Obf envs.

    Turn the continuous (latent) action-space
    into Discrete space by using closest vector
    """

    REQUIRED_ARGUMENTS = ["centroids"]

    def __init__(self, env, centroids="None"):
        """
        centroids: See CentroidActions.__init__
        """
        super().__init__(env)

        self.centroid_actions = CentroidActions(centroids)

        self.action_space = spaces.Discrete(n=self.centroid_actions.num_actions)

    def step(self, action):
        return_action = {}
        return_action["vector"] = self.centroid_actions.to_minerl(action)
        return self.env.step(return_action)


def fit_kmeans(remaining_args):
    import h5py
    from sklearn.cluster import KMeans

    parser = ArgumentParser("Fit k-Means to actions and save centroids")
    parser.add_argument("data", type=str, help="Path to the HDF5 file containing actions")
    parser.add_argument("output", type=str, help="Where to store centroids")
    parser.add_argument("--max-points", type=int, default=int(1e7), help="Maximum number of actions to include")
    parser.add_argument("--n-clusters", type=int, default=100, help="Number of centroids to use")
    parser.add_argument("--n-init", type=int, default=30, help="Number of times to init clustering")
    args = parser.parse_args(remaining_args)

    data = h5py.File(args.data, "r")
    print("Loading...")
    action_data = data["actions/vector"][:].copy()
    data.close()

    np.random.shuffle(action_data)
    action_data = action_data[:args.max_points]

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=args.n_init, verbose=2)
    kmeans.fit(action_data)

    centroids = kmeans.cluster_centers_

    np.save(args.output, centroids)


def update_hdf5_with_centroids(remaining_args):
    """
    Take HDF5 file, and add (or replace) datasets
    that will tell new frameskip amounts
    with these centroids.

    Adds datasets
        /actions
          /discrete_actions
          /num_action_repeated_discrete (number of times action is repeated, 1 being smallest)
          /start_of_new_action_discrete (1 if this index is start of new action)
          /vector_centroids

    TODO add n_discrete_actions to metadata
    """
    import h5py
    from tqdm import tqdm

    parser = ArgumentParser("Fit k-Means to actions and save centroids")
    parser.add_argument("data", type=str, help="Path to the HDF5 file to be updated")
    parser.add_argument("centroids", type=str, help="Path to centroids to use")
    args = parser.parse_args(remaining_args)

    centroid_data = np.load(args.centroids)
    action_converter = CentroidActions(centroid_data)

    data = h5py.File(args.data, "a")

    if "vector_centroids" in data:
        del data["vector_centroids"]
    vector_centroids = data.create_dataset("vector_centroids", shape=centroid_data.shape, dtype=centroid_data.dtype)
    vector_centroids[:] = centroid_data[:]

    num_action_repeated = data["actions/num_action_repeated"]
    if "actions/discrete_actions" in data:
        del data["actions/discrete_actions"]
    discrete_actions = data.create_dataset(
        "actions/discrete_actions",
        shape=num_action_repeated.shape,
        dtype=np.uint16
    )
    if "actions/num_action_repeated_discrete" in data:
        del data["actions/num_action_repeated_discrete"]
    num_action_repeated_discrete = data.create_dataset(
        "actions/num_action_repeated_discrete",
        shape=num_action_repeated.shape,
        dtype=np.uint8
    )
    if "actions/start_of_new_action_discrete" in data:
        del data["actions/start_of_new_action_discrete"]
    start_of_new_action_discrete = data.create_dataset(
        "actions/start_of_new_action_discrete",
        shape=num_action_repeated.shape,
        dtype=np.uint8
    )

    actions = data["actions/vector"]
    dones = data["dones"]

    # Go through actions, turn to centroided actions
    # and check where actions change
    # Mostly copy/paste from utils/handle_dataset.py
    last_action = None
    action_repeat_num = 0
    for i in tqdm(range(actions.shape[0])):
        action = actions[i]
        action = action_converter.from_minerl(action)

        discrete_actions[i] = action
        if last_action is None or action != last_action:
            if last_action is not None:
                # Store how often the last action was repeated,
                # each index telling how many times that action was
                # going to be executed in future.
                for j in range(1, action_repeat_num + 1):
                    num_action_repeated_discrete[i - j] = j
            last_action = action
            action_repeat_num = 1
        else:
            action_repeat_num += 1
        start_of_new_action_discrete[i] = 1 if action_repeat_num == 1 else 0

        if dones[i]:
            last_action = None

    data.close()


if __name__ == "__main__":
    VALID_OPERATIONS = {
        "cluster": fit_kmeans,
        "update_frameskip": update_hdf5_with_centroids
    }
    parser = ArgumentParser("Run action-wrapper related stuff")
    parser.add_argument("operation", type=str, choices=list(VALID_OPERATIONS.keys()), help="Operation to run")
    args, remainin_args = parser.parse_known_args()

    VALID_OPERATIONS[args.operation](remainin_args)
