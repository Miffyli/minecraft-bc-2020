from argparse import ArgumentParser
import numpy as np
import cv2

from gym import spaces, Wrapper


def resize_image(img, width_and_height):
    new_img = cv2.resize(img, width_and_height)
    return new_img


class MineRLPOVResizeWrapper(Wrapper):
    """
    Simple MineRL specific POV-image
    resizer
    """

    def __init__(self, env, target_shape=64):
        """
        target_shape: Image dimension per axis
        """
        super().__init__(env)
        self.target_shape = target_shape

        new_spaces = self.env.observation_space
        new_spaces.spaces["pov"] = spaces.Box(low=0, high=255, shape=(self.target_shape, self.target_shape, 3), dtype=np.uint8)
        self.observation_space = new_spaces

    def _check_and_resize(self, obs):
        if self.target_shape != 64:
            # Do simple copy to avoid modifying obs in place
            obs = dict(obs.items())
            obs["pov"] = resize_image(obs["pov"], (self.target_shape, self.target_shape))
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._check_and_resize(obs)
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs = self._check_and_resize(obs)
        return obs


def fit_kmeans(remaining_args):
    import h5py
    from sklearn.cluster import KMeans

    parser = ArgumentParser("Fit k-Means to observation vectors and save centroids")
    parser.add_argument("data", type=str, help="Path to the HDF5 file containing observations")
    parser.add_argument("output", type=str, help="Where to store the centroids")
    parser.add_argument("--max-points", type=int, default=int(1e7), help="Maximum number of observations to include")
    parser.add_argument("--n-clusters", type=int, default=100, help="Number of centroids to use")
    parser.add_argument("--n-init", type=int, default=30, help="Number of times to init clustering")
    args = parser.parse_args(remainin_args)

    data = h5py.File(args.data, "r")
    print("Loading...")
    vector_data = data["observations/vector"][:].copy()
    data.close()

    np.random.shuffle(vector_data)
    vector_data = vector_data[:args.max_points]

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=args.n_init, verbose=2)
    kmeans.fit(vector_data)

    centroids = kmeans.cluster_centers_

    np.save(args.output, centroids)


def update_hdf5_with_centroids(remaining_args):
    """
    Take HDF5 file, and add (or replace) datasets
    that contain discretized versions of the observations

    Adds datasets
        /observations
          /discrete_vector
          /vector_centroids

    """
    import h5py
    from tqdm import tqdm

    from action_wrappers import CentroidActions

    CENTROIDS_PATH = "observations/vector_centroids"
    DISCRETE_OBS_PATH = "observations/discrete_vectors"

    parser = ArgumentParser("Update hdf5 file with vectorized observations")
    parser.add_argument("data", type=str, help="Path to the HDF5 file to be updated")
    parser.add_argument("centroids", type=str, help="Path to centroids to use")
    args = parser.parse_args(remaining_args)

    centroid_data = np.load(args.centroids)
    # Reuse CentroidActions object here
    centroid_converter = CentroidActions(centroid_data)

    data = h5py.File(args.data, "a")

    if CENTROIDS_PATH in data:
        del data[CENTROIDS_PATH]
    vector_centroids = data.create_dataset(CENTROIDS_PATH, shape=centroid_data.shape, dtype=centroid_data.dtype)
    vector_centroids[:] = centroid_data[:]

    vector_obs = data["observations/vector"]
    if DISCRETE_OBS_PATH in data:
        del data[DISCRETE_OBS_PATH]
    discrete_vectors = data.create_dataset(
        DISCRETE_OBS_PATH,
        shape=(vector_obs.shape[0], 1),
        dtype=np.uint16
    )

    discrete_vectors_np = np.zeros((vector_obs.shape[0], 1), dtype=np.uint16)
    # Go through observation vectors and discretize
    # using kmeans.
    for i in tqdm(range(vector_obs.shape[0])):
        vector = vector_obs[i]
        vector = centroid_converter.from_minerl(vector)

        discrete_vectors_np[i] = vector

    discrete_vectors[:] = discrete_vectors_np[:]

    data.close()


if __name__ == "__main__":
    VALID_OPERATIONS = {
        "cluster": fit_kmeans,
        "update_discrete_obs": update_hdf5_with_centroids
    }
    parser = ArgumentParser("Run observation-wrapper related stuff")
    parser.add_argument("operation", type=str, choices=list(VALID_OPERATIONS.keys()), help="Operation to run")
    args, remainin_args = parser.parse_known_args()

    VALID_OPERATIONS[args.operation](remainin_args)
