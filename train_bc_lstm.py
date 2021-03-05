#
# train_bc.py
#
# Train simple behavioural cloning on the dataset
#

from argparse import ArgumentParser
from collections import deque
from functools import partial
from time import time

import numpy as np
import torch
from tqdm import tqdm
from gym import spaces

from dataloaders.hdf5_loader import HDF5Loader, HDF5SequenceSampler
from torch_codes.modules import IMPALANetworkWithLSTM
from utils.misc_utils import parse_keyword_arguments
from wrappers.action_wrappers import CentroidActions
from wrappers.observation_wrappers import resize_image

LSTM_LATENT_SIZE = 512

RESNETS = [
    "ResNetHeadFor64x64",
    "ResNetHeadFor32x32",
    "ResNetHeadFor64x64DoubleFilters",
    "ResNetHeadFor64x64QuadrupleFilters",
    "ResNetHeadFor64x64DoubleFiltersWithExtra"
]

parser = ArgumentParser("Train PyTorch networks on MineRL data with behavioural cloning on LSTM networks.")
parser.add_argument("hdf5_file", type=str, help="MineRL dataset as a HDF5 file (see utils/handle_dataset.py)")
parser.add_argument("output", type=str, help="Where to store the trained model")
parser.add_argument("--batch-size", type=int, default=64, help="Yer standard batch size")
parser.add_argument("--num-epochs", type=int, default=5, help="Number of times to go over the dataset")
parser.add_argument("--include-frameskip", type=int, default=None, help="If provided, predict frameskip and this is max")
parser.add_argument("--lr", type=float, default=0.0005, help="Good old learning rate for Adam")
parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for Adam (i.e. L2 loss)")
parser.add_argument("--entropy-weight", type=float, default=0.0, help="Entropy coefficient for discrete actions")
parser.add_argument("--image-size", type=int, default=64, help="Resized image shape (original is 64)")
parser.add_argument("--resnet", type=str, default="ResNetHeadFor64x64", choices=RESNETS, help="ResNet type to use for images")

parser.add_argument("--seq-len", type=int, default=32, help="Length of sequences to store (and backprop over)")
parser.add_argument("--frameskip-from-vector", action="store_true", help="Use frameskip targets based on action vectors, not discrezited actions")
parser.add_argument("--num-discrete-actions", type=int, default=100, help="DIRTY way of providing number of discrete options, for now")

parser.add_argument("--horizontal-flipping", action="store_true", help="Flip images horizontally randomly (per trajectory)")
parser.add_argument("--dropout-rate", type=float, default=None, help="If given, this is probability of any input value being replaced with zero.")


def main(args, unparsed_args):
    # Create dataloaders
    assert args.include_frameskip is not None, "This code only works with frameskip enabled."

    resize_func = None if args.image_size == 64 else partial(resize_image, width_and_height=(args.image_size, args.image_size))
    dataset_mappings = {
        "observations/pov": ("pov", resize_func),
        "observations/vector": ("obs_vector", None),
        "rewards": ("reward", None),
    }

    dataset_mappings["actions/discrete_actions"] = ("action", None)
    if args.frameskip_from_vector:
        dataset_mappings["actions/num_action_repeated"] = ("frameskip", None)
    else:
        dataset_mappings["actions/num_action_repeated_discrete"] = ("frameskip", None)

    data_sampler = HDF5SequenceSampler(
        args.hdf5_file,
        dataset_mappings,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        traj_length=args.seq_len,
    )

    # Create temporary HDF5Loader to get the types
    temp_loader = HDF5Loader(args.hdf5_file, dataset_mappings)
    # TODO read n_discrete_actions here
    shapes_and_types = temp_loader.get_types()
    temp_loader.close()

    # This is hardcoded in PyTorch format
    image_shape = (3, args.image_size, args.image_size)

    num_additional_features = shapes_and_types["obs_vector"]["shape"][0]
    # Add one-hot vector sizes (action and frameskip) and reward
    num_additional_features += args.num_discrete_actions + args.include_frameskip + 1

    # Define the action_space so we know to do scaling etc later,
    # as well as how many scalars we need from network
    # TODO need prettier way to tell what is the maximum action
    num_action_outputs = args.num_discrete_actions

    output_dict = {
        "action": num_action_outputs,
        "frameskip": args.include_frameskip
    }

    # Bit of sanity checking
    if args.resnet != "ResNetHeadFor32x32" and args.image_size < 64:
        raise ValueError("Using a big network for smaller image. You suuuuure you want to do that?")

    network = IMPALANetworkWithLSTM(
        image_shape,
        output_dict,
        num_additional_features,
        cnn_head_class=args.resnet,
        latent_size=LSTM_LATENT_SIZE
    ).cuda()
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Keep track of hidden states per episodes ("h" and "c" components of LSTM)
    hidden_state_h = torch.zeros(
        data_sampler.num_episodes,
        LSTM_LATENT_SIZE
    ).cuda()

    hidden_state_c = torch.zeros(
        data_sampler.num_episodes,
        LSTM_LATENT_SIZE
    ).cuda()

    # Also keep track on if we should flip the images horizontally in the episode.
    # NOTE that we do not flip actions! This might be wrong, but also with LSTM
    #      the network _could_ learn to distinguish which way the world works
    #      -> works as a an augmentation?
    horizontal_flip_episodes = np.random.randint(0, 2, size=(data_sampler.num_episodes,)).astype(np.bool)

    # A quick (altho dirty) way of creating one-hot encodings
    action_eye = np.eye(args.num_discrete_actions)
    frameskip_eye = np.eye(args.include_frameskip)

    losses = deque(maxlen=1000)
    start_time = time()

    for i, data_batch_and_episode_indeces in enumerate(tqdm(data_sampler, desc="train")):
        data_batch = data_batch_and_episode_indeces[0]
        episode_indeces = data_batch_and_episode_indeces[1]

        # Easy way out of the whole "episodes ending at different times":
        # Discard samples where final step is nans (i.e. the sampled
        # trajectory ended prematurely)
        sample_mask = ~np.isnan(data_batch["obs_vector"][-1, :, 0])

        # Augment observation vectors with previous actions, frameskips and rewards
        obs_vector = data_batch["obs_vector"][:, sample_mask]
        target_action = data_batch["action"][:, sample_mask]
        target_frameskip = data_batch["frameskip"][:, sample_mask]
        # Move frameskip=1 to be in index 0, also clip at higher end
        target_frameskip = np.clip(target_frameskip, 1, args.include_frameskip) - 1
        reward = data_batch["reward"][:, sample_mask]

        # Offset by one and append with zeros
        # TODO not correct (first sample has wrong action to bootstrap to),
        #      but hopefully sequence length fixes for this mostly...
        target_action_onehot = np.pad(target_action[:-1, :, :], ((1, 0), (0, 0), (0, 0)))
        target_frameskip_onehot = np.pad(target_frameskip[:-1, :, :], ((1, 0), (0, 0), (0, 0)))
        reward = np.pad(reward[:-1, :, :], ((1, 0), (0, 0), (0, 0)))
        # Turn into onehot (remove last dimension, it will be replaced with the one-hot)
        target_action_onehot = action_eye[target_action_onehot[..., 0]]
        target_frameskip_onehot = frameskip_eye[target_frameskip_onehot[..., 0]]
        # Convert reward into something more comfy
        reward = np.log2(reward + 1)
        # Finally concatenate everything
        obs_vector = np.concatenate(
            (
                obs_vector,
                target_action_onehot,
                target_frameskip_onehot,
                reward
            ),
            axis=2
        )

        # Random dropout
        if args.dropout_rate is not None:
            obs_vector *= np.random.random(obs_vector.shape) > args.dropout_rate

        # Gather episode indeces and the stored states.
        # Each episode is sampled at random, but will be always
        # provided in a sequence so we can trust that the latest hidden
        # states to what we should have at this point of in the episode
        masked_episode_indeces = episode_indeces[sample_mask]

        # Transpose to channel-first
        pov = data_batch["pov"][:, sample_mask].transpose(0, 1, 4, 2, 3)
        if args.horizontal_flipping:
            # Flip on width-axis (last now)
            masked_horizontal_flip = horizontal_flip_episodes[masked_episode_indeces]
            pov[:, masked_horizontal_flip] = np.flip(pov[:, masked_horizontal_flip], 4)

        pov = torch.from_numpy(pov).cuda()
        obs_vector = torch.from_numpy(obs_vector).float().cuda()

        # Add the initial "num-layers shape"
        hidden_states = (
            hidden_state_h[masked_episode_indeces][None],
            hidden_state_c[masked_episode_indeces][None]
        )

        network_output, new_states = network(pov, obs_vector, hidden_states=hidden_states, return_sequence=True)

        # Store new stats back to tracked episode states.
        # Kill the gradient so we do not try to backprop.
        hidden_state_h[masked_episode_indeces] = new_states[0][0].detach()
        hidden_state_c[masked_episode_indeces] = new_states[1][0].detach()

        # Also reset hidden states of any episodes that ended
        if not np.all(sample_mask):
            terminal_indeces = episode_indeces[~sample_mask]
            hidden_state_h[terminal_indeces].zero_()
            hidden_state_c[terminal_indeces].zero_()
            # Flip coin if those episodes should be horizontally flipped
            horizontal_flip_episodes[terminal_indeces] = np.random.randint(0, 2, size=(len(terminal_indeces),)).astype(np.bool)

        total_loss = 0

        # pi-loss (i.e. predict correct action).
        predicted_action = network_output["action"]
        # Remove the extra dimension in the end
        target_action = torch.from_numpy(target_action.astype(np.int64)[..., -1]).cuda()

        # Maximize llk
        dist = torch.distributions.Categorical(logits=predicted_action)
        log_prob = dist.log_prob(target_action).mean()
        total_loss += -log_prob

        if args.entropy_weight != 0.0:
            # Maximize entropy with some small weight
            total_loss += (-dist.entropy().mean()) * args.entropy_weight

        # Action-frameskip loss
        predicted_frameskip = network_output["frameskip"]
        # Remove extra dimension in the end
        target_frameskip = torch.from_numpy(target_frameskip[..., -1]).long().cuda()

        dist = torch.distributions.Categorical(logits=predicted_frameskip)
        log_prob = dist.log_prob(target_frameskip).mean()
        total_loss += -log_prob

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        losses.append(total_loss.cpu().detach().item())

        if (i % 10000) == 0:
            tqdm.write("Steps {:<9} Time {:<9} Avrg loss {:<10.5f}".format(
                i,
                int(time() - start_time),
                sum(losses) / len(losses)
            ))

            # TODO consider using state_dict variant,
            #      to avoid any pickling issues etc
            torch.save(network, args.output)
    data_sampler.close()
    torch.save(network, args.output)


if __name__ == "__main__":
    args, unparsed_args = parser.parse_known_args()
    main(args, unparsed_args)
