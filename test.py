import os
# To speed things up. This speeds things up
# by avoiding any parallelization of the
# numpy computations, which seem to
# happen bunch with VectorObf envs.
os.environ["OMP_NUM_THREADS"] = "1"

import math
import json
import select
import time
import logging
import threading

from typing import Callable

import gym
import minerl
import abc
import numpy as np

import coloredlogs

# My imports
import torch
from torch_codes.modules import IMPALANetwork

coloredlogs.install(logging.DEBUG)

# All the evaluations will be evaluated on MineRLObtainDiamondVectorObf-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLObtainDiamondVectorObf-v0')
MINERL_MAX_EVALUATION_EPISODES = int(os.getenv('MINERL_MAX_EVALUATION_EPISODES', 5))

# Parallel testing/inference, **you can override** below value based on compute
# requirements, etc to save OOM in this phase.
EVALUATION_THREAD_COUNT = int(os.getenv('EPISODES_EVALUATION_THREAD_COUNT', 2))


class EpisodeDone(Exception):
    pass


class Episode(gym.Env):
    """A class for a single episode.
    """
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self._done = False

    def reset(self):
        if not self._done:
            return self.env.reset()

    def step(self, action):
        s, r, d, i = self.env.step(action)
        if d:
            self._done = True
            raise EpisodeDone()
        else:
            return s, r, d, i

# DO NOT CHANGE THIS CLASS, THIS IS THE BASE CLASS FOR YOUR AGENT.


class MineRLAgentBase(abc.ABC):
    """
    To compete in the competition, you are required to implement a
    SUBCLASS to this class.

    YOUR SUBMISSION WILL FAIL IF:
        * Rename this class
        * You do not implement a subclass to this class

    This class enables the evaluator to run your agent in parallel,
    so you should load your model only once in the 'load_agent' method.
    """

    @abc.abstractmethod
    def load_agent(self):
        """
        This method is called at the beginning of the evaluation.
        You should load your model and do any preprocessing here.
        THIS METHOD IS ONLY CALLED ONCE AT THE BEGINNING OF THE EVALUATION.
        DO NOT LOAD YOUR MODEL ANYWHERE ELSE.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def run_agent_on_episode(self, single_episode_env : Episode):
        """This method runs your agent on a SINGLE episode.

        You should just implement the standard environment interaction loop here:
            obs  = env.reset()
            while not done:
                env.step(self.agent.act(obs))
                ...

        NOTE: This method will be called in PARALLEL during evaluation.
            So, only store state in LOCAL variables.
            For example, if using an LSTM, don't store the hidden state in the class
            but as a local variable to the method.

        Args:
            env (gym.Env): The env your agent should interact with.
        """
        raise NotImplementedError()


#######################
# YOUR CODE GOES HERE #
#######################

class MineRLMatrixAgent(MineRLAgentBase):
    """
    An example random agent.
    Note, you MUST subclass MineRLAgentBase.
    """

    def load_agent(self):
        """In this example we make a random matrix which
        we will use to multiply the state by to produce an action!

        This is where you could load a neural network.
        """
        # Some helpful constants from the environment.
        flat_video_obs_size = 64 * 64 * 3
        obs_size = 64
        ac_size = 64
        self.matrix = np.random.random(size=(ac_size, flat_video_obs_size + obs_size)) * 2 - 1
        self.flatten_obs = lambda obs: np.concatenate([obs['pov'].flatten() / 255.0, obs['vector'].flatten()])
        self.act = lambda flat_obs: {'vector': np.clip(self.matrix.dot(flat_obs), -1, 1)}

    def run_agent_on_episode(self, single_episode_env : Episode):
        """Runs the agent on a SINGLE episode.

        Args:
            single_episode_env (Episode): The episode on which to run the agent.
        """
        obs = single_episode_env.reset()
        done = False
        while not done:
            obs, reward, done, _ = single_episode_env.step(self.act(self.flatten_obs(obs)))


class MineRLRandomAgent(MineRLAgentBase):
    """A random agent"""

    def load_agent(self):
        pass # Nothing to do, this agent is a random agent.

    def run_agent_on_episode(self, single_episode_env: Episode):
        obs = single_episode_env.reset()
        done = False
        while not done:
            random_act = single_episode_env.action_space.sample()
            single_episode_env.step(random_act)


class MineRLRandomKMeansAgent(MineRLAgentBase):
    """A random agent based on kmeans actions"""

    def load_agent(self):
        # Fixed parameters
        self.centroids = np.load("train/action_centroids.npy")
        # Fixed frameskip, courtesy of Guss (from the baseline example)
        self.frameskip = 10

    def run_agent_on_episode(self, single_episode_env: Episode):
        obs = single_episode_env.reset()
        done = False
        while not done:
            random_act = self.centroids[np.random.randint(0, self.centroids.shape[0])]
            random_act = {"vector": random_act}
            for i in range(self.frameskip):
                obs, reward, done, _ = single_episode_env.step(random_act)
                if done:
                    break


class TorchDiscreteActionPolicy(MineRLAgentBase):
    """
    Agent using a PyTorch network to predict
    discrete actions, which are mapped to action
    vectors with given clusters.
    """

    def load_agent(self):
        # TODO hardcoded settings
        self.centroids = np.load("train/action_centroids.npy")
        self.model = torch.load("train/trained_model.th")

        # Deduce if we want to use LSTM from the model.
        # Hidden-state models should have `get_initial_state` function.
        self.lstm = hasattr(self.model, "get_initial_state")
        # Get maximum number of frameskip and actions
        self.max_action = self.model.output_dict["action"]
        self.max_frameskip = self.model.output_dict["frameskip"]
        self.action_eye = np.eye(self.max_action)
        self.frameskip_eye = np.eye(self.max_frameskip)

    def run_agent_on_episode(self, single_episode_env: Episode):
        # Define step stuff etc here to avoid any problems with
        # parallel calls to run_agent_on_episode
        previous_action = 0
        previous_frameskip = 0
        previous_reward = 0
        hidden_state = self.model.get_initial_state(batch_size=1)

        obs = single_episode_env.reset()
        done = False
        reward = 0
        while not done:
            prediction = None
            with torch.no_grad():
                if self.lstm:
                    processed_reward = np.array([math.log2(previous_reward + 1)])
                    vector_obs = np.concatenate(
                        (
                            # Mind the ordering here (must match training)
                            obs["vector"],
                            self.action_eye[previous_action],
                            # "1 frameskip" -> [1 0 0 0 ...]
                            self.frameskip_eye[previous_frameskip - 1],
                            processed_reward
                        ),
                        axis=0
                    )
                    prediction, hidden_state = self.model(
                        # TODO move this transposing somewhere else...
                        # TODO make the whole "float()"  stuff unified somehow
                        # Add batch and time dimensions here
                        torch.from_numpy(obs["pov"].transpose(2, 0, 1)[None, None]).cuda(),
                        torch.from_numpy(vector_obs[None, None]).float().cuda(),
                        hidden_states=hidden_state
                    )
                else:
                    prediction = self.model(
                        # TODO move this transposing somewhere else...
                        # TODO make the whole "float()"  stuff unified somehow
                        torch.from_numpy(obs["pov"].transpose(2, 0, 1)[None]).cuda(),
                        torch.from_numpy(obs["vector"][None]).float().cuda(),
                        None
                    )

            action_prediction = prediction["action"][0]
            action_prediction = torch.softmax(action_prediction, 0).cpu().detach().numpy()
            # Sample action
            action = np.random.choice(np.arange(len(action_prediction)), p=action_prediction)
            action = int(action)

            frameskip = 1
            if "frameskip" in prediction.keys():
                frameskip_probs = torch.softmax(prediction["frameskip"][0], 0).cpu().detach().numpy()
                # Sample frameskip to do
                frameskip = np.random.choice(np.arange(len(frameskip_probs)), p=frameskip_probs)
                # Add +1, because that is single action
                frameskip += 1

            previous_action = action
            previous_frameskip = frameskip
            previous_reward = reward

            action_vector = self.centroids[action]

            action = {"vector": action_vector}
            for i in range(frameskip):
                obs, reward, done, _ = single_episode_env.step(action)
                if done:
                    break

#####################################################################
# IMPORTANT: SET THIS VARIABLE WITH THE AGENT CLASS YOU ARE USING 
######################################################################
AGENT_TO_TEST = TorchDiscreteActionPolicy


####################
# EVALUATION CODE  #
####################


def main():
    agent = AGENT_TO_TEST()
    assert isinstance(agent, MineRLAgentBase)
    agent.load_agent()

    assert MINERL_MAX_EVALUATION_EPISODES > 0
    assert EVALUATION_THREAD_COUNT > 0

    # Create the parallel envs (sequentially to prevent issues!)
    envs = [gym.make(MINERL_GYM_ENV) for _ in range(EVALUATION_THREAD_COUNT)]
    episodes_per_thread = [MINERL_MAX_EVALUATION_EPISODES // EVALUATION_THREAD_COUNT for _ in range(EVALUATION_THREAD_COUNT)]
    episodes_per_thread[-1] += MINERL_MAX_EVALUATION_EPISODES - EVALUATION_THREAD_COUNT *(MINERL_MAX_EVALUATION_EPISODES // EVALUATION_THREAD_COUNT)

    # A simple funciton to evaluate on episodes!
    def evaluate(i, env):
        print("[{}] Starting evaluator.".format(i))
        for i in range(episodes_per_thread[i]):
            try:
                agent.run_agent_on_episode(Episode(env))
            except EpisodeDone:
                print("[{}] Episode complete".format(i))
                pass

    evaluator_threads = [threading.Thread(target=evaluate, args=(i, envs[i])) for i in range(EVALUATION_THREAD_COUNT)]
    for thread in evaluator_threads:
        thread.start()

    # wait fo the evaluation to finish
    for thread in evaluator_threads:
        thread.join()


if __name__ == "__main__":
    main()
