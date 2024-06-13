"""

baseline code : https://github.com/idreesshaikh/Autonomous-Driving-in-Carla-using-Deep-Reinforcement-Learning/tree/30a2bb25cc12c56c35f47b8175307ada3247a4b1

"""
import os
import sys
import time
import random
import numpy as np
import argparse
import logging
import pickle
import torch

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
# from autoencoder.encoder_init import EncodeState

from carla_sim.environment import *
import torch
import gymnasium
import gym
from gymnasium import spaces
# 추가된 모듈
import ray
from ray.rllib.algorithms.sac.sac import SACConfig
from ray.rllib.utils.framework import try_import_torch
from ray.tune import register_env

# from ray.rllib.execution import ReplayBuffer, StoreToReplayBuffer, Replay, MixInReplay

import gym
from gymnasium.spaces import Box




def create_env(config):
    town = config["town"]
    checkpoint_frequency = config["checkpoint_frequency"]

    return CarlaEnv(town, checkpoint_frequency)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, help='name of the experiment')
    parser.add_argument('--env-name', type=str, default='carla', help='name of the simulation environment')
    # parser.add_argument('--learning-rate', type=float, default=PPO_LEARNING_RATE, help='learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=0, help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=1e6,help='total timesteps of the experiment')
    parser.add_argument('--test-timesteps', type=int, default=5e4, help='timesteps to test our model')
    parser.add_argument('--episode-length', type=int, default=7500, help='max timesteps in an episode')
    parser.add_argument('--train', default=True, type=bool, help='is it training?')
    parser.add_argument('--town', type=str, default="Town02", help='which town do you like?')
    parser.add_argument('--load-checkpoint', type=bool, default=False, help='resume training?')
    # parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,help='if toggled, `torch.backends.cudnn.deterministic=False`')
    # parser.add_argument('--cuda', type=str, default='cuda:0')
    args = parser.parse_args()

    return args




def runner():
    # ========================================================================
    #                           BASIC PARAMETER & LOGGING SETUP
    # ========================================================================
    args = parse_args()
    exp_name = args.exp_name
    train = args.train
    town = args.town
    checkpoint_load = args.load_checkpoint
    total_timesteps = args.total_timesteps

    try:
        run_name = "SAC"
    except Exception as e:
        print(e.message)
        sys.exit()

    if train:
        writer = SummaryWriter(f"runs/{run_name}_{int(total_timesteps)}/{town}")
    else:
        writer = SummaryWriter(f"runs/{run_name}_{int(total_timesteps)}_TEST/{town}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}" for key, value in vars(args).items()])))

    # Seeding to reproduce the results
    random.seed(args.seed)
    np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)

    # torch.backends.cudnn.deterministic = args.torch_deterministic

    timestep = 0
    episode = 0
    cumulative_score = 0
    episodic_length = list()
    scores = list()
    deviation_from_center = 0
    distance_covered = 0

    # ========================================================================
    #                           CREATING THE SIMULATION
    # ========================================================================
    # try:
    #     env = CarlaEnv(town)
    #     logging.info("Connection has been setup successfully.")
    # except:
    #     logging.error("Connection has been refused by the server.")
    #     ConnectionRefusedError

    if train:
        env = CarlaEnv(town)
    else:
        env = CarlaEnv(town, checkpoint_frequency=None)

    print("in main file Observation Space:", env.observation_space)  # Debug statement
    print("in main file  Action Space:", env.action_space)  # Debug statement

    #encode = EncodeState(LATENT_DIM)

    register_env('CarlaEnv-v0', create_env)
    # ========================================================================
    #                           ALGORITHM
    # ========================================================================

    #register_env("carla_env", create_env(town))
    ray.init()
    torch, nn = try_import_torch()



    sac_config = (
        SACConfig()
        .environment("CarlaEnv-v0", env_config={"town": town, "checkpoint_frequency": 100 if train else None})  # Replace with your environment
        .training(
            twin_q=True,
            gamma=0.99
        )
        .framework("torch")  # Use PyTorch as the framework
        .resources(num_gpus=0)  # Change this to the number of GPUs you want to use  # .env_runners(num_env_runners=2)

    )


    sac_config.observation_space = env.observation_space
    sac_config.action_space = env.action_space

    sac_config.learning_starts = 1000
    sac_config.buffer_size = 1000000
    sac_config.prioritized_replay = True
    sac_config.prioritized_replay_alpha = 0.6
    sac_config.prioritized_replay_beta = 0.4
    sac_config.prioritized_replay_eps = 1e-6
    sac_config["_disable_preprocessor_api"] = True

    sac_agent = sac_config.build()

    if checkpoint_load:
        sac_agent.restore('path/to/checkpoint')

    try:
        time.sleep(0.5)


        if train:
            # Training
            while timestep < total_timesteps:
                observation = env.reset()
                observation={'vae_latent':observation['vae_latent'],
                             'vehicle_measures':observation['vehicle_measures']}
                #observation = encode.process(observation)


                current_ep_reward = 0
                t1 = datetime.now()

                for t in range(args.episode_length):
                    action = sac_agent.compute_single_action(observation)#.cpu().numpy()
                    observation, reward, done, info = env.step(action)
                    if observation is None:
                        break
                    #observation = encode.process(observation)

                    timestep += 1
                    current_ep_reward += reward

                    if timestep == total_timesteps - 1:
                        sac_agent.save('path/to/checkpoint')

                    if done:
                        episode += 1
                        t2 = datetime.now()
                        t3 = t2 - t1
                        episodic_length.append(abs(t3.total_seconds()))
                        break

                deviation_from_center += info[1]
                distance_covered += info[0]
                scores.append(current_ep_reward)
                cumulative_score = np.mean(scores)

                print('Episode: {}'.format(episode), ', Timestep: {}'.format(timestep),
                      ', Reward:  {:.2f}'.format(current_ep_reward),
                      ', Average Reward:  {:.2f}'.format(cumulative_score))

                if episode % 10 == 0:
                    sac_agent.train()
                    sac_agent.save('path/to/checkpoint')

                if episode % 5 == 0:
                    writer.add_scalar("Episodic Reward/episode", scores[-1], episode)
                    writer.add_scalar("Cumulative Reward/info", cumulative_score, episode)
                    writer.add_scalar("Cumulative Reward/(t)", cumulative_score, timestep)
                    writer.add_scalar("Average Episodic Reward/info", np.mean(scores[-5:]), episode)
                    writer.add_scalar("Average Reward/(t)", np.mean(scores[-5:]), timestep)
                    writer.add_scalar("Episode Length (s)/info", np.mean(episodic_length), episode)
                    writer.add_scalar("Reward/(t)", current_ep_reward, timestep)
                    writer.add_scalar("Average Deviation from Center/episode", deviation_from_center / 5, episode)
                    writer.add_scalar("Average Deviation from Center/(t)", deviation_from_center / 5, timestep)
                    writer.add_scalar("Average Distance Covered (m)/episode", distance_covered / 5, episode)
                    writer.add_scalar("Average Distance Covered (m)/(t)", distance_covered / 5, timestep)

                    episodic_length = list()
                    deviation_from_center = 0
                    distance_covered = 0

                if episode % 100 == 0:
                    sac_agent.save('path/to/checkpoint')

            print("Terminating the run.")
            sys.exit()
        else:
            # Testing
            while timestep < args.test_timesteps:
                observation = env.reset()
                observation={'vae_latent':observation['vae_latent'],
                             'vehicle_measures':observation['vehicle_measures']}
                #observation = encode.process(observation)

                current_ep_reward = 0
                t1 = datetime.now()

                for t in range(args.episode_length):
                    action = sac_agent.compute_action(observation, explore=False)
                    observation, reward, done, info = env.step(action)
                    if observation is None:
                        break
                    #observation = encode.process(observation)

                    timestep += 1
                    current_ep_reward += reward

                    if done:
                        episode += 1
                        t2 = datetime.now()
                        t3 = t2 - t1
                        episodic_length.append(abs(t3.total_seconds()))
                        break

                deviation_from_center += info[1]
                distance_covered += info[0]
                scores.append(current_ep_reward)
                cumulative_score = np.mean(scores)

                print('Episode: {}'.format(episode), ', Timestep: {}'.format(timestep),
                      ', Reward:  {:.2f}'.format(current_ep_reward),
                      ', Average Reward:  {:.2f}'.format(cumulative_score))

                writer.add_scalar("TEST: Episodic Reward/episode", scores[-1], episode)
                writer.add_scalar("TEST: Cumulative Reward/info", cumulative_score, episode)
                writer.add_scalar("TEST: Cumulative Reward/(t)", cumulative_score, timestep)
                writer.add_scalar("TEST: Episode Length (s)/info", np.mean(episodic_length), episode)
                writer.add_scalar("TEST: Reward/(t)", current_ep_reward, timestep)
                writer.add_scalar("TEST: Deviation from Center/episode", deviation_from_center, episode)
                writer.add_scalar("TEST: Deviation from Center/(t)", deviation_from_center, timestep)
                writer.add_scalar("TEST: Distance Covered (m)/episode", distance_covered, episode)
                writer.add_scalar("TEST: Distance Covered (m)/(t)", distance_covered, timestep)

                episodic_length = list()
                deviation_from_center = 0
                distance_covered = 0

            print("Terminating the run.")
            sys.exit()

    finally:
        sys.exit()
        ray.shutdown()


if __name__ == "__main__":
    try:

        runner()
        print("im in 22 ")
    except KeyboardInterrupt:
        print("runner falied")
        sys.exit()
    finally:
        print('\nExit')

