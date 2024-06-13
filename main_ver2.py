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
import copy
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
# from autoencoder.encoder_init import EncodeState

from carla_sim.environment import *
import torch
import gymnasium
import gym
from gymnasium import spaces
# 추가된 모듈


# from ray.rllib.execution import ReplayBuffer, StoreToReplayBuffer, Replay, MixInReplay

import gym
from gymnasium.spaces import Box


import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
import torch.optim as optim

import argparse
from torch.utils.tensorboard import SummaryWriter
import time

device ='cuda'

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_size=256, init_w=3e-3, log_std_min=-20, log_std_max=2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std_linear = nn.Linear(hidden_size, action_size)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def evaluate(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()

        dist = Normal(0, 1)
        e = dist.sample().to(device)
        action = torch.tanh(mu + e * std)
        log_prob = Normal(mu, std).log_prob(mu + e * std) - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob


    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        mu, log_std = self.forward(state.unsqueeze(0))
        std = log_std.exp()

        dist = Normal(0, 1)
        e = dist.sample().to(device)
        action = torch.tanh(mu + e * std).cpu()
        return action[0]

class RBFCritic(nn.Module):
    def __init__(self, state_dim, action_dim, num_centroids, beta):
        super(RBFCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_centroids = num_centroids
        self.beta = beta

        # Define the layers to compute the centroids a_i(s; θ)
        self.centroid_layer = nn.Linear(state_dim, num_centroids * action_dim)
        # Define the layers to compute the values v_i(s; θ)
        self.value_layer = nn.Linear(state_dim, num_centroids)

    def forward(self, state, action):
        # Compute the centroids a_i(s; θ)
        centroids = self.centroid_layer(state).view(-1, self.num_centroids, self.action_dim)
        # Compute the values v_i(s; θ)
        values = self.value_layer(state).view(-1, self.num_centroids)

        # Compute the RBF distances
        diff = (action.unsqueeze(1) - centroids).pow(2).sum(dim=2)
        rbf = torch.exp(-self.beta * diff)

        # Compute the Q values using the RBF weights
        rbf_sum = rbf.sum(dim=1, keepdim=True)
        weighted_values = rbf * values
        q_values = weighted_values.sum(dim=1) / rbf_sum

        return q_values


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, num_centroids = 256,hidden_size=256):
        super(Critic, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.beta = 1.0

        # Network to learn state-dependent centroid locations a_i(s; θ)
        self.centroid_net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_centroids * action_size)
        )

        # Network to learn state-dependent centroid values v_i(s; θ)
        self.value_net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_centroids)
        )
        self.fc3 = nn.Linear(num_centroids, 1)

    def forward(self, state, action):
        batch_size = state.size(0)

        # Get centroid locations a_i(s; θ)
        centroid_locations = self.centroid_net(state).view(batch_size, -1, action.size(1))  # Shape: (batch_size, num_centroids, action_size)

        # Get centroid values v_i(s; θ)
        centroid_values = self.value_net(state)  # Shape: (batch_size, num_centroids)

        # Compute distances between actions and centroids
        action = action.unsqueeze(1).expand_as(centroid_locations)  # Shape: (batch_size, num_centroids, action_size)
        distances = torch.norm(action - centroid_locations, dim=2)  # Shape: (batch_size, num_centroids)

        # Compute RBF weights
        rbf_weights = torch.exp(-self.beta * distances)  # Shape: (batch_size, num_centroids)

        # Compute weighted sum of centroid values
        q_values = torch.sum(rbf_weights * centroid_values, dim=1) / torch.sum(rbf_weights, dim=1)  # Shape: (batch_size,)

        return self.fc3(q_values)#q_values.unsqueeze(1)  # Shape: (batch_size, 1)

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed, action_prior="uniform"):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        self.target_entropy = -action_size  # -dim(A)
        self.alpha = 1
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=5e-4)
        self._action_prior = action_prior

        print("Using: ", device)

        # Actor Network
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=5e-4)

        #self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        # Critic Network (w/ Target Network)
        self.critic1 = Critic(state_size, action_size, random_seed).to(device)
        self.critic2 = Critic(state_size, action_size, random_seed).to(device)

        self.critic1_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(state_size, action_size, random_seed).to(device)#Critic(state_size, action_size, random_seed).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=5e-4, weight_decay=1e-2 )
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=5e-4, weight_decay=1e-2 )

        # Replay memory
        self.memory = PrioritizedReplay(capacity=int(1e6))
    def save(self,filename):
        torch.save(self.actor_local.state_dict(),f"{filename}_actor")
        torch.save(self.actor_optimizer.state_dict(), f"{filename}_actor_optimizer")

        torch.save(self.critic1.state_dict(), f"{filename}_critic1")
        torch.save(self.critic2.state_dict(), f"{filename}_critic2")
        torch.save(self.critic1_optimizer.state_dict(), f"{filename}_critic1_optimizer")
        torch.save(self.critic2_optimizer.state_dict(), f"{filename}_critic2_optimizer")


    def load(self,filename):
        self.actor_local.load_state_dict(torch.load(f"{filename}_actor"))
        self.actor_optimizer.load_state_dict(torch.load(f"{filename}_actor_optimizer"))

        self.critic1.load_state_dict(torch.load(f"{filename}_critic1"))
        self.critic2.load_state_dict(torch.load(f"{filename}_critic2"))
        self.critic1_optimizer.load_state_dict(torch.load(f"{filename}_critic1_optimizer"))
        self.critic2_optimizer.load_state_dict(torch.load(f"{filename}_critic2_optimizer"))
        #self.critic1_target = self.critic1_target.load_state_dict(self.critic1.state_dict())

        #self.critic2_target = self.critic2_target.load_state_dict(self.critic2.state_dict())

    def step(self, state, action, reward, next_state, done, step,e):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.push(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if e%10==0:
            experiences = self.memory.sample(256)
            self.learn(step, experiences, 0.99)


    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""


        #state = torch.from_numpy(state).unsqueeze(0).float().to(device)
        action = self.actor_local.get_action(state).detach()
        return action

    def learn(self, step, experiences, gamma, d=1):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, idx, weights = experiences
        states = torch.FloatTensor(np.float32(states)).to(device)
        next_states = torch.FloatTensor(np.float32(next_states)).to(device)
        actions = torch.stack(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device).unsqueeze(1)
        dones = torch.FloatTensor(dones).to(device).unsqueeze(1)
        weights = torch.FloatTensor(weights).unsqueeze(1)
        #print(actions.shape)
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        next_action, log_pis_next = self.actor_local.evaluate(next_states)

        Q_target1_next = self.critic1_target(next_states.to(device), next_action.squeeze(0).to(device))
        Q_target2_next = self.critic2_target(next_states.to(device), next_action.squeeze(0).to(device))

        # take the mean of both critics for updating
        Q_target_next = torch.min(Q_target1_next, Q_target2_next).cpu()

        # Compute Q targets for current states (y_i)
        Q_targets = rewards.cpu() + (gamma * (1 - dones.cpu()) * (Q_target_next - self.alpha * log_pis_next.mean(1).unsqueeze(1).cpu()))

        # Compute critic loss
        Q_1 = self.critic1(states, actions).cpu()
        Q_2 = self.critic2(states, actions).cpu()
        td_error1 = Q_targets.detach()-Q_1#,reduction="none"
        td_error2 = Q_targets.detach()-Q_2
        critic1_loss = 0.5* (td_error1.pow(2)*weights).mean()
        critic2_loss = 0.5* (td_error2.pow(2)*weights).mean()
        prios = abs(((td_error1 + td_error2)/2.0 + 1e-5).squeeze())

        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        self.memory.update_priorities(idx, prios.data.cpu().numpy())
        if step % d == 0:
        # ---------------------------- update actor ---------------------------- #

            alpha = torch.exp(self.log_alpha)
            # Compute alpha loss
            actions_pred, log_pis = self.actor_local.evaluate(states)
            alpha_loss = - (self.log_alpha.cpu() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()

            if abs(alpha_loss.item())==0:
                alpha_loss += 1e-6

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = alpha
            # Compute actor loss
            # if self._action_prior == "normal":
            #     policy_prior = MultivariateNormal(loc=torch.zeros(self.action_size), scale_tril=torch.ones(self.action_size).unsqueeze(0))
            #     policy_prior_log_probs = policy_prior.log_prob(actions_pred)
            # elif self._action_prior == "uniform":
            #     policy_prior_log_probs = 0.0

            policy_prior_log_probs = 0.0

            actor_loss = ((alpha * log_pis.squeeze(0).cpu() - self.critic1(states, actions_pred.squeeze(0)).cpu() - policy_prior_log_probs )*weights).mean()
            actor_loss.requires_grad_()
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic1, self.critic1_target, 1e-2)
            self.soft_update(self.critic2, self.critic2_target, 1e-2)



    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)



class PrioritizedReplay(object):
    """
    Proportional Prioritization
    """
    def __init__(self, capacity, alpha=0.6,beta_start = 0.4,beta_frames=100000):
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1 #for beta calculation
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def beta_by_frame(self, frame_idx):
        """
        Linearly increases beta from beta_start to 1 over time from 1 to beta_frames.

        3.4 ANNEALING THE BIAS (Paper: PER)
        We therefore exploit the flexibility of annealing the amount of importance-sampling
        correction over time, by defining a schedule on the exponent
        that reaches 1 only at the end of learning. In practice, we linearly anneal from its initial value 0 to 1
        """
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        state = np.expand_dims(state.cpu(), 0)
        next_state = np.expand_dims(next_state.cpu(), 0)

        max_prio = self.priorities.max() if self.buffer else 1.0 # gives max priority if buffer is not empty else 1

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action.cpu(), reward, next_state, done))
        else:
            # puts the new data on the position of the oldes since it circles via pos variable
            # since if len(buffer) == capacity -> pos == 0 -> oldest memory (at least for the first round?)
            self.buffer[self.pos] = (state, action.cpu(), reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity # lets the pos circle in the ranges of capacity if pos+1 > cap --> new posi = 0

    def sample(self, batch_size):
        N = len(self.buffer)
        if N == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        # calc P = p^a/sum(p^a)
        probs  = prios ** self.alpha
        P = probs/probs.sum()

        #gets the indices depending on the probability p
        indices = np.random.choice(N, batch_size, p=P)
        samples = [self.buffer[idx] for idx in indices]

        beta = self.beta_by_frame(self.frame)
        self.frame+=1

        #Compute importance-sampling weight
        weights  = (N * P[indices]) ** (-beta)
        # normalize weights
        weights /= weights.max()
        weights  = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones = zip(*samples)
        return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = abs(prio)

    def __len__(self):
        return len(self.buffer)




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, help='name of the experiment')
    parser.add_argument('--env-name', type=str, default='carla', help='name of the simulation environment')
    # parser.add_argument('--learning-rate', type=float, default=PPO_LEARNING_RATE, help='learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=0, help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=3e6,help='total timesteps of the experiment')
    parser.add_argument('--test-timesteps', type=int, default=1e4, help='timesteps to test our model')
    parser.add_argument('--episode-length', type=int, default=7500, help='max timesteps in an episode')
    parser.add_argument('--train', default=False, type=bool, help='is it training?')
    parser.add_argument('--town', type=str, default="Town02", help='which town do you like?')
    #parser.add_argument('--test-mode', type=bool, default=True, help='eval model')


    parser.add_argument('--load-checkpoint', type=str, default="./sac_232318")

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
    state_size = 69
    action_size = env.action_space.shape[0]
    action_high = env.action_space.high[0]
    action_low = env.action_space.low[0]
    #encode = EncodeState(LATENT_DIM)


    # ========================================================================
    #                           ALGORITHM
    # ========================================================================
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=args.seed, action_prior="normal") #"normal"

    # if checkpoint_load:
    #     sac_agent.restore('path/to/checkpoint')

    try:
        time.sleep(0.5)

        if train:
            print("Train start\n")
            agent.load(args.load_checkpoint)
            agent.actor_local.train()
            agent.critic1.train()
            #agent.critic1_target.train()
            agent.critic2.train()
            #agent.critic2_target.train()


            # agent = agent.actor_local.load_state_dict(torch.load(args.load_checkpoint))
            # Training
            while timestep < total_timesteps:
                observation = env.reset()
                image_obs = torch.tensor(observation['vae_latent'], dtype=torch.float).to(device)

                navigation_obs = torch.tensor(observation['vehicle_measures'], dtype=torch.float).to(device)
                observation = torch.cat((image_obs.view(-1), navigation_obs), -1)
                #observation = encode.process(observation)
                #action = agent.act(observation)


                current_ep_reward = 0
                t1 = datetime.now()

                for t in range(args.episode_length):
                    timestep += 1
                    action = agent.act(observation)

                    action_v = action.numpy()
                    action_v = np.clip(action_v * action_high, action_low, action_high)
                    # print('\n------\n')
                    # print(action_v)
                    # print('\n------\n')

                    next_observation, reward, done, info = env.step(action_v)
                    image_obs = torch.tensor(next_observation['vae_latent'], dtype=torch.float).to(device)

                    navigation_obs = torch.tensor(next_observation['vehicle_measures'], dtype=torch.float).to(device)

                    next_observation = torch.cat((image_obs.view(-1), navigation_obs), -1)

                    agent.step(observation,action,reward,next_observation,done,timestep,episode)

                    observation = next_observation
                    current_ep_reward += reward

                    if timestep == total_timesteps - 1:
                        agent.save(f"./sac_{timestep}")
                        #torch.save(agent.actor_local.state_dict(),f"./sac_actor_{timestep}.pth")
                        #torch.save(agent.critic_local.state_dict(), f"./sac_critic{timestep}.pth")

                        #sac_agent.save('path/to/checkpoint')

                    if done:
                        episode += 1
                        t2 = datetime.now()
                        t3 = t2 - t1
                        episodic_length.append(abs(t3.total_seconds()))
                        break


                scores.append(current_ep_reward)
                cumulative_score = np.mean(scores)

                print('Episode: {}'.format(episode), ', Timestep: {}'.format(timestep),
                      ', Reward:  {:.2f}'.format(current_ep_reward),
                      ', Average Reward:  {:.2f}'.format(cumulative_score))

                if episode % 5 == 0:
                    writer.add_scalar("Episodic Reward/episode", scores[-1], episode)
                    writer.add_scalar("Cumulative Reward/info", cumulative_score, episode)
                    writer.add_scalar("Cumulative Reward/(t)", cumulative_score, timestep)
                    writer.add_scalar("Average Episodic Reward/info", np.mean(scores[-5:]), episode)
                    writer.add_scalar("Average Reward/(t)", np.mean(scores[-5:]), timestep)
                    writer.add_scalar("Episode Length (s)/info", np.mean(episodic_length), episode)
                    writer.add_scalar("Reward/(t)", current_ep_reward, timestep)



                    episodic_length = list()


                if episode % 100 == 0:
                    agent.save(f"./sac_{timestep}")
                    #torch.save(agent.actor_local.state_dict(), f"./sac_actor_{timestep}.pth")
                    #torch.save(agent.critic_local.state_dict(), f"./sac_critic{timestep}.pth")

            print("Terminating the run.")
            sys.exit()
        else:
            print("Start Eval mode!\n")
            agent.load(args.load_checkpoint)
            agent.actor_local.eval()

            agent.critic1.eval()
            #agent.critic1_target.train()
            agent.critic2.eval()
            # Testing
            while timestep < args.test_timesteps:

                observation = env.reset()
                image_obs = torch.tensor(observation['vae_latent'], dtype=torch.float).to(device)

                navigation_obs = torch.tensor(observation['vehicle_measures'], dtype=torch.float).to(device)
                observation = torch.cat((image_obs.view(-1), navigation_obs), -1)
                #observation = encode.process(observation)

                current_ep_reward = 0
                t1 = datetime.now()

                for t in range(args.episode_length):

                    action = agent.act(observation)

                    action_v = action.numpy()
                    action_v = np.clip(action_v * action_high, action_low, action_high)


                    next_observation, reward, done, info = env.step(action_v)
                    image_obs = torch.tensor(next_observation['vae_latent'], dtype=torch.float).to(device)

                    navigation_obs = torch.tensor(next_observation['vehicle_measures'], dtype=torch.float).to(device)

                    next_observation = torch.cat((image_obs.view(-1), navigation_obs), -1)
                    observation = next_observation

                    #observation = encode.process(observation)

                    timestep += 1
                    current_ep_reward += reward

                    if done:
                        episode += 1
                        t2 = datetime.now()
                        t3 = t2 - t1
                        episodic_length.append(abs(t3.total_seconds()))
                        break

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

                episodic_length = list()

            print("Terminating the run.")
            sys.exit()

    finally:
        agent.save(f"./sac_{timestep}")
        #torch.save(agent.actor_local.state_dict(), f"./sac_3_final.pth")
        sys.exit()



if __name__ == "__main__":
    try:

        runner()
        print("im in 22 ")
    except KeyboardInterrupt:
        print("runner falied")
        sys.exit()
    finally:
        print('\nExit')

