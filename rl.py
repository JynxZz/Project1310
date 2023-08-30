import gym
import random
import time
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from collections import namedtuple, deque
from itertools import count


# PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Setup matplotlib
is_python = "inline" in matplotlib.get_backend()
if is_python:
    from IPython import display

# Setup Env
# env = gym.make("LunarLander-v2", render_mode="human")
env = gym.make("BipedalWalker-v3", hardcore=True, render_mode="human")

# Setup GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replay Mem
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMem(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# DQN
class DQN(nn.Module):
    def __init__(self, n_obs, n_actions):
        super(DQN, self).__init__()

        # Input
        self.input_layer = nn.Linear(n_obs, 128)

        # Hidden
        self.hidden_layer_1 = nn.Linear(128, 128)
        self.hidden_layer_2 = nn.Linear(128, 128)

        # Output
        self.output_layer = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden_layer_1(x))
        x = F.relu(self.hidden_layer_2(x))

        return self.output_layer(x)

# Config class
class Configuration:
    def __init__(self):
        self.BATCH_SIZE = 128  # nb of transition for the replay buffer
        self.GAMMA = 0.99  # discount factor
        self.EPS_START = 0.9  # strating epsilon
        self.EPS_END = 0.05  # ending epsilon
        self.EPS_DECAY = 1000  # rate of expo decay for the epsilon
        self.TAU = 0.005  # update rate for the target
        self.LR = 1e-4  # learning rate for optimizer
        self.AMSGRAD = True  # stochastic optimisation ofr the optimizer
        self.PATH = "model_lunar_gen_{}.pt"


CFG = Configuration()

# Timer Class
class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""
class Timer():
    def __init__(self):
        self._start_time = None

    def start(self):
        if self._start_time is not None:
            raise TimerError("Timer is still running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        if self._start_time is None:
            raise TimeoutError("Timer is not running. Use .stop() to stop it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"Elpased time : {elapsed_time:0.4f}")





# Get actions space
# n_actions = env.action_space.n
n_actions = env.action_space.shape[0]

# Get the state of obs
state, info = env.reset()
# n_obs = len(state)
n_obs = env.observation_space.shape[0]


# Set DQN policy, DQN target, optimizer & memory, nb of generations
policy_net = DQN(n_obs, n_actions).to(device)
target_net = DQN(n_obs, n_actions).to(device)

target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=CFG.LR, amsgrad=CFG.AMSGRAD)

memory = ReplayMem(10_000)


# Method to selct action
gens = 0


def select_action(state):
    global gens
    sample = random.random()

    epsilon_threshold = CFG.EPS_END + (CFG.EPS_START - CFG.EPS_END) * np.exp(
        -1 * gens / CFG.EPS_DECAY
    )
    gens += 1

    if sample > epsilon_threshold:
        with torch.no_grad():
            return policy_net(torch.tensor(state, dtype=torch.float32))
    else:
        return env.action_space.sample()
        # return torch.tensor(
        #     env.action_space.sample(), device=device, dtype=torch.float32
        # )


# Method to optimize model
def optimize_model():
    if len(memory) < CFG.BATCH_SIZE:
        return

    transitions = memory.sample(CFG.BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute state before the ending of sim
    non_final_mask = torch.tensor(
        tuple(map(lambda x: x is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    ).unsqueeze(1)

    non_final_next_state = torch.tensor(batch.next_state)

    state_batch = torch.tensor(batch.state)
    #action_batch = torch.tensor(batch.action)
    reward_batch = torch.tensor(batch.reward)
    #print(reward_batch.shape)

    print('b')# Compute Q(s_t, a)
    state_action_values = policy_net(state_batch)
    print('a')
    # Compute V(s{t+1}) for all next states
    next_state_values = torch.zeros(CFG.BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values = target_net(non_final_next_state)#*(1-non_final_mask)

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * CFG.GAMMA) + reward_batch.unsqueeze(1)

    # Compute HuberLoss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize model
    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


# Method to plot results
gen_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(gen_durations, dtype=torch.float)

    if show_result:
        plt.title("Results")
    else:
        plt.clf()
        plt.title("Training ...")

    plt.xlabel("Gen")
    plt.ylabel("Duration")

    plt.plot(durations_t.numpy())

    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)
    if is_python:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def save_model(model, path, gen):
    torch.save(model.state_dict(), path.format(gen))

def load_model(path, gen):
    model = DQN(n_obs, n_actions).to(device)
    model.load_state_dict(torch.load(path.format(gen)))
    return model

def main_loop(path= None, gen= None, new_model = True):
    if torch.cuda.is_available():
        n_gens = 1_000
    else:
        n_gens = 100
    print(f"Setup Env with {n_gens} episodes \n Ready to train ... ")

    # Setup Timer
    timer = Timer()
    timer.start()

    if not new_model:
        pass
    else:
        for i_gen in range(n_gens):
            state, info = env.reset()
            state = state

            for t in count():
                action = select_action(state=state)

                obs, reward, terminated, truncated, _ = env.step(action)

                reward = torch.tensor([reward], device=device)
                done = terminated or truncated

                # if terminated:
                #     next_state = None
                # else:
                #     next_state = torch.tensor(
                #         obs, dtype=torch.float32, device=device
                #     )

                # Store transition in mem
                memory.push(state, action, obs, reward)

                # Move to next state
                state = obs

                optimize_model()

                # Softupdate the target
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()

                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[
                        key
                    ] * CFG.TAU + target_net_state_dict[key] * (1 - CFG.TAU)

                target_net.load_state_dict(target_net_state_dict)

                if done:
                    env.reset()
                    gen_durations.append(t + 1)
                    plot_durations()
                    # env.close()
                    break

    print("Completed !!!")
    timer.stop()

    plot_durations(show_result=True)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main_loop()
