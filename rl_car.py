### Generals Import ###
import gym
import random
import time
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image

from collections import namedtuple, deque
from itertools import count

### Pytorch Import ###
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

### Setups ###
# Setup matplotlib
is_python = "inline" in matplotlib.get_backend()
if is_python:
    from IPython import display

# Setup Env
env = gym.make("CarRacing-v2", continuous= False, domain_randomize= True)

# Setup GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replay Mem
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

### Class ###
# ReplayMem
class ReplayMem(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# DQN Agent
class DQN(nn.Module):
    def __init__(self, n_obs, n_actions):
        super(DQN, self).__init__()

        # Input
        self.input_layer = nn.Conv2d(n_obs, 6, kernel_size= 4, stride= 4)

        # Hidden
        self.conv2d = nn.Conv2d(6, 24, kernel_size= 4, stride= 1)
        self.max_pooling = nn.MaxPool2d(kernel_size= 2)
        self.flatten = nn.Flatten()

        self.linear = nn.Linear(18, 432) # Output of Conv2D give input
        self.linear_1 = nn.Linear(432, 1_000) # Output of Conv2D give input
        self.linear_2 = nn.Linear(1_000, 256)

        # Output
        self.output_layer = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.input_layer(x))

        x = F.relu(self.conv2d(x))
        x = F.relu(self.linear(x))
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))

        return F.softmax(self.output_layer(x))


# Class Config
class Configuration:
    def __init__(self):
        self.BATCH_SIZE = 64  # nb of transition for the replay buffer
        self.GAMMA = 0.99  # discount factor
        self.EPS_START = 0.9  # strating epsilon
        self.EPS_END = 0.05  # ending epsilon
        self.EPS_DECAY = 1000  # rate of expo decay for the epsilon
        self.TAU = 0.005  # update rate for the target
        self.LR = 1e-4  # learning rate for optimizer
        self.AMSGRAD = True

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

# Preprocessing Image
# HandMade
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def equalizer(img_array):
    img_copy = np.copy(img_array)

    depth = 256
    height, width = img_copy.shape

    incidence = np.zeros((img_copy.size))
    occurrence = np.zeros(depth)

    k = 0
    for i in range(height):
        for j in range(width):
            incidence[k] = img_copy[i, j]
            k = k + 1
    count = 0

    for j in range(len(incidence)):
        occurrence[int(incidence[j])] += 1

    cummul = np.zeros(depth)
    count = 0
    for i in range(depth):
        count += occurrence[i]
        cummul[i] = count

    const = (depth-1) / (height * width)

    for i in range(depth):
        cummul[i] = round(cummul[i] * const, 0)

    equalizer = np.copy(img_array)
    i1, j1 = np.nonzero(equalizer)

    for j in range(len(i1)):
        equalizer[i1[j], j1[j]] = cummul[int(equalizer[i1[j], j1[j]])]

    equalizer = equalizer.astype("uint8")
    equalizer = np.stack((equalizer,) * 3, axis=-1)
    return equalizer

def preproc_img(frame):
    # Cropping 84 x 84
    frame_crop = frame[:84, 3:87]

    # Greyscale
    gray_frame = rgb2gray(frame_crop)

    # Equalize
    img = equalizer(gray_frame)

    # To numpy
    img_array = np.asarray(img)
    return img_array

# Pytorch Preproc
def preprocessing_img(frame):
    # Convert to PIL
    img = TF.to_pil_image(frame)
    # Grayscale
    img_gray = TF.to_grayscale(img)
    # Crop
    img_crop = TF.crop(img_gray, 0, 3, 84, 84)
    # Equalize
    equalize_img = TF.equalize(img_crop)
    return TF.to_tensor(equalize_img)

### Logic ###

# Get actions space
n_actions = env.action_space.n

# Get the state of obs
state, info = env.reset()
state = preprocessing_img(state)
n_obs = len(state)

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
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor(
            [[env.action_space.sample()]], device=device, dtype=torch.long
        )


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
    )
    non_final_next_state = torch.cat([x for x in batch.next_state if x is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s{t+1}) for all next states
    next_state_values = torch.zeros(CFG.BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_state).max(1)[0]

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * CFG.GAMMA) + reward_batch

    # Compute HuberLoss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)

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

            for t in count():
                state = preprocessing_img(state)
                action = select_action(state=state)
                obs, reward, terminated, truncated, _ = env.step(action.item())

                reward = torch.tensor([reward], device=device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(
                        obs, dtype=torch.float32, device=device
                    )

                # Store transition in mem
                memory.push(state, action, next_state, reward)

                # Move to next state
                state = next_state

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

    plot_durations(show_result=False)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main_loop()
