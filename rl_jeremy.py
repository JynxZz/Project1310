import gym
import collections
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

HIDDEN_SIZE = 256
HIDDEN_DEPTH = 2

GAMMA = 0.99
BATCH_SIZE = 256
REPLAY_SIZE = 1000
LEARNING_RATE = 1e-3
SYNC_TARGET = 50
REPLAY_START_SIZE = 1000

EPSILON_DECAY_LAST_FRAME = 10**7
EPSILON_START = 1.0
EPSILON_FINAL = 0.0001

N_GAMES = 10000

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)

class DQN(nn.Module) :
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.inp = nn.Linear(input_shape, HIDDEN_SIZE*(HIDDEN_DEPTH+1))
        self.hidden = [nn.Linear((i+1)*HIDDEN_SIZE , i*HIDDEN_SIZE ) for i in range(HIDDEN_DEPTH,0,-1)]
        self.out = nn.Linear(HIDDEN_SIZE, n_actions)

    def forward(self, x):
        x = self.inp(x)
        for i in range(HIDDEN_DEPTH):
            x = self.hidden[i](x)
            x = F.relu(x)
        return self.out(x)

class Agent :
    def __init__(self, env, buffer):
        self.env = env
        self.buffer = buffer
        self.reset()
        self.net = DQN(env.observation_space.shape[0], env.action_space.shape[0])
        self.target = DQN(env.observation_space.shape[0], env.action_space.shape[0])
        self.optimizer = optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        self.epsilon = EPSILON_START
        self.target.load_state_dict(self.net.state_dict())
        self.target.eval()

    def reset(self):
        self.state = env.reset()[0]
        self.total_reward = 0.0

    def step(self, state, action, reward, new_state, done):

        exp = Experience(state, action, reward, done, new_state)
        self.buffer.append(exp)


        if len(self.buffer) > BATCH_SIZE:
            self.learn()
            self.buffer.buffer.clear()

    def learn(self):

        states, actions, rewards, dones, new_states = buffer.sample(BATCH_SIZE)

        states_v = torch.tensor(states, dtype=torch.float32)
        next_states_v = torch.tensor(new_states, dtype=torch.float32)
        rewards_v = torch.tensor(rewards, dtype=torch.float32)
        done_mask = torch.ByteTensor(dones)


        state_action_values = self.net(states_v)
        next_state_values = self.target(next_states_v)
        next_state_values = next_state_values.detach()
        expected_state_action_values = next_state_values * GAMMA * (1-done_mask.unsqueeze(1))+ rewards_v.unsqueeze(1)

        loss = F.mse_loss(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.net.parameters():
            # Clip the error term to be between -1 and 1
            param.grad.data.clamp_(-1,1)
        self.optimizer.step()
        print("Peach peach peach peachonette")

    def choose(self, state):
        if np.random.random() < self.epsilon:
            action = env.action_space.sample()
            #print('random')
        else:
            state_v = torch.tensor(state, dtype = torch.float32)
            action = self.net(state_v).detach()
        if self.epsilon > EPSILON_FINAL :
            self.epsilon -= (EPSILON_START-EPSILON_FINAL)/EPSILON_DECAY_LAST_FRAME
        return action










if __name__ == "__main__":
    env = gym.make("HumanoidStandup-v4", render_mode="human")
    buffer = ExperienceBuffer(REPLAY_SIZE)

    agent = Agent(env, buffer)





    max_score = -10000
    max_game = 0
    scores = []

    for game in range(N_GAMES):
        env = gym.make("HumanoidStandup-v4", render_mode="human")
        print(game)
        agent.buffer.buffer.clear()
        done = False
        score = 0
        observation = env.reset()[0]
        game_actions = [] # actions taken during this game
        i=0
        while (not done) and (i<2000):
            i+=1
            # Depending on probability of EPSILON, either select a random action or select an action based on the Bellman Equation
            action = agent.choose(observation)

            # Execute the action in env and observe reward and next state
            next_observation, reward, done, _, _ = env.step(action)
            env.render()

            # Stores experiences and learns
            agent.step(observation, action, reward, next_observation, done)

            # Update variables each step
            game_actions.append(action)
            score += float(reward)
            observation = next_observation
            print(i)
            if i ==2000 :
                print("wtf step bro")
                env.close()


        # Every TARGET_UPDATE games, reset the target model to the main model
        if game % SYNC_TARGET == 0:
            agent.target.load_state_dict(agent.net.state_dict())

        # Update variables each game
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        print(f'En attendant : {score}, eps : {agent.epsilon}')

        # Checks if the max score has been beaten
        if score > max_score:
            max_score = score
            max_game = game
            print(max_score)
    plt.figure()
    plt.plot(scores)
    plt.show()
